"""
DefectFill Evaluation Module
Includes KID (Kernel Inception Distance) and IC-LPIPS (Inter-image Contextual LPIPS) algorithms.

Metric Descriptions:
- KID: Measures the distribution distance between generated and real images (Quality). Lower is better.
- IC-LPIPS: Measures perceptual differences between generated images (Diversity). Higher is better.
"""

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lpips
from PIL import Image
from torchvision import transforms, models
from datetime import datetime
import argparse
from tqdm import tqdm
from itertools import combinations


class KIDEvaluator:
    """
    Kernel Inception Distance (KID) Evaluator
    
    KID uses a polynomial kernel to calculate Maximum Mean Discrepancy (MMD).
    It is better suited for small sample sizes than FID (MVTec often has only dozens of images per class).
    """
    
    def __init__(self, device="cuda"):
        self.device = device
        
        # Load InceptionV3 model using the pool3 layer features (2048 dimensions)
        self.inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        self.inception.fc = nn.Identity()  # Remove classification head
        self.inception = self.inception.to(device)
        self.inception.eval()
        
        # Standard InceptionV3 input preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def extract_features(self, images):
        """
        Extract InceptionV3 features from images.
        
        Args:
            images: List of PIL Images or paths, or Tensor [N, 3, H, W]
        
        Returns:
            features: [N, 2048] feature vectors
        """
        if isinstance(images, list):
            # Process list of PIL Images or paths
            tensors = []
            for img in images:
                if isinstance(img, str):
                    img = Image.open(img).convert('RGB')
                tensor = self.preprocess(img)
                tensors.append(tensor)
            images = torch.stack(tensors)
        
        images = images.to(self.device)
        
        # Batch processing to avoid VRAM overflow
        batch_size = 32
        features_list = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            feat = self.inception(batch)
            features_list.append(feat.cpu())
        
        return torch.cat(features_list, dim=0)
    
    def polynomial_kernel(self, x, y, degree=3, gamma=None, coef0=1):
        """
        Calculate Polynomial Kernel
        k(x, y) = (gamma * <x, y> + coef0)^degree
        """
        if gamma is None:
            gamma = 1.0 / x.shape[1]
        
        return (gamma * torch.mm(x, y.t()) + coef0) ** degree
    
    def compute_mmd(self, x, y):
        """
        Calculate Maximum Mean Discrepancy (MMD)
        MMD^2 = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]
        """
        k_xx = self.polynomial_kernel(x, x)
        k_yy = self.polynomial_kernel(y, y)
        k_xy = self.polynomial_kernel(x, y)
        
        n = x.shape[0]
        m = y.shape[0]
        
        # Unbiased estimator: remove diagonal elements
        mmd = (k_xx.sum() - k_xx.trace()) / (n * (n - 1))
        mmd += (k_yy.sum() - k_yy.trace()) / (m * (m - 1))
        mmd -= 2 * k_xy.mean()
        
        return mmd
    
    def compute_kid(self, real_images, gen_images, num_subsets=100, subset_size=None):
        """
        Calculate KID score.
        
        Args:
            real_images: Real defect images (list of PIL images or paths)
            gen_images: Generated defect images (list of PIL images or paths)
            num_subsets: Number of subset samplings (for mean and std)
            subset_size: Size of each subset (defaults to min of real/gen counts)
        """
        print("Extracting features from real images...")
        real_features = self.extract_features(real_images)
        print(f"  Real features shape: {real_features.shape}")
        
        print("Extracting features from generated images...")
        gen_features = self.extract_features(gen_images)
        print(f"  Generated features shape: {gen_features.shape}")
        
        if subset_size is None:
            subset_size = min(len(real_features), len(gen_features))
        
        # Compute KID via multiple subset sampling
        kid_scores = []
        for _ in range(num_subsets):
            idx_real = np.random.choice(len(real_features), subset_size, replace=False)
            idx_gen = np.random.choice(len(gen_features), subset_size, replace=False)
            
            mmd = self.compute_mmd(
                real_features[idx_real],
                gen_features[idx_gen]
            )
            kid_scores.append(mmd.item())
        
        return np.mean(kid_scores), np.std(kid_scores)


class ICLPIPSEvaluator:
    """
    Inter-image Contextual LPIPS (IC-LPIPS) Evaluator
    
    Calculates perceptual differences between generated images to evaluate diversity.
    Larger IC-LPIPS indicates higher generation diversity.
    """
    
    def __init__(self, net='vgg', device="cuda"):
        self.device = device
        # Use standard LPIPS (non-spatial)
        self.lpips_net = lpips.LPIPS(net=net, spatial=False).to(device)
        self.lpips_net.eval()
        
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def _load_image(self, img):
        """Load and preprocess image"""
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        if isinstance(img, Image.Image):
            img = self.preprocess(img)
        return img.to(self.device)
    
    @torch.no_grad()
    def compute_pairwise_lpips(self, img1, img2):
        """Calculate LPIPS score between two images."""
        img1_tensor = self._load_image(img1).unsqueeze(0)
        img2_tensor = self._load_image(img2).unsqueeze(0)
        
        lpips_score = self.lpips_net(img1_tensor, img2_tensor)
        return lpips_score.item()
    
    @torch.no_grad()
    def compute_ic_lpips(self, generated_images, max_pairs=1000):
        """
        Calculate IC-LPIPS score for a set of generated images.
        Computes the mean LPIPS distance across pairs of generated images.
        """
        n_images = len(generated_images)
        
        if n_images < 2:
            print("Warning: Insufficient images to calculate IC-LPIPS")
            return float('nan'), float('nan')
        
        print(f"Loading {n_images} generated images...")
        image_tensors = []
        for img in tqdm(generated_images, desc="Loading images"):
            img_tensor = self._load_image(img)
            image_tensors.append(img_tensor)
        
        image_batch = torch.stack(image_tensors, dim=0)
        all_pairs = list(combinations(range(n_images), 2))
        n_pairs = len(all_pairs)
        print(f"Total {n_pairs} image pairs found")
        
        if n_pairs > max_pairs:
            print(f"Randomly sampling {max_pairs} pairs for calculation")
            selected_pairs = np.random.choice(n_pairs, max_pairs, replace=False)
            pairs_to_compute = [all_pairs[i] for i in selected_pairs]
        else:
            pairs_to_compute = all_pairs
        
        lpips_scores = []
        batch_size = 32
        for i in tqdm(range(0, len(pairs_to_compute), batch_size), desc="Computing IC-LPIPS"):
            batch_pairs = pairs_to_compute[i:i+batch_size]
            
            img1_batch = torch.stack([image_batch[p[0]] for p in batch_pairs], dim=0)
            img2_batch = torch.stack([image_batch[p[1]] for p in batch_pairs], dim=0)
            
            scores = self.lpips_net(img1_batch, img2_batch)
            lpips_scores.extend(scores.squeeze().cpu().tolist() if len(batch_pairs) > 1 else [scores.item()])
        
        return np.mean(lpips_scores), np.std(lpips_scores)


def collect_generated_images(directory):
    """Collects only files ending with *_generated.png."""
    images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('_generated.png'):
                images.append(os.path.join(root, file))
    return images


def collect_real_defect_images(directory):
    """Collects all real images from directory, excluding masks."""
    images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                if '_mask' not in file:
                    images.append(os.path.join(root, file))
    return images


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\nInitializing evaluators...")
    kid_evaluator = KIDEvaluator(device=device)
    ic_lpips_evaluator = ICLPIPSEvaluator(device=device)
    
    print("\nCollecting images...")
    gen_images_all = collect_generated_images(args.generated_dir)
    print(f"  Generated images: {len(gen_images_all)}")
    
    real_images_all = collect_real_defect_images(args.real_dir)
    print(f"  Real images: {len(real_images_all)}")
    
    print("\nCalculating KID (Quality Assessment)...")
    if len(gen_images_all) > 0 and len(real_images_all) > 0:
        kid_mean, kid_std = kid_evaluator.compute_kid(
            real_images_all, gen_images_all,
            num_subsets=min(100, len(gen_images_all)),
            subset_size=min(len(gen_images_all), len(real_images_all))
        )
        print(f"  KID: {kid_mean:.6f} ± {kid_std:.6f} (Lower is better)")
    else:
        kid_mean, kid_std = float('nan'), float('nan')
        print("  Warning: Insufficient images to calculate KID")
    
    print("\nCalculating IC-LPIPS (Diversity Assessment)...")
    if len(gen_images_all) >= 2:
        ic_lpips_mean, ic_lpips_std = ic_lpips_evaluator.compute_ic_lpips(
            gen_images_all,
            max_pairs=min(1000, len(gen_images_all) * (len(gen_images_all) - 1) // 2)
        )
        print(f"  IC-LPIPS: {ic_lpips_mean:.6f} ± {ic_lpips_std:.6f} (Higher is better)")
    else:
        ic_lpips_mean, ic_lpips_std = float('nan'), float('nan')
        print("  Warning: Insufficient images to calculate IC-LPIPS")
    
    # Save results to CSV
    timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    result_row = [
        timestamp, args.class_name, args.config_name, args.category_type,
        f"{kid_mean:.6f}", f"{kid_std:.6f}", f"{ic_lpips_mean:.6f}", f"{ic_lpips_std:.6f}"
    ]
    
    file_exists = os.path.exists(args.output_csv)
    with open(args.output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'class', 'config', 'category_type', 
                             'KID_mean', 'KID_std', 'IC_LPIPS_mean', 'IC_LPIPS_std'])
        writer.writerow(result_row)
    
    print(f"\nResults appended to: {args.output_csv}")
    return {'kid_mean': kid_mean, 'ic_lpips_mean': ic_lpips_mean}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DefectFill Evaluation Module")
    parser.add_argument("--generated_dir", type=str, required=True, help="Generated images directory")
    parser.add_argument("--real_dir", type=str, required=True, help="Real defect images directory")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV path")
    parser.add_argument("--class_name", type=str, required=True, help="Class name")
    parser.add_argument("--config_name", type=str, required=True, help="Config name")
    parser.add_argument("--category_type", type=str, default="unknown", choices=["object", "texture", "unknown"])
    
    args = parser.parse_args()
    evaluate(args)