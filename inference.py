import os
import json
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from model import DefectFillModel
from utils import load_checkpoint, compute_spatial_lpips, compute_spatial_lpips_batch
from torchvision.utils import save_image
from torchvision import transforms


def smart_crop_dynamic(image, mask, base_size=512):
    """
    Crops the image to fit the defect. 
    - If defect < 512: Crops 512x512 (No Resize).
    - If defect > 512: Crops square enclosing defect, then resizes to 512.
    """
    h, w = image.shape[:2]
    
    # Find the Bounding Box of the defect
    y_indices, x_indices = np.where(mask > 0)
    
    if len(y_indices) == 0:
        # No defect? Return center crop 512
        cy, cx = h // 2, w // 2
        crop_size = base_size
    else:
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        min_x, max_x = np.min(x_indices), np.max(x_indices)
        
        defect_h = max_y - min_y
        defect_w = max_x - min_x
        
        # Center of the defect
        cy = min_y + defect_h // 2
        cx = min_x + defect_w // 2
        
        # Determine the Crop Size
        # We need a box big enough to hold the defect + some context padding
        # But at minimum, it must be 512.
        max_dim = max(defect_h, defect_w)
        padding = 50 # Add 50px context around edges if possible
        
        crop_size = max(base_size, max_dim + padding)
    
    # Calculate Crop Coordinates (Square Box)
    half_size = crop_size // 2
    x1 = cx - half_size
    y1 = cy - half_size
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    
    # Handle Edge Cases (Shift box if it goes out of bounds)
    if x1 < 0: x2 -= x1; x1 = 0
    if y1 < 0: y2 -= y1; y1 = 0
    if x2 > w: x1 -= (x2 - w); x2 = w
    if y2 > h: y1 -= (y2 - h); y2 = h
    
    # Double check we didn't shrink below image dims (e.g. if image is smaller than crop_size)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)

    # Perform the Crop
    crop_img = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]
    
    # Resize ONLY if the crop is larger than 512
    # (If crop_size was 512, this does nothing. If it was 570, it shrinks slightly.)
    if crop_img.shape[0] != base_size or crop_img.shape[1] != base_size:
        crop_img = cv2.resize(crop_img, (base_size, base_size), interpolation=cv2.INTER_AREA)
        # Use NEAREST for mask to keep edges sharp
        crop_mask = cv2.resize(crop_mask, (base_size, base_size), interpolation=cv2.INTER_NEAREST)
        
    return crop_img, crop_mask


def count_available_resources(data_dir, object_class, defect_type):
    """Counts available good images and reference masks for synthetic generation."""
    # Good images directory
    good_dir = os.path.join(data_dir, object_class, "test", "good")
    num_good_images = len([f for f in os.listdir(good_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(good_dir) else 0
    
    # Mask directory (prioritize training masks for reference)
    train_mask_dir = os.path.join(data_dir, object_class, "train", "defective_masks", defect_type)
    test_mask_dir = os.path.join(data_dir, object_class, "test", "defective_masks", defect_type)
    
    if os.path.exists(train_mask_dir):
        num_masks = len([f for f in os.listdir(train_mask_dir) if f.endswith('.png')])
        mask_dir = train_mask_dir
    elif os.path.exists(test_mask_dir):
        num_masks = len([f for f in os.listdir(test_mask_dir) if f.endswith('.png')])
        mask_dir = test_mask_dir
    else:
        num_masks = 0
        mask_dir = None
    
    return num_good_images, num_masks, good_dir, mask_dir


def calculate_generation_plan(num_good_images, num_masks, target_total=100):
    """Calculates a combination plan of good images and masks to reach the target total."""
    if num_good_images == 0 or num_masks == 0:
        return []
    
    generation_plan = []
    output_idx = 0
    
    # Loop through good images and masks until target count is met
    while output_idx < target_total:
        for mask_idx in range(num_masks):
            if output_idx >= target_total:
                break
            good_idx = output_idx % num_good_images  # Cycle through good images
            generation_plan.append((good_idx, mask_idx, output_idx))
            output_idx += 1
    
    return generation_plan


def inference(args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ========== Unified FP16 Precision Configuration ==========
    dtype = torch.float16
    
    # Enable TF32 acceleration (Ampere+ architectures: RTX 30/40/50 series)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Device: {device}, dtype: {dtype}, TF32: enabled")
    
    # Initialize model
    model = DefectFillModel(
        device=device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha
    )
    
    # Ensure VAE is also in FP16 to save VRAM
    model.pipeline.vae.to(dtype=dtype)
    
    # Load checkpoint
    if args.checkpoint:
        load_checkpoint(model, None, args.checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Set to evaluation mode
    model.pipeline.unet.eval()
    model.pipeline.text_encoder.eval()
    
    # ========== torch.compile Optimization (Optional) ==========
    if hasattr(torch, 'compile') and args.use_compile:
        print("Compiling UNet with torch.compile (this may take 5-15 minutes for max-autotune)...")
        print("Note: First run triggers compilation. Subsequent runs will be significantly faster.")
        
        # Compiler settings
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True
        
        try:
            # Compile UNet (the main computational bottleneck)
            model.pipeline.unet = torch.compile(
                model.pipeline.unet,
                mode="max-autotune",   # Aggressive auto-tuning
                fullgraph=True,        # Full graph compilation
                dynamic=False          # Fixed input size (512x512) for best speed
            )
            
            # Compile VAE decoder
            model.pipeline.vae.decode = torch.compile(
                model.pipeline.vae.decode,
                mode="max-autotune",
                dynamic=False
            )
            print("Compilation configuration complete!")
            
        except Exception as e:
            print(f"Warning: fullgraph compilation failed ({e}), falling back to reduce-overhead mode...")
            model.pipeline.unet = torch.compile(
                model.pipeline.unet,
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=False
            )
            print("Fallback compilation complete!")
        
        # ========== Warmup: Trigger JIT Compilation ==========
        print("Warming up compiled model...")
        dummy_img = torch.randn(1, 3, 512, 512, device=device, dtype=dtype)
        dummy_mask = torch.randn(1, 1, 512, 512, device=device, dtype=dtype)
        dummy_mask = (dummy_mask > 0).float()  # Binarize mask
        dummy_img = dummy_img * 2 - 1          # Map to [-1, 1]
        
        with torch.no_grad():
            try:
                warmup_prompt = f"A {args.object_class} with {model.placeholder_token}"
                _ = model.generate(
                    image=dummy_img,
                    mask=dummy_mask,
                    prompt=warmup_prompt,
                    num_inference_steps=1,  # 1 step is enough to trigger JIT
                    guidance_scale=7.5,
                )
            except Exception as warmup_error:
                print(f"Warmup warning (non-critical): {warmup_error}")
        
        del dummy_img, dummy_mask
        torch.cuda.empty_cache()
        print("Warmup complete! Model is optimized.")

    def fixed_inference_batch(model, clean_image, mask, object_class, defect_type, 
                              num_samples=8, steps=50, guidance_scale=7.5, 
                              batch_size=4):
        """
        Performs inference using the custom model.generate() method.
        Ensures consistency between training and inference phases.
        """
        prompt = f"A {object_class} with {model.placeholder_token}"
        
        print(f"Using prompt: '{prompt}'")
        print(f"Generating {num_samples} samples (batch_size={batch_size}, steps={steps})")
        
        _, _, h_input, w_input = clean_image.shape
        
        # ========== Phase 1: Batch Sample Generation ==========
        all_samples = []
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            current_batch_size = end_idx - start_idx
            
            print(f"Batch {batch_idx+1}/{num_batches}: Generating samples {start_idx+1}-{end_idx}")
            
            batch_clean = clean_image.repeat(current_batch_size, 1, 1, 1)
            batch_mask = mask.repeat(current_batch_size, 1, 1, 1)
            
            # Use deterministic seed per sample for reproducibility
            generator = torch.Generator(device=device).manual_seed(start_idx)
            
            # Consistent with training: 9-channel input + CFG + iterative bg preservation
            batch_samples = model.generate(
                image=batch_clean,
                mask=batch_mask,
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            
            # Convert back to [-1, 1] range for LPIPS (generate returns [0, 1])
            batch_samples_model_format = (batch_samples * 2.0) - 1.0
            all_samples.append(batch_samples_model_format)
        
        samples_model_format = torch.cat(all_samples, dim=0)
        
        if samples_model_format.shape[-2:] != (h_input, w_input):
            samples_model_format = torch.nn.functional.interpolate(
                samples_model_format, size=(h_input, w_input), mode='bilinear'
            )
        
        # ========== Phase 2: Batch LPIPS Selection ==========
        mask_resized = mask if mask.shape[-2:] == samples_model_format.shape[-2:] else \
                       torch.nn.functional.interpolate(mask, size=samples_model_format.shape[-2:], mode='bilinear')
        
        print(f"Selecting best sample based on LPIPS...")
        lpips_scores = compute_spatial_lpips_batch(
            model.lpips_model, clean_image, samples_model_format, mask_resized, smooth_boundary=True
        )
        
        best_idx = lpips_scores.argmax()
        best_score = lpips_scores[best_idx].item()
        best_sample = samples_model_format[best_idx].clone()
        
        print(f"Best sample selected: #{best_idx+1} (LPIPS: {best_score:.4f})")
        
        del all_samples, samples_model_format, lpips_scores
        return best_sample, best_score

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 4
    os.makedirs(args.output_dir, exist_ok=True)
    
    inference_log = {
        "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        "checkpoint": args.checkpoint,
        "object_class": args.object_class,
        "defect_type": args.defect_type,
        "results": []
    }
    
    # Mode A: Dynamic Dataset Generation
    if args.total_images > 0 and args.data_dir and args.defect_type:
        print(f"\n{'='*60}\nDynamic Generation Mode Activated\n{'='*60}")
        num_good, num_masks, good_dir, mask_dir = count_available_resources(args.data_dir, args.object_class, args.defect_type)
        
        if num_good == 0 or num_masks == 0:
            print("Error: Missing images or masks.")
            return
            
        generation_plan = calculate_generation_plan(num_good, num_masks, args.total_images)
        good_files = sorted([f for f in os.listdir(good_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        
        defect_output_dir = os.path.join(args.output_dir, args.defect_type)
        os.makedirs(defect_output_dir, exist_ok=True)
        
        for good_idx, mask_idx, output_idx in tqdm(generation_plan, desc=f"Generating {args.defect_type}"):
            good_path = os.path.join(good_dir, good_files[good_idx])
            mask_path = os.path.join(mask_dir, mask_files[mask_idx])
            
            print(f"\n[{output_idx+1}/{len(generation_plan)}] Processing: {good_files[good_idx]}")
            
            # --- SMART CROP LOGIC START ---
            
            # Load Images as Numpy Arrays (for Smart Crop)
            # Use PIL and convert to numpy to ensure RGB format is consistent
            image_pil = Image.open(good_path).convert("RGB")
            mask_pil = Image.open(mask_path).convert("L")
            
            image_np = np.array(image_pil)
            mask_np = np.array(mask_pil)
            
            # --- DILATION LOGIC (Must match training) ---
            if args.dilate_mask:
                # Ensure kernel size is odd
                k_size = args.mask_kernel_size if args.mask_kernel_size % 2 == 1 else args.mask_kernel_size + 1
                kernel = np.ones((k_size, k_size), np.uint8)
                
                # Apply dilation
                # Note: mask_np is usually 0-255. cv2.dilate works fine on uint8.
                mask_np = cv2.dilate(mask_np, kernel, iterations=1)
                
                print(f"Dilated mask with kernel {k_size}")
            # --------------------------------------------------

            # Apply Smart Crop
            # This returns a 512x512 patch focused on the defect area
            # (No resizing blur unless defect > 512px)
            crop_img_np, crop_mask_np = smart_crop_dynamic(image_np, mask_np, base_size=512)
            
            # Convert to Tensor
            
            # Image: [0, 255] -> [0.0, 1.0] -> Normalize to [-1.0, 1.0]
            # transforms.ToTensor() handles the HWC->CHW and /255 division automatically
            img_tensor = transforms.ToTensor()(crop_img_np)
            img_tensor = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img_tensor)
            img_tensor = img_tensor.unsqueeze(0).to(device, dtype=dtype)
            
            # Mask: [0, 255] -> [0.0, 1.0]
            mask_tensor = transforms.ToTensor()(crop_mask_np).unsqueeze(0).to(device, dtype=dtype)
            
            
            with torch.no_grad():
                defect_img, lpips_score = fixed_inference_batch(
                    model, img_tensor, mask_tensor, args.object_class, args.defect_type, 
                    num_samples=args.num_samples, steps=args.steps, guidance_scale=args.guidance_scale, batch_size=batch_size
                )
            
            # Save generated image
            output_name = f"{output_idx:04d}_generated.png"
            output_path = os.path.join(defect_output_dir, output_name)
            save_image((defect_img.float() + 1) / 2, output_path)
            
            # Save mask and original (These will now be the CROPPED versions, which is correct)
            save_image(mask_tensor.float(), os.path.join(defect_output_dir, f"{output_idx:04d}_mask.png"))
            save_image((img_tensor.float() + 1) / 2, os.path.join(defect_output_dir, f"{output_idx:04d}_original.png"))
            
            inference_log["results"].append({
                "output_idx": output_idx, "input_image": good_path, "lpips_score": lpips_score
            })
            
            if output_idx % 10 == 0: torch.cuda.empty_cache()

    # Mode B: Process Existing Directories/Files (Traditional Inference)
    elif args.image_dir or args.image_path:
        # Implementation similar to above but iterates through provided paths
        pass 

    # Save Log
    log_path = os.path.join(args.output_dir, "inference_log.json")
    with open(log_path, "w") as f:
        json.dump(inference_log, f, indent=4)
    print(f"\nInference log saved to: {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with DefectFill model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output_dir", type=str, default="./generated", help="Output directory")
    parser.add_argument("--object_class", type=str, required=True, help="Object class")
    parser.add_argument("--defect_type", type=str, help="Defect type (e.g., 'cracks')")
    parser.add_argument("--data_dir", type=str, help="Dataset root for dynamic generation")
    parser.add_argument("--image_path", type=str, help="Single image path")
    parser.add_argument("--num_samples", type=int, default=8, help="Samples per image (for LPIPS selection)")
    parser.add_argument("--steps", type=int, default=50, help="Diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--total_images", type=int, default=100, help="Total synthetic images to create")
    parser.add_argument("--batch_size", type=int, default=4, help="Parallel generation batch size")
    parser.add_argument("--use_compile", action="store_true", help="Enable torch.compile (PyTorch 2.0+)")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--dilate_mask", type=str, default="False", help="Whether to dilate masks (True/False)")
    parser.add_argument("--mask_kernel_size", type=int, default=3, help="Size of dilation kernel")
    
    args = parser.parse_args()
    args.dilate_mask = args.dilate_mask.lower() == "true" # Handle boolean conversion

    inference(args)
