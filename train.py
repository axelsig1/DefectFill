import os
import json
import torch
import argparse
import torch.nn.functional as F
import random
import time
from torch.optim import AdamW
from tqdm import tqdm
from diffusers import DDPMScheduler
from model import DefectFillModel, USE_MODELSCOPE
from data_loader import get_data_loaders
from utils import save_checkpoint, load_checkpoint
# TensorBoard support
from torch.utils.tensorboard import SummaryWriter
import datetime


def generate_seed_from_timestamp():
    """Generates a random seed based on the current timestamp"""
    return int(time.time() * 1000) % (2**31)

def train(args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory structure
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Create log file in the output directory
    log_file_path = os.path.join(args.output_dir, "train_log.txt")
    log_file = open(log_file_path, "a")
    log_file.write(f"\n\n{'='*60}\n")
    log_file.write(f"Training started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Object class: {args.object_class}\n")
    log_file.write(f"Defect type: {args.defect_type if args.defect_type else 'all'}\n")
    log_file.write(f"Config name: {args.config_name}\n")
    log_file.write(f"Lambda defect: {args.lambda_defect}\n")
    log_file.write(f"Lambda obj: {args.lambda_obj}\n")
    log_file.write(f"Lambda attn: {args.lambda_attn}\n")
    log_file.write(f"Alpha (obj branch bg weight): {args.alpha}\n")
    log_file.write(f"Gradient accumulation steps: {args.gradient_accumulation_steps}\n")
    log_file.write(f"Random seed: {args.seed}\n")
    log_file.write(f"{'='*60}\n\n")
    
    # Save training configuration to JSON
    train_config = {
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        "object_class": args.object_class,
        "defect_type": args.defect_type if args.defect_type else "all",
        "config_name": args.config_name,
        "lambda_defect": args.lambda_defect,
        "lambda_obj": args.lambda_obj,
        "lambda_attn": args.lambda_attn,
        "alpha": args.alpha,
        "batch_size": args.batch_size,
        "max_train_steps": args.max_train_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "text_encoder_lr": args.text_encoder_lr,
        "unet_lr": args.unet_lr,
        "lr_warmup_steps": args.lr_warmup_steps,
        "save_steps": args.save_steps,
        "seed": args.seed
    }
    config_path = os.path.join(args.output_dir, "train_config.json")
    with open(config_path, "w") as f:
        json.dump(train_config, f, indent=4)
    print(f"Training config saved to: {config_path}")
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(tensorboard_dir)
    
    # Load data
    train_loader, test_loader = get_data_loaders(
        root_dir=args.data_dir,
        object_class=args.object_class,
        batch_size=args.batch_size,
        defect_type=args.defect_type
    )
    
    # Initialize model
    model = DefectFillModel(
        device=device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        seed=args.seed
    )
    
    # Set up optimizers with specific learning rates for Text Encoder and UNet
    text_encoder_params = [p for n, p in model.pipeline.text_encoder.named_parameters() if "lora" in n]
    unet_params = [p for n, p in model.pipeline.unet.named_parameters() if "lora" in n]
    
    optimizer = AdamW([
        {"params": text_encoder_params, "lr": args.text_encoder_lr},
        {"params": unet_params, "lr": args.unet_lr}
    ])
    
    # Save original LRs for warmup calculation
    base_lrs = [args.text_encoder_lr, args.unet_lr]
    
    # Set up noise scheduler (handling ModelScope vs HuggingFace)
    hf_model_id = "sd2-community/stable-diffusion-2-inpainting"
    if USE_MODELSCOPE:
        try:
            from modelscope import snapshot_download
            print(f"[ModelScope] Downloading scheduler: {hf_model_id}")
            local_model_path = snapshot_download(hf_model_id)
            noise_scheduler = DDPMScheduler.from_pretrained(local_model_path, subfolder="scheduler")
        except ImportError:
            print("[Warning] modelscope not installed, falling back to HuggingFace...")
            noise_scheduler = DDPMScheduler.from_pretrained(hf_model_id, subfolder="scheduler")
    else:
        noise_scheduler = DDPMScheduler.from_pretrained(hf_model_id, subfolder="scheduler")
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(model, optimizer, args.resume_from)
        print(f"Resuming from step {start_step}")
        log_file.write(f"Resuming from step {start_step}\n")
    
    # Set models to training mode
    model.pipeline.unet.train()
    model.pipeline.text_encoder.train()
    
    total_steps = args.max_train_steps
    progress_bar = tqdm(range(start_step, total_steps), desc="Training Progress")
    
    global_step = start_step
    accumulation_step = 0
    
    while global_step < total_steps:
        for batch in train_loader:
            if global_step >= total_steps:
                break
                
            # Move data to device
            images = batch["image"].to(device, dtype=torch.float16)
            masks = batch["mask"].to(device, dtype=torch.float16)
            backgrounds = batch["background"].to(device, dtype=torch.float16)
            adjusted_masks = batch["adjusted_mask"].to(device, dtype=torch.float16)
            is_defect = batch["is_defect"]
            
            # Ensure we only process defective samples
            defect_samples = torch.nonzero(is_defect).squeeze(1)
            if len(defect_samples) == 0:
                continue  # Skip batch if no defects present
                
            # Extract defect-only samples
            defect_images = images[defect_samples]
            defect_masks = masks[defect_samples]
            defect_backgrounds = backgrounds[defect_samples]
            defect_adjusted_masks = adjusted_masks[defect_samples]
            object_classes = [batch["object_class"][i] for i in defect_samples]
            
            # Extract defect type from file path for specific prompting
            defect_types = []
            for i in defect_samples:
                if hasattr(train_loader.dataset, 'images') and i < len(train_loader.dataset.images):
                    img_path = train_loader.dataset.images[i]
                    parts = img_path.split(os.sep)
                    for j, part in enumerate(parts):
                        if part == "defective" and j + 1 < len(parts):
                            defect_types.append(parts[j + 1])
                            break
                    else:
                        defect_types.append("defect")
                else:
                    defect_types.append("defect")
            
            # Learning rate warmup
            if global_step < args.lr_warmup_steps:
                lr_scale = min(1.0, (global_step + 1) / args.lr_warmup_steps)
                for i, param_group in enumerate(optimizer.param_groups):
                    param_group["lr"] = base_lrs[i] * lr_scale
            
            # Reset attention maps
            if hasattr(model, 'attention_maps'):
                model.attention_maps = {}
            
            # ========== PHASE 1: Defect Branch (Defect Texture Learning) ==========
            # Using <defect> learnable token for concept isolation
            defect_prompts = [f"A photo of {model.placeholder_token}" for _ in range(len(defect_samples))]
            text_embeddings = model.get_text_embeddings(defect_prompts, enable_grad=True)
            
            # Latent space processing
            with torch.no_grad():
                latents = model.pipeline.vae.encode(defect_images).latent_dist.sample()
                latents = latents * model.pipeline.vae.config.scaling_factor  # z_0
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)  # z_t
            
            if len(defect_masks.shape) == 3:
                defect_masks = defect_masks.unsqueeze(1)
            
            # Build masked_image_latents: b = E(I * (1-M))
            with torch.no_grad():
                masked_images = defect_images * (1 - defect_masks)
                masked_image_latents = model.pipeline.vae.encode(masked_images).latent_dist.sample()
                masked_image_latents = masked_image_latents * model.pipeline.vae.config.scaling_factor
            
            mask_latents = F.interpolate(defect_masks, size=(latents.shape[2], latents.shape[3]))
            
            # Forward pass (9-channel input)
            outputs = model(
                noisy_latents=noisy_latents,
                masked_image_latents=masked_image_latents,
                mask_latents=mask_latents,
                timesteps=timesteps,
                encoder_hidden_states=text_embeddings
            )
            
            # Compute losses
            noise_pred = outputs["noise_pred"]
            defect_loss = model.compute_defect_loss(noise_pred, noise, mask_latents)
            attention_loss = outputs.get("attention_loss", torch.tensor(0.0, device=device))
            
            # ========== PHASE 2: Object Branch (Object Integrity Learning) ==========
            # Generate random masks for structural context learning
            num_random_boxes = 30 
            random_masks = torch.zeros_like(defect_images[:, :1])
            for i in range(random_masks.shape[0]):
                mask = random_masks[i, 0]
                h, w = mask.shape
                for _ in range(num_random_boxes):
                    min_size, max_size = int(min(h, w) * 0.03), int(min(h, w) * 0.15)
                    rect_h = torch.randint(min_size, max(min_size+1, max_size), (1,)).item()
                    rect_w = torch.randint(min_size, max(min_size+1, max_size), (1,)).item()
                    y = torch.randint(0, max(1, h - rect_h), (1,)).item()
                    x = torch.randint(0, max(1, w - rect_w), (1,)).item()
                    mask[y:y+rect_h, x:x+rect_w] = 1.0
            
            # Prompts linking object class with the defect token
            obj_prompts = [f"A {obj_class} with {model.placeholder_token}" for obj_class in object_classes]
            obj_text_embeddings = model.get_text_embeddings(obj_prompts, enable_grad=True)
            
            # Forward pass for Object Branch
            obj_noise = torch.randn_like(latents)
            obj_timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
            obj_noisy_latents = noise_scheduler.add_noise(latents, obj_noise, obj_timesteps)
            
            with torch.no_grad():
                random_masked_images = defect_images * (1 - random_masks)
                random_masked_image_latents = model.pipeline.vae.encode(random_masked_images).latent_dist.sample()
                random_masked_image_latents = random_masked_image_latents * model.pipeline.vae.config.scaling_factor
            
            random_mask_latents = F.interpolate(random_masks, size=(latents.shape[2], latents.shape[3]))
            
            obj_outputs = model(
                noisy_latents=obj_noisy_latents,
                masked_image_latents=random_masked_image_latents,
                mask_latents=random_mask_latents,
                timesteps=obj_timesteps,
                encoder_hidden_states=obj_text_embeddings
            )
            
            object_loss = model.compute_object_loss(obj_outputs["noise_pred"], obj_noise, random_mask_latents, alpha=args.alpha)
            
            # Total Loss Calculation
            total_loss = args.lambda_defect * defect_loss + args.lambda_obj * object_loss + args.lambda_attn * attention_loss
            total_loss = total_loss / args.gradient_accumulation_steps
            
            # NaN Check
            if torch.isnan(total_loss):
                print(f"Warning: NaN loss detected at step {global_step}")
                log_file.write(f"Warning: NaN loss at step {global_step}\n")
                optimizer.zero_grad()
                continue
            
            total_loss.backward()
            accumulation_step += 1
            
            # Optimization step
            if accumulation_step >= args.gradient_accumulation_steps:
                optimizer.step()
                optimizer.zero_grad()
                accumulation_step = 0
                
                progress_bar.update(1)
                global_step += 1
                
                # Logging and TensorBoard updates
                writer.add_scalar("Loss/Defect", defect_loss.item(), global_step)
                writer.add_scalar("Loss/Object", object_loss.item(), global_step)
                writer.add_scalar("Loss/Attention", attention_loss.item(), global_step)
                writer.add_scalar("Loss/Total", total_loss.item() * args.gradient_accumulation_steps, global_step)
                
                if global_step % 10 == 0:
                    for i, param_group in enumerate(optimizer.param_groups):
                        writer.add_scalar(f"LearningRate/group{i}", param_group["lr"], global_step)
                
                # Periodic checkpointing
                if global_step % args.save_steps == 0 or global_step == total_steps:
                    checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_{global_step}.pt")
                    save_checkpoint(model, optimizer, global_step, checkpoint_path)
                    log_file.write(f"Checkpoint saved at step {global_step}\n")

    # Save final model
    final_checkpoint_path = os.path.join(checkpoints_dir, "checkpoint_final.pt")
    save_checkpoint(model, optimizer, global_step, final_checkpoint_path)
    print(f"Final model saved to: {final_checkpoint_path}")
    
    writer.close()
    log_file.close()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DefectFill model")
    
    # Paths
    parser.add_argument("--data_dir", type=str, required=True, help="Path to MVTec AD dataset")
    parser.add_argument("--object_class", type=str, required=True, help="Object class to train on")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save models")
    
    # Loss Weights
    parser.add_argument("--lambda_defect", type=float, default=0.5, help="Defect loss weight (L_def)")
    parser.add_argument("--lambda_obj", type=float, default=0.2, help="Object integrity loss weight (L_obj)")
    parser.add_argument("--lambda_attn", type=float, default=0.05, help="Attention loss weight (L_attn)")
    parser.add_argument("--alpha", type=float, default=0.3, help="Background weight for object branch")
    
    # Training Config
    parser.add_argument("--config_name", type=str, default="base", help="Experiment name (base/tex/obj)")
    parser.add_argument("--defect_type", type=str, default=None, help="Specific defect type to train on")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--text_encoder_lr", type=float, default=4e-5)
    parser.add_argument("--unet_lr", type=float, default=2e-4)
    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=-1, help="-1 for timestamp-based seed")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    
    args = parser.parse_args()
    
    # Seed setup
    if args.seed == -1:
        args.seed = generate_seed_from_timestamp()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    train(args)