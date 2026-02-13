import os
# Hugging Face Mirror Configuration
# Option 1: hf-mirror.com (Available in some regions)
# Option 2: Use ModelScope as an alternative
USE_MODELSCOPE = False  # Set to True for ModelScope, False for HuggingFace

import torch
import torch.nn as nn
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel
from peft import LoraConfig, get_peft_model
import lpips
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
from diffusers.models.attention_processor import Attention, AttnProcessor


class AttentionStoreProcessor(AttnProcessor):
    """Attention Processor used to store cross-attention maps for steering"""
    def __init__(self, model=None):
        super().__init__()
        self.model = model  # Reference to the main model instance

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        query = attn.to_q(hidden_states)
        
        is_cross_attention = encoder_hidden_states is not None
        
        if not is_cross_attention:
            # For Self-Attention, use standard processing
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        else:
            # For Cross-Attention, we need to capture and store the maps
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        # Calculate attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores * attn.scale
        
        # Apply softmax to get attention probabilities
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        
        # Store cross-attention probabilities in the model instance
        if is_cross_attention and self.model is not None:
            # Get the name of the current block
            for name, module in self.model.pipeline.unet.named_modules():
                if module == attn and "attn2" in name and "up_blocks" in name:
                    try:
                        num_heads = attn.heads
                        total_elements = attention_probs.numel()
                        query_len = hidden_states.shape[1]
                        
                        # Key length is usually 77 (CLIP text encoder output length)
                        key_len = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else query_len
                        
                        # Safe Reshape - verify shape compatibility
                        expected_size = batch_size * num_heads * query_len * key_len
                        if total_elements == expected_size:
                            reshaped_probs = attention_probs.reshape(
                                batch_size, 
                                num_heads, 
                                query_len,
                                key_len
                            )
                            if not hasattr(self.model, "attention_maps"):
                                self.model.attention_maps = {}
                            self.model.attention_maps[name] = reshaped_probs.detach().clone()
                        else:
                            print(f"Warning: Attention shape mismatch - Elements: {total_elements}, Expected: {expected_size}")
                    except Exception as e:
                        print(f"Attention Map Processing Error: {e}, Shape: {attention_probs.shape}")
                    break
        
        hidden_states = torch.matmul(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states

class DefectFillModel(nn.Module):
    def __init__(self, device="cuda", lora_rank=8, lora_alpha=16, seed=42, placeholder_token="<defect>"):
        super().__init__()
        torch.manual_seed(seed)
        self.device = device
        
        # Base Model ID
        hf_model_id = "sd2-community/stable-diffusion-2-inpainting"
        
        # Select model source based on configuration
        if USE_MODELSCOPE:
            try:
                from modelscope import snapshot_download
                print(f"[ModelScope] Downloading model: {hf_model_id}")
                local_model_path = snapshot_download(hf_model_id)
                print(f"[ModelScope] Model downloaded to: {local_model_path}")
                
                self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    local_model_path,
                    torch_dtype=torch.float16
                ).to(device)
                
                self.scheduler = DDIMScheduler.from_pretrained(
                    local_model_path,
                    subfolder="scheduler"
                )
            except ImportError:
                print("[Warning] modelscope not installed. Try: pip install modelscope")
                print("[Info] Attempting HuggingFace fallback...")
                self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    hf_model_id, torch_dtype=torch.float16
                ).to(device)
                self.scheduler = DDIMScheduler.from_pretrained(hf_model_id, subfolder="scheduler")
        else:
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                hf_model_id, torch_dtype=torch.float16
            ).to(device)
            self.scheduler = DDIMScheduler.from_pretrained(hf_model_id, subfolder="scheduler")
        
        self.pipeline.set_progress_bar_config(disable=True)
        self.scheduler.set_timesteps(30)
        
        # ========== Textual Inversion: Add learnable defect token [V*] ==========
        self.placeholder_token = placeholder_token
        
        # Add new token to tokenizer
        num_added_tokens = self.pipeline.tokenizer.add_tokens([self.placeholder_token])
        if num_added_tokens == 0:
            print(f"[Warning] Token {self.placeholder_token} already exists in tokenizer")
        else:
            print(f"[Textual Inversion] Added {num_added_tokens} new token: {self.placeholder_token}")
        
        # Resize text encoder embeddings
        self.pipeline.text_encoder.resize_token_embeddings(len(self.pipeline.tokenizer))
        
        # Get ID for the new token
        self.placeholder_token_id = self.pipeline.tokenizer.convert_tokens_to_ids(self.placeholder_token)
        print(f"[Textual Inversion] placeholder_token_id = {self.placeholder_token_id}")
        
        # Initialize new token with the embedding of "defect"
        initializer_token = "defect"
        initializer_token_ids = self.pipeline.tokenizer.encode(initializer_token, add_special_tokens=False)
        if len(initializer_token_ids) > 0:
            initializer_token_id = initializer_token_ids[0]
            token_embeds = self.pipeline.text_encoder.get_input_embeddings().weight.data
            token_embeds[self.placeholder_token_id] = token_embeds[initializer_token_id].clone()
            print(f"[Textual Inversion] Initialized '{self.placeholder_token}' using '{initializer_token}' (id={initializer_token_id})")
        
        # LoRA Configuration
        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            init_lora_weights="gaussian"
        )
        
        text_encoder_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            init_lora_weights="gaussian"
        )
        
        # Apply LoRA adapters
        self.pipeline.unet = get_peft_model(self.pipeline.unet, unet_lora_config)
        self.pipeline.text_encoder = get_peft_model(self.pipeline.text_encoder, text_encoder_lora_config)
        
        # Freeze VAE parameters
        for param in self.pipeline.vae.parameters():
            param.requires_grad = False
        
        # VGG model for LPIPS loss
        self.lpips_model = lpips.LPIPS(net='vgg', spatial=True).to(device)
        
        self.attention_maps = {}
        self.register_attention_processor()
        self.defect_token_indices = []

    def register_attention_processor(self):
        """Replace standard UNet attention processors with custom ones"""
        self.attention_maps = {}
        for name, module in self.pipeline.unet.named_modules():
            if isinstance(module, Attention) and "attn2" in name:  # Target Cross-Attention only
                module.processor = AttentionStoreProcessor(model=self)

    def get_attention_loss(self, mask_latents: torch.Tensor) -> torch.Tensor:
        """
        Calculates Attention Loss - forces <defect> token attention maps to align with the defect mask.
        """
        if not self.attention_maps:
            return torch.tensor(0.0, device=mask_latents.device)
        
        if len(mask_latents.shape) == 3:
            mask_latents = mask_latents.unsqueeze(1)
        
        batch_size = mask_latents.shape[0]
        attention_loss = torch.tensor(0.0, device=mask_latents.device)
        
        # Use only decoder (up_blocks) attention maps
        decoder_attention_maps = {
            name: attn_map for name, attn_map in self.attention_maps.items()
            if "up_blocks" in name
        }
        
        if not decoder_attention_maps:
            return torch.tensor(0.0, device=mask_latents.device)
        
        for b in range(batch_size):
            token_idx = self.defect_token_indices[b] if b < len(self.defect_token_indices) else -1
            if token_idx < 0:
                continue
            
            mask = mask_latents[b].squeeze(0)  # (H, W)
            resized_attention_maps = []
            
            for name, attn_map in decoder_attention_maps.items():
                try:
                    if b < attn_map.shape[0]:
                        # Average attention across all heads for the specific token
                        defect_attn = attn_map[b, :, :, token_idx].mean(dim=0)
                        
                        seq_len = defect_attn.shape[0]
                        h = int(math.sqrt(seq_len))
                        if h * h == seq_len:
                            defect_attn = defect_attn.reshape(h, h)
                            resized_attn = F.interpolate(
                                defect_attn.unsqueeze(0).unsqueeze(0),
                                size=mask.shape,
                                mode='bilinear',
                                align_corners=False
                            ).squeeze()
                            resized_attention_maps.append(resized_attn)
                except Exception:
                    continue
            
            if resized_attention_maps:
                avg_attn_map = torch.stack(resized_attention_maps).mean(dim=0)
                # L2 Loss: ||AttentionMap - Mask||^2
                sample_loss = F.mse_loss(avg_attn_map, mask)
                attention_loss += sample_loss
        
        return attention_loss / batch_size if batch_size > 0 else attention_loss

    def get_text_embeddings(self, prompts, enable_grad=True):
        """Encodes prompts and locates the precise index of the <defect> token"""
        if not hasattr(self, 'pipeline') or self.pipeline is None:
            raise ValueError("Pipeline not initialized")
            
        if isinstance(prompts, str):
            prompts = [prompts]
        
        text_inputs = self.pipeline.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.pipeline.device)
        
        input_ids = text_inputs.input_ids
        
        # Locate the <defect> token position in each prompt
        self.defect_token_indices = []
        for ids in input_ids:
            positions = (ids == self.placeholder_token_id).nonzero(as_tuple=True)[0]
            self.defect_token_indices.append(positions[0].item() if len(positions) > 0 else -1)
        
        if enable_grad:
            text_embeddings = self.pipeline.text_encoder(input_ids)[0]
        else:
            with torch.no_grad():
                text_embeddings = self.pipeline.text_encoder(input_ids)[0]
            
        return text_embeddings
    
    def forward(
        self,
        noisy_latents: torch.Tensor,
        masked_image_latents: torch.Tensor,
        mask_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Training Forward Pass - Implements 9-channel input.
        Input format: [noisy_latents(4), masked_background(4), mask(1)]
        """
        self.attention_maps = {}
        concat_latents = torch.cat([noisy_latents, masked_image_latents, mask_latents], dim=1)
        
        noise_pred = self.pipeline.unet(
            concat_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample
        
        attention_loss = self.get_attention_loss(mask_latents)
        
        return {
            "noise_pred": noise_pred,
            "attention_loss": attention_loss
        }

    @staticmethod
    def compute_masked_mse(noise_pred: torch.Tensor, noise: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Helper to calculate MSE loss only within the masked area"""
        weighted_loss = mask * ((noise_pred - noise) ** 2)
        return torch.sum(weighted_loss) / (torch.sum(mask) + 1e-8)
    
    def compute_defect_loss(self, noise_pred: torch.Tensor, noise: torch.Tensor, mask_latents: torch.Tensor) -> torch.Tensor:
        """L_def loss: MSE restricted to the defect mask region"""
        return self.compute_masked_mse(noise_pred, noise, mask_latents)
    
    def compute_object_loss(self, noise_pred: torch.Tensor, noise: torch.Tensor, mask_latents: torch.Tensor, alpha: float = 0.3) -> torch.Tensor:
        """L_obj loss: Uses weighted mask M' = M + alpha*(1-M) to preserve object context"""
        weighted_mask = mask_latents + alpha * (1 - mask_latents)
        return self.compute_masked_mse(noise_pred, noise, weighted_mask)
    
    def generate(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Complete Inference Pipeline:
        1. 9-channel input configuration
        2. Classifier-Free Guidance (CFG)
        3. Iterative background preservation: x_t = M * x_t_pred + (1-M) * x_t_background
        """
        device = image.device
        dtype = image.dtype
        batch_size = image.shape[0]
        
        # Normalize image to [-1, 1] if needed
        if image.min() >= 0 and image.max() <= 1:
            image = 2 * image - 1
        
        if len(mask.shape) == 3: mask = mask.unsqueeze(1)
        if mask.max() > 1: mask = mask / 255.0
        
        with torch.no_grad():
            # Encode clean image and create masked background latent b = E(I * (1-M))
            latents_clean = self.pipeline.vae.encode(image).latent_dist.sample()
            latents_clean = latents_clean * self.pipeline.vae.config.scaling_factor
            
            masked_image = image * (1 - mask)
            masked_image_latents = self.pipeline.vae.encode(masked_image).latent_dist.sample()
            masked_image_latents = masked_image_latents * self.pipeline.vae.config.scaling_factor
        
        mask_latents = F.interpolate(mask, size=latents_clean.shape[-2:], mode='nearest')
        
        # Text embeddings for CFG
        text_embeddings = self.get_text_embeddings([prompt] * batch_size, enable_grad=False)
        uncond_embeddings = self.get_text_embeddings([""] * batch_size, enable_grad=False)
        text_embeddings_cfg = torch.cat([uncond_embeddings, text_embeddings])
        
        self.scheduler.set_timesteps(num_inference_steps)
        latents = torch.randn(latents_clean.shape, generator=generator, device=device, dtype=dtype)
        
        # Denoising loop
        for t in self.scheduler.timesteps:
            # Generate background noise for current timestep (for background preservation)
            noise_for_bg = torch.randn(latents_clean.shape, generator=generator, device=device, dtype=dtype)
            latents_background = self.scheduler.add_noise(latents_clean, noise_for_bg, t)
            
            # Prepare inputs for CFG
            latent_input = torch.cat([latents] * 2)
            masked_input = torch.cat([masked_image_latents] * 2)
            mask_input = torch.cat([mask_latents] * 2)
            concat_input = torch.cat([latent_input, masked_input, mask_input], dim=1)
            
            timestep_tensor = torch.tensor([t] * (batch_size * 2), device=device, dtype=torch.long)
            
            noise_pred = self.pipeline.unet(
                concat_input,
                timestep_tensor,
                encoder_hidden_states=text_embeddings_cfg
            ).sample
            
            # Perform CFG
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            # ========== KEY STEP: Iterative Background Preservation ==========
            latents = mask_latents * latents + (1 - mask_latents) * latents_background
        
        # Decode latents to pixels
        latents = latents / self.pipeline.vae.config.scaling_factor
        with torch.no_grad():
            images = self.pipeline.vae.decode(latents).sample
        
        return (images + 1) / 2  # Convert back to [0, 1] range