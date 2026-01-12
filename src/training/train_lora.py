"""LoRA fine-tuning script for SDXL."""

import os
import argparse
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model, TaskType, set_peft_model_state_dict
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
import sys
from typing import Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.data_utils import MultiImageDataset, load_config

logger = get_logger(__name__)


def collate_fn(examples):
    """Collate function for DataLoader."""
    pixel_values = [example["pixel_values"] for example in examples if example.get("pixel_values") is not None]
    prompts = [example["prompt"] for example in examples]
    
    return {
        "pixel_values": pixel_values,
        "prompts": prompts,
    }


def encode_prompt(prompt_batch, text_encoders, tokenizers, device):
    """Encode prompts using SDXL's dual text encoders."""
    prompt_embeds_list = []
    pooled_prompt_embeds_list = []
    
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompt_batch,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        
        with torch.no_grad():
            prompt_embeds = text_encoder(
                text_input_ids,
                output_hidden_states=True,
            )
            # SDXL uses the last hidden state and pooler output
            pooled_prompt_embeds = prompt_embeds.pooler_output
            prompt_embeds = prompt_embeds.hidden_states[-2]
            prompt_embeds_list.append(prompt_embeds)
            pooled_prompt_embeds_list.append(pooled_prompt_embeds)
    
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.concat(pooled_prompt_embeds_list, dim=-1)
    return prompt_embeds, pooled_prompt_embeds


def main():
    parser = argparse.ArgumentParser(description="Train LoRA for SDXL")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                       help="Path to training config file")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Path to training data directory")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)
    
    # Output directory
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize accelerator
    project_config = ProjectConfiguration(
        project_dir=str(output_dir),
        logging_dir=str(output_dir / "logs"),
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        mixed_precision=config["training"]["mixed_precision"],
        log_with="tensorboard" if os.getenv("USE_TENSORBOARD") else None,
        project_config=project_config,
    )
    
    if accelerator.is_local_main_process:
        accelerator.init_trackers("sdxl_lora_training")
    
    # Load base model
    logger.info(f"Loading base model: {config['model']['base_model']}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config["model"]["base_model"],
        torch_dtype=torch.float16 if config["training"]["mixed_precision"] == "fp16" else torch.float32,
        variant="fp16" if config["training"]["mixed_precision"] == "fp16" else None,
    )
    
    # Setup LoRA
    lora_config = LoraConfig(
        r=config["lora"]["rank"],
        lora_alpha=config["lora"]["alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["dropout"],
        bias="none",
        task_type=TaskType.TEXT_TO_IMAGE,
    )
    
    # Apply LoRA to UNet
    unet = pipe.unet
    unet = get_peft_model(unet, lora_config)
    
    # Enable gradient checkpointing
    if config["training"]["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
    
    # Load components
    tokenizer_one = pipe.tokenizer
    tokenizer_two = pipe.tokenizer_2
    text_encoder_one = pipe.text_encoder
    text_encoder_two = pipe.text_encoder_2
    vae = pipe.vae
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    
    # Freeze non-trainable parameters
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    vae.requires_grad_(False)
    
    # Prepare optimizer
    if config["optimizer"]["type"] == "adamw8bit":
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
        except ImportError:
            logger.warning("bitsandbytes not available, using standard AdamW")
            optimizer_class = torch.optim.AdamW
    else:
        optimizer_class = torch.optim.AdamW
    
    optimizer = optimizer_class(
        unet.parameters(),
        lr=config["training"]["learning_rate"],
        betas=config["optimizer"]["betas"],
        weight_decay=config["optimizer"]["weight_decay"],
        eps=config["optimizer"]["epsilon"],
    )
    
    # Setup learning rate scheduler
    lr_warmup_steps = config["training"].get("lr_warmup_steps", 0)
    if config["training"]["lr_scheduler"] == "constant":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: 1.0 if step < lr_warmup_steps else 1.0
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["training"]["max_train_steps"]
        )
    
    # Prepare dataset
    logger.info("Preparing dataset...")
    dataset = MultiImageDataset(
        args.data_dir,
        caption_file=config["data"].get("caption_file"),
        resolution=config["training"]["resolution"],
    )
    
    # Split dataset
    validation_split = config["data"].get("validation_split", 0.1)
    dataset_size = len(dataset)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["train_batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )
    
    # Calculate number of training steps
    num_update_steps_per_epoch = len(train_dataloader) // config["training"]["gradient_accumulation_steps"]
    if config["training"].get("max_train_steps"):
        max_train_steps = config["training"]["max_train_steps"]
    else:
        max_train_steps = config["training"]["num_train_epochs"] * num_update_steps_per_epoch
    
    # Prepare with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move text encoders and VAE to device
    text_encoder_one.to(accelerator.device)
    text_encoder_two.to(accelerator.device)
    vae.to(accelerator.device)
    
    # Load checkpoint if resuming
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = Path(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.name.split("-")[1]))
            path = dirs[-1] if dirs else None
        
        if path and path.exists():
            accelerator.load_state(str(path))
            global_step = int(path.name.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            logger.info(f"Resumed from checkpoint {path}")
        else:
            logger.warning(f"Checkpoint {args.resume_from_checkpoint} not found, starting from scratch")
    
    # Training loop
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config['training']['num_train_epochs']}")
    logger.info(f"  Instantaneous batch size per device = {config['training']['train_batch_size']}")
    logger.info(f"  Total train batch size = {config['training']['train_batch_size'] * accelerator.num_processes * config['training']['gradient_accumulation_steps']}")
    logger.info(f"  Gradient Accumulation steps = {config['training']['gradient_accumulation_steps']}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(first_epoch, config["training"]["num_train_epochs"]):
        unet.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latents
                pixel_values = batch["pixel_values"]
                prompts = batch["prompts"]
                
                # Skip if no images
                if not pixel_values:
                    continue
                
                # Convert PIL images to tensors and normalize
                images = []
                for img in pixel_values:
                    if isinstance(img, Image.Image):
                        img = np.array(img).astype(np.float32) / 255.0
                        img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
                        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
                        images.append(img)
                
                if not images:
                    continue
                
                pixel_values = torch.cat(images, dim=0).to(accelerator.device, dtype=vae.dtype)
                
                # Encode images to latents
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
                ).long()
                
                # Add noise to latents
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                
                # Encode prompts
                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        prompts, [text_encoder_one, text_encoder_two], [tokenizer_one, tokenizer_two], accelerator.device
                    )
                
                # Prepare time_ids for SDXL (height, width, crops_coords_top_left, target_size)
                # Default: 1024x1024, no crop, target 1024x1024
                batch_size = latents.shape[0]
                time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]] * batch_size, 
                                       dtype=prompt_embeds.dtype, device=accelerator.device)
                
                # Predict noise
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": time_ids},
                ).sample
                
                # Calculate loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # Backward pass
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss += loss.item()
                
                if accelerator.is_local_main_process:
                    if global_step % config["training"]["logging_steps"] == 0:
                        avg_loss = train_loss / config["training"]["logging_steps"]
                        train_loss = 0.0
                        accelerator.log({"train_loss": avg_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                        logger.info(f"Step {global_step}: Loss = {avg_loss:.4f}, LR = {lr_scheduler.get_last_lr()[0]:.2e}")
                
                if global_step % config["training"]["save_steps"] == 0:
                    if accelerator.is_local_main_process:
                        checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save accelerator state
                        accelerator.save_state(str(checkpoint_dir))
                        
                        # Save LoRA weights
                        unwrapped_unet = accelerator.unwrap_model(unet)
                        unwrapped_unet.save_pretrained(str(checkpoint_dir / "lora_weights"))
                        lora_config.save_pretrained(str(checkpoint_dir / "lora_config"))
                        
                        # Save checkpoint metadata
                        import json
                        metadata = {
                            "global_step": global_step,
                            "epoch": epoch,
                            "loss": avg_loss if train_loss > 0 else None,
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                        }
                        with open(checkpoint_dir / "checkpoint_metadata.json", "w") as f:
                            json.dump(metadata, f, indent=2)
                        
                        logger.info(f"Saved checkpoint to {checkpoint_dir}")
                        
                        # Clean up old checkpoints if save_total_limit is set
                        save_total_limit = config["training"].get("save_total_limit")
                        if save_total_limit and save_total_limit > 0:
                            checkpoints = sorted(
                                [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                                key=lambda x: int(x.name.split("-")[1]),
                                reverse=True
                            )
                            if len(checkpoints) > save_total_limit:
                                for old_checkpoint in checkpoints[save_total_limit:]:
                                    import shutil
                                    shutil.rmtree(old_checkpoint)
                                    logger.info(f"Removed old checkpoint: {old_checkpoint.name}")
            
            if global_step >= max_train_steps:
                break
        
        if global_step >= max_train_steps:
            break
    
    # Save final model
    if accelerator.is_local_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(str(output_dir / "final_lora_weights"))
        lora_config.save_pretrained(str(output_dir / "final_lora_config"))
        logger.info(f"Training completed! Final model saved to {output_dir / 'final_lora_weights'}")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
