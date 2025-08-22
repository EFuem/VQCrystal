#!/usr/bin/env python3
"""
Simple training script for CrystalVAE using PyTorch Lightning
"""

import argparse
from pathlib import Path
import yaml
from main import CrystalVAELightningModule, set_seed
from data import CIFData, collate_pool, get_train_val_test_loader
from model.model2 import GraphVAE
from utils import FocalLoss
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger


def main():
    parser = argparse.ArgumentParser(description="Train CrystalVAE with PyTorch Lightning")
    parser.add_argument("--data_path", default="./data/mp_20", type=str, help="Path to dataset")
    parser.add_argument("--config_path", default="./config/config_mp_20.yaml", type=str, help="Path to config file")
    parser.add_argument("--max_epochs", default=100, type=int, help="Maximum training epochs")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--accelerator", default="auto", type=str, help="Accelerator (auto, gpu, cpu)")
    parser.add_argument("--devices", default="auto", type=str, help="Number of devices")
    #parser.add_argument("--precision", default="64", type=str, help="Training precision")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--finetuning", action="store_true", help="Load pretrained weights for finetuning")
    parser.add_argument("--save", action="store_true", help="Save final model")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Override config with command line args
    config["batch_size"] = args.batch_size
    config["max_epochs"] = args.max_epochs
    config["cuda"] = torch.cuda.is_available()
    config["data_path"] = args.data_path
    
    # Set seed for reproducibility
    set_seed(config["random_seed"])
    
    # Initialize wandb if enabled
    if args.wandb:
        import wandb
        wandb.login()
        wandb.init(project="crystalvae", config=config)
        config = wandb.config
    
    # Create dataset and dataloaders
    print(f"Loading dataset from {args.data_path}")
    dataset = CIFData(
        config["data_path"], 
        keep=False, 
        max_N=config["k"],
        proper=config.get("property", 'formation_energy'),
        force=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Max atoms per crystal: {config['k']}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_pool,
        batch_size=config["batch_size"],
        train_ratio=config["train_ratio"],
        num_workers=config["workers"],
        val_ratio=config["val_ratio"],
        test_ratio=config["test_ratio"],
        pin_memory=True,
        return_test=True,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create loss function
    loss = FocalLoss(class_num=config["class_dim"])
    
    # Create model
    print("Creating model...")
    model_config = {
        "loss": loss,
        "class_dim": config["class_dim"],
        "pos_dim": config["pos_dim"],
        "lattice_dim": config["lattice_dim"],
        "hidden_dim": config["hidden_dim"],
        "latent_dim": config["latent_dim"],
        "k": config["k"],
        "encoder1_dim": config["encoder1_dim"],
        "num_encoder2_layers": config["num_encoder2_layers"],
        "decoder2_dims": config["decoder2_dims"],
        "decoder1_dims": config["decoder1_dims"],
        "decoder_lattice_dims": config["decoder_lattice_dims"],
        "decoder_lattice_dims2": config["decoder_lattice_dims2"],
        "codebooksize1": config["codebooksize1"],
        "codebooksize2": config["codebooksize2"],
        "codebooksizel": config["codebooksizel"],
        "num_quantizers1": config["num_quantizers1"],
        "num_quantizers2": config["num_quantizers2"],
        "num_quantizersl": config["num_quantizersl"],
        "sample_codebook_temp": config["sample_codebook_temp"],
        'lambdas': config.get("lambdas", None),
        "normalizer": dataset.norm,
        "normalizer_properties": dataset.norm_properties,
    }
    
    model = GraphVAE(**model_config)
    
    # Load pretrained weights if finetuning
    if args.finetuning:
        print("Loading pretrained weights for finetuning...")
        name = Path(args.data_path).name
        checkpoint_path = f"./ckpt/{name}/{name}_model_weights_{config['codebooksize1']}_{config['num_quantizers1']}_.pth"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print("Pretrained weights loaded successfully!")
    
    # Create Lightning module
    lightning_module = CrystalVAELightningModule(config, model, loss)
    
    # Create callbacks
    callbacks = []
    
    # Model checkpointing
    name = Path(args.data_path).name
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'./checkpoints/{name}',
        filename=f'{name}_best_loss_{{epoch:02d}}_{{val_loss:.4f}}',
        save_top_k=3,
        mode='min',
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Best match accuracy checkpoint
    #match_checkpoint_callback = ModelCheckpoint(
    #    monitor='val_match_acc_with_lattice',
    #    dirpath=f'./checkpoints/{name}',
    #    filename=f'{name}_best_match_{{epoch:02d}}_{{val_match_acc_with_lattice:.4f}}',
    #    save_top_k=1,
    #    mode='max',
    #)
    #callbacks.append(match_checkpoint_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=50,
        mode='min'
    )
    callbacks.append(early_stopping)
    
    # Create logger
    logger = None
    if args.wandb:
        logger = WandbLogger(project="crystalvae", log_model=True)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator=args.accelerator,
        devices=args.devices,
        #precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=config.get("validate_every_epoch", 50),
        gradient_clip_val=config.get("grad_clip", None),
        accumulate_grad_batches=1,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True,
        num_sanity_val_steps=1,
    )
    
    # Print model summary
    print("Model summary:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train the model
    print("Starting training...")
    trainer.fit(lightning_module, train_loader, val_loader)
    
    # Test the model
    print("Running test...")
    trainer.test(lightning_module, test_loader)
    
    # Save final model with normalizers
    if args.save:
        final_save = {
            "model_state_dict": lightning_module.model.state_dict(),
            "normalizer": lightning_module.normalizer,
            "normalizer_properties": lightning_module.normalizer_properties,
            "config": config,
            "best_match_acc": lightning_module.best_match_acc,
            "best_rms_dist": lightning_module.best_rms_dist,
        }
        save_path = f"{name}_final_model_weights_{config['codebooksize1']}_{config['num_quantizers1']}.pth"
        torch.save(final_save, save_path)
        print(f"Final model saved to {save_path}")
    
    print("Training completed!")
    print(f"Best Match Accuracy: {lightning_module.best_match_acc:.4f}")
    print(f"Best RMS Distance: {lightning_module.best_rms_dist:.4f}")
    print(f"Best RMS Match: {lightning_module.best_rms_match:.4f}")


if __name__ == "__main__":
    main()
