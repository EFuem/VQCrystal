import argparse
import os
from pickle import NONE
import shutil
import sys
import time
import warnings
from random import sample
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import r2_score
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
import pdb
import yaml
from tqdm import tqdm
from data import CIFData, NestedTensor, collate_pool, get_train_val_test_loader
from model.model2 import GraphVAE
from utils import *
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
import warnings
import warnings
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="basic argument for crystalvae")
# about data
parser.add_argument("--data_path", default="./data/p_20", type=str)
parser.add_argument("--config_path", default="./config/config_mp_20.yaml", type=str)
parser.add_argument("--wandb", default=False, type=bool)
parser.add_argument("--finetuning", default=False, type=bool)
parser.add_argument("--save", default=False, type=bool)
parser.add_argument("--max_epochs", default=2000, type=int)
parser.add_argument("--accelerator", default="cpu", type=str)
parser.add_argument("--devices", default="auto", type=str)
parser.add_argument("--precision", default="16-mixed", type=str)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class CrystalVAELightningModule(pl.LightningModule):
    def __init__(self, config, model, loss_fn):
        super().__init__()
        self.config = config
        self.model = model
        self.loss_fn = loss_fn
        self.save_hyperparameters(ignore=['model', 'loss_fn'])
        
        # Track best metrics
        self.best_match_acc = 0.0
        self.best_loss = float('inf')
        self.best_rms_dist = float('inf')
        
        # Store normalizers for saving
        self.normalizer = model.normalizer
        self.normalizer_properties = model.normalizer_properties

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch_dict, other = batch
        for k, v in batch_dict.items():
            if isinstance(v, torch.Tensor):
                batch_dict[k] = v.to(device)
            elif isinstance(v, list):
                batch_dict[k] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in v]
            elif isinstance(v, NestedTensor):
                batch_dict[k] = NestedTensor(
                    data=v.data.to(device),
                    mask=v.mask.to(device)
                )
            else:
                batch_dict[k] = v

        return (batch_dict, other)

        
    def forward(self, F, A, flat_mask, lattice, nbr_fea, nbr_fea_idx, crystal_atom_idx, properties):
        return self.model(F, A, flat_mask, lattice, nbr_fea, nbr_fea_idx, crystal_atom_idx, properties)
    
    def training_step(self, batch, batch_idx):
        batch_dict, _ = batch
        atom_fea = batch_dict['atom_fea']
        A = batch_dict['ad_matrix']
        coords = batch_dict['frac_coords_tensor']
        lattice = batch_dict['lattice']
        crystals = batch_dict['crystals']
        nbr_fea = batch_dict['nbr_fea']
        nbr_fea_idx = batch_dict['nbr_fea_idx']
        crystal_atom_idx = batch_dict['crystal_atom_idx']
        properties = batch_dict['property'].to(self.device)
        
        F = torch.cat((atom_fea.data, coords.data), dim=2)
        flat_mask = atom_fea.mask
        F = F.to(self.device)
        A = A.data.to(self.device)
        nbr_fea, nbr_fea_idx = nbr_fea.to(self.device), nbr_fea_idx.to(self.device)
        crystal_atom_idx = [crys_idx.to(self.device) for crys_idx in crystal_atom_idx]
        flat_mask = flat_mask.to(self.device)
        lattice = lattice.to(self.device)
        
        # Forward pass
        output = self(F, A, flat_mask, lattice, nbr_fea, nbr_fea_idx, crystal_atom_idx, properties)
        
        # Log training loss
        self.log('train_loss', output['total_loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_V_loss', output['V_loss'], on_step=True, on_epoch=True)
        self.log('train_P_loss', output['P_loss'], on_step=True, on_epoch=True)
        self.log('train_commitment_loss', output['commitment_loss'], on_step=True, on_epoch=True)
        self.log('train_lattice_loss', output['lattice_loss'], on_step=True, on_epoch=True)
        self.log('train_property_loss', output['property_loss'], on_step=True, on_epoch=True)
        
        # Evaluation forward pass for additional metrics
        with torch.no_grad():
            eval_output = self.model.evaluation_forward(
                F, A, flat_mask, lattice, nbr_fea, nbr_fea_idx, 
                crystal_atom_idx, properties, crystals, accuracy=False
            )
            self.log('train_eval_loss', eval_output['total_loss'], on_step=True, on_epoch=True)
        
        return output['total_loss']
    
    def validation_step(self, batch, batch_idx):
        batch_dict, _ = batch
        atom_fea = batch_dict['atom_fea']
        A = batch_dict['ad_matrix']
        coords = batch_dict['frac_coords_tensor']
        lattice = batch_dict['lattice']
        crystals = batch_dict['crystals']
        nbr_fea = batch_dict['nbr_fea']
        nbr_fea_idx = batch_dict['nbr_fea_idx']
        crystal_atom_idx = batch_dict['crystal_atom_idx']
        properties = batch_dict['property'].to(self.device)
        
        F = torch.cat((atom_fea.data, coords.data), dim=2)
        flat_mask = atom_fea.mask
        F = F.to(self.device)
        A = A.data.to(self.device)
        nbr_fea, nbr_fea_idx = nbr_fea.to(self.device), nbr_fea_idx.to(self.device)
        crystal_atom_idx = [crys_idx.to(self.device) for crys_idx in crystal_atom_idx]
        flat_mask = flat_mask.to(self.device)
        lattice = lattice.to(self.device)
        
        # Evaluation forward pass
        output = self.model.evaluation_forward(
            F, A, flat_mask, lattice, nbr_fea, nbr_fea_idx, 
            crystal_atom_idx, properties, crystals
        )
        
        # Log validation metrics
        self.log('val_loss', output['total_loss'], on_epoch=True, prog_bar=True)
        self.log('val_V_loss', output['V_loss'], on_epoch=True)
        self.log('val_P_loss', output['P_loss'], on_epoch=True)
        self.log('val_commitment_loss', output['commitment_loss'], on_epoch=True)
        self.log('val_lattice_loss', output['lattice_loss'], on_epoch=True)
        self.log('val_property_loss', output['property_loss'], on_epoch=True)
        #self.log('val_atom_acc', output['atom_acc'], on_epoch=True, prog_bar=True)
        #self.log('val_match_acc_with_lattice', output['match_accuracy_with_lattice'], on_epoch=True, prog_bar=True)
        #self.log('val_rms_dist_with_lattice', output['rms_dist_with_lattice'][0], on_epoch=True)
        
        # Update best metrics
        #if output['match_accuracy_with_lattice'] > self.best_match_acc:
        #    self.best_match_acc = output['match_accuracy_with_lattice']
        #    self.best_match_rms = output['rms_dist_with_lattice'][0]
        
        #if output['rms_dist_with_lattice'][0] < self.best_rms_dist:
        #    self.best_rms_dist = output['rms_dist_with_lattice'][0]
        #    self.best_rms_match = output['match_accuracy_with_lattice']
        
        return output
    
    def configure_optimizers(self):
        optimizer = getattr(optim, self.config["optimizer"]["type"])(
            self.parameters(), **self.config["optimizer"]["params"]
        )
        
        scheduler = getattr(torch.optim.lr_scheduler, self.config["scheduler"]["type"])(
            optimizer, **self.config["scheduler"]["params"]
        )
        
        # Add warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=self.config["warmup_factor"], 
            end_factor=1.0, total_iters=self.config["warmup_epochs"]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
            "warmup_scheduler": {
                "scheduler": warmup_scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }
    
    def on_train_epoch_start(self):
        # Apply warmup learning rate
        if self.current_epoch < self.config["warmup_epochs"]:
            warmup_factor = self.config["warmup_factor"]
            max_lr = self.config["optimizer"]["params"]["lr"]
            lr = warmup_factor * max_lr + (max_lr - warmup_factor * max_lr) * (
                self.current_epoch / self.config["warmup_epochs"]
            )
            for param_group in self.optimizers().param_groups:
                param_group["lr"] = lr
    
    def on_train_epoch_end(self):
        # Print epoch summary
        train_loss = self.trainer.callback_metrics.get('train_loss', torch.tensor(0.0))
        val_loss = self.trainer.callback_metrics.get('val_loss', torch.tensor(0.0))
        print(f"Epoch {self.current_epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    def on_validation_end(self):
        pass
        # Print best metrics
        #print(f"Best Match Acc: {self.best_match_acc:.4f}")
        #print(f"Best RMS Dist: {self.best_rms_dist:.4f}")
        #print(f"Best RMS Match: {self.best_rms_match:.4f}")


if __name__ == "__main__":
    args = parser.parse_args()
    name = Path(args.data_path).name
    
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config["cuda"] = torch.cuda.is_available()
    config["data_path"] = args.data_path
    config["max_epochs"] = args.max_epochs
    
    # Set seed
    set_seed(config["random_seed"])
    
    # Initialize wandb if enabled
    if args.wandb:
        import wandb
        wandb.login()
        wandb.init(project="crystalvae", config=config)
        config = wandb.config
    
    # Create dataset and dataloaders
    dataset = CIFData(
        config["data_path"], 
        keep=False, 
        max_N=config["k"],
        proper=config.get("property", 'formation_energy'),
        force=False
    )
    
    element_counter = dataset.element_counter
    normalizer = dataset.norm
    normalizer_properties = dataset.norm_properties
    
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_pool,
        batch_size=config["batch_size"],
        train_ratio=config["train_ratio"],
        num_workers=config["workers"],
        val_ratio=config["val_ratio"],
        test_ratio=config["test_ratio"],
        pin_memory=True,
        train_size=NONE,
        val_size=NONE,
        test_size=NONE,
        return_test=True,
    )
    
    # Create loss function
    loss = FocalLoss(class_num=config["class_dim"])
    
    # Create model
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
        "normalizer": normalizer,
        "normalizer_properties": normalizer_properties,
    }
    
    model = GraphVAE(**model_config)
    
    # Load pretrained weights if finetuning
    if args.finetuning:
        print("Loading model weights for finetuning...")
        checkpoint_path = f"./ckpt/{name}/{name}_model_weights_{config['codebooksize1']}_{config['num_quantizers1']}_.pth"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print("Load OK!")
    
    # Create Lightning module
    lightning_module = CrystalVAELightningModule(config, model, loss)
    
    # Create callbacks
    callbacks = []
    
    # Model checkpointing
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
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=config.get("validate_every_epoch", 50),
        gradient_clip_val=config.get("grad_clip", None),
        accumulate_grad_batches=1,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True,
    )
    
    # Train the model
    trainer.fit(lightning_module, train_loader, val_loader)
    
    # Test the model
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
        torch.save(final_save, f"{name}_final_model_weights_{config['codebooksize1']}_{config['num_quantizers1']}.pth")
    
    print("Training completed!")
    print(f"Best Match Accuracy: {lightning_module.best_match_acc:.4f}")
    print(f"Best RMS Distance: {lightning_module.best_rms_dist:.4f}")
    print(f"Best RMS Match: {lightning_module.best_rms_match:.4f}")
    

