import lightning as L
import torch
from torch.optim import AdamW, SGD
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vit_custom import create_vit_tiny
from datasets.datasets import create_dataloader

class ViTLightningModule(L.LightningModule):
    def __init__(
        self,
        rope_type: str = 'standard',
        num_classes: int = 200,
        lr: float = 1e-4,
        weight_decay: float = 0.1,
        max_epochs: int = 100
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_vit_tiny(num_classes=num_classes, rope_type=rope_type)
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)
        classification_loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        return classification_loss, acc
    
    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=True, on_epoch=True)
        
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }

def train(data_path: str, rope_type: str = 'standard'):
    # data
    train_loader = create_dataloader(
        data_path=data_path,
        train=True,
        batch_size=128
    )
    val_loader = create_dataloader(
        data_path=data_path,
        train=False,
        batch_size=128
    )
    
    # model
    model = ViTLightningModule(rope_type=rope_type)
    
    wandb_logger = WandbLogger(
        project="vit-training",
        name=f"{rope_type}-vit-tiny",
        log_model=True
    )

    early_stop_callback = EarlyStopping(
        monitor='val_acc',      # the metric to monitor
        patience=15,             # if 15 epochs no improvement, stop
        mode='max',             # higher is better
        verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        save_top_k=1,               # only save the best one
        filename='best-vit-{epoch:02d}-{val_acc:.4f}',  # file name format
        save_weights_only=False     
    )

    # trainer
    trainer = L.Trainer(
        max_epochs=300,
        accelerator='auto',
        devices=1,
        precision='16-mixed',
        callbacks=[
            early_stop_callback,
            checkpoint_callback,
            LearningRateMonitor(logging_interval='step')
        ],
        logger=wandb_logger  
    )
    
    # start training
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    print(f"Best model path: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-d", type=str, default='/root/autodl-tmp/tiny-imagenet-200')
    parser.add_argument("--rope_type", 
                        "-r", 
                        type=str, 
                        default='standard',
                        choices=['standard','cayley','givens','householder'],
                        help='RoPE type')
    
    args = parser.parse_args()

    L.seed_everything(42, workers=True)
    train(data_path=args.data_path, rope_type=args.rope_type)

    