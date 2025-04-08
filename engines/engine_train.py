import lightning as L
import torch
from torch.optim import AdamW
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from torch.optim.lr_scheduler import CosineAnnealingLR
from models.vit_custom import create_axial_vit_tiny
from datasets.datasets import create_dataloader

class ViTLightningModule(L.LightningModule):
    def __init__(
        self,
        num_classes: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        max_epochs: int = 100
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_axial_vit_tiny(num_classes=num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        return loss, acc
    
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

def train():
    # data
    train_loader = create_dataloader(
        root='./data',
        train=True,
        batch_size=128
    )
    val_loader = create_dataloader(
        root='./data',
        train=False,
        batch_size=128
    )
    
    # model
    model = ViTLightningModule()
    
    wandb_logger = WandbLogger(
    project="vit-training",
    name="axial-vit-tiny",
    log_model=True
    )

    early_stop_callback = EarlyStopping(
        monitor='val_acc',      # the metric to monitor
        patience=5,             # if 5 epochs no improvement, stop
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
            L.callbacks.LearningRateMonitor()
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
    train()
