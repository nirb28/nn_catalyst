import torch
import pytorch_lightning as pl
from model import NN
from dataset import CatalystDataModule
import config
from pytorch_lightning.callbacks import EarlyStopping

torch.set_float32_matmul_precision("medium") # to make lightning happy

if __name__ == "__main__":
    model = NN(
        input_size=config.INPUT_SIZE,
        learning_rate=config.LEARNING_RATE,
        target=config.NUM_CLASSES,
    )
    dm = CatalystDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    trainer = pl.Trainer(
        accelerator=config.ACCELERATOR,
        #devices=config.DEVICES,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
        callbacks=[EarlyStopping(monitor="val_loss")],
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
