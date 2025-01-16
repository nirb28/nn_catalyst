from torch import nn, optim
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl

class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.r2 = torchmetrics.R2Score()
        self.loss_fn = nn.MSELoss()
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "train_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        accuracy = self.r2(scores, y)
        self.log("train_acc", accuracy, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("val_loss", loss)
        self.validation_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.log("validation_epoch_average", epoch_average)
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        if DEBUG == True:
            print(f"loss: {loss}, len: {len(y)}")
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(lr=self.lr, params=self.parameters())
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=0.000000001, threshold=0.001)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

class SingleTargetNet(BaseModel):

    def __init__(self, input_size, learning_rate: float=0.001, dropout_rate: float = 0.2, target=1):
        super(SingleTargetNet, self).__init__()
        self.lr = learning_rate
        self.loss_fn = nn.MSELoss()

        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1)
        self.fc_skip = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.save_hyperparameters()

    def forward(self, x):
        x1 = F.relu(self.bn1(self.fc1(x)))
        x1 = self.dropout(x1)

        x2 = F.relu(self.bn2(self.fc2(x1)))
        x2 = self.dropout(x2)

        # Skip connection
        x2 += self.fc_skip(x1)

        x3 = self.fc3(x2)
        return x3