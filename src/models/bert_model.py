import torch
import torchmetrics
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, AdamW, get_linear_schedule_with_warmup


class Classifier(pl.LightningModule):
    def __init__(self, arg, n_classes, steps_per_epoch):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy()
        self.f1_score = torchmetrics.F1(average="weighted", num_classes=n_classes)
        self.category_f1_score = torchmetrics.F1(average="none", num_classes=n_classes)
        self.lr = arg.lr
        self.n_epochs = arg.n_epochs
        self.steps_per_epoch = steps_per_epoch  # BCEWithLogitsLoss
        self.criterion = torch.nn.CrossEntropyLoss()  # CrossEntropyLoss()
        self.bert = RobertaModel.from_pretrained(arg.roberta_model_path, return_dict=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.25),
            torch.nn.Linear(in_features=self.bert.config.hidden_size,
                            out_features=self.bert.config.hidden_size // 2),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(in_features=self.bert.config.hidden_size // 2,
                            out_features=self.bert.config.hidden_size // 4),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(in_features=self.bert.config.hidden_size // 4,
                            out_features=n_classes)
        )
        # self.classifier = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
        self.pooling = torch.nn.AvgPool1d(kernel_size=arg.max_length)

        # self.dropout = torch.nn.Dropout(0.2)
        self.save_hyperparameters()

    def forward(self, input_ids, attn_mask, token_type_ids):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attn_mask,
                                token_type_ids=token_type_ids)
        # bert_output.last_hidden_state.size() = [batch_size, sen_len, 768]

        bert_output = bert_output.pooler_output

        output = self.classifier(bert_output)  # output.pooler_output

        return output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["label"].flatten()
        outputs = self.forward(input_ids, attention_mask, token_type_ids)
        loss = self.criterion(outputs, labels)
        self.log("train_acc", self.accuracy(torch.softmax(outputs, dim=1), labels),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1", self.f1_score(torch.softmax(outputs, dim=1), labels),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1_first_class", self.category_f1_score(
            torch.softmax(outputs, dim=1), labels)[0],
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1_second_class", self.category_f1_score(
            torch.softmax(outputs, dim=1), labels)[1],
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["label"].flatten()
        outputs = self.forward(input_ids, attention_mask, token_type_ids)
        loss = self.criterion(outputs, labels)
        self.log("val_acc", self.accuracy(torch.softmax(outputs, dim=1), labels),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1", self.f1_score(torch.softmax(outputs, dim=1), labels),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1_first_class", self.category_f1_score(
            torch.softmax(outputs, dim=1), labels)[0],
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1_second_class", self.category_f1_score(
            torch.softmax(outputs, dim=1), labels)[1],
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["label"].flatten()
        outputs = self.forward(input_ids, attention_mask, token_type_ids)
        loss = self.criterion(outputs, labels)
        self.log("test_acc", self.accuracy(torch.softmax(outputs, dim=1), labels),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_f1", self.f1_score(torch.softmax(outputs, dim=1), labels),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_f1_first_class", self.category_f1_score(
            torch.softmax(outputs, dim=1), labels)[0],
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_f1_second_class", self.category_f1_score(
            torch.softmax(outputs, dim=1), labels)[1],
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        warmup_steps = self.steps_per_epoch // 3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps
        # scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer]  # , [scheduler]


if __name__ == '__main__':
    from src.configuration.config import BaseConfig

    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    MODEL = Classifier(CONFIG, n_classes=2,
                       steps_per_epoch=10)
    x = torch.rand((64, 150))
    y = torch.rand((64, 150))
    z = torch.rand((64, 150))

    MODEL.forward(x.long(), y.long(), z.long())
