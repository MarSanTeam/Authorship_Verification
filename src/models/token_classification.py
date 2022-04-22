"""
    ChitChat Splitter:
        BertForTokenClassification models
"""
# pylint: disable-msg=too-many-ancestors
# pylint: disable-msg=too-many-arguments
# pylint: disable-msg=arguments-differ
# pylint: disable-msg=no-member
# pylint: disable-msg=unused-argument
from typing import List
import pytorch_lightning as pl
import torch
import torchmetrics
from transformers import BertForTokenClassification, \
    AdamW, get_linear_schedule_with_warmup


class TokenClassificationModel(pl.LightningModule):
    """
    creates a pytorch lightning models
    """

    def __init__(self, config, label_columns: List[str],
                 n_warmup_steps: int = None,
                 n_training_steps: int = None,
                 n_classes: int = None):
        super().__init__()
        self.config = config
        self.bert = BertForTokenClassification.from_pretrained(
            config.ParsBERT_model_path,
            num_labels=n_classes, return_dict=True)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.label_columns = label_columns
        self.accuracy = torchmetrics.Accuracy()
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels=None):
        """

        :param input_ids:
        :param attention_mask:
        :param labels:
        :return:
        """
        loss = self.bert(input_ids=input_ids,
                         attention_mask=attention_mask,
                         labels=labels).loss
        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask,
                           labels=labels).logits
        return loss, output

    def training_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:
        :return:
        """
        input_ids = batch['input_ids'].to(dtype=torch.long)
        input_ids = input_ids.squeeze(1)
        attention_mask = batch['attention_mask'].to(dtype=torch.long)
        labels = batch['labels'].to(dtype=torch.long)
        loss, output = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": output, "labels": labels}

    def validation_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:
        :return:
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, _ = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:
        :return:
        """
        ids = batch['input_ids'].to(dtype=torch.long)
        mask = batch['attention_mask'].to(dtype=torch.long)
        labels = batch['labels'].to(dtype=torch.long)

        loss, eval_logits = self(input_ids=ids, attention_mask=mask, labels=labels)

        # compute evaluation accuracy

        # shape (batch_size * seq_len,)
        flattened_targets = labels.view(-1)
        # shape (batch_size * seq_len, num_labels)
        active_logits = eval_logits.view(-1, self.bert.num_labels)
        # shape (batch_size * seq_len,)
        flattened_predictions = torch.argmax(active_logits, dim=1)

        # only compute accuracy at active labels

        # shape (batch_size, seq_len)
        active_accuracy = labels.view(-1) != -100

        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        self.log("test accuracy", self.accuracy(predictions,
                                                torch.as_tensor(labels, dtype=torch.int)),
                 prog_bar=True, logger=True)
        self.log("test_loss", loss, prog_bar=True,
                 logger=True)
        return loss

    def configure_optimizers(self):
        """

        :return:
        """
        optimizer = AdamW(self.parameters(), lr=self.config.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.n_warmup_steps,
                                                    num_training_steps=self.n_training_steps)
        return dict(optimizer=optimizer,
                    lr_scheduler=dict(scheduler=scheduler, interval="step"))
