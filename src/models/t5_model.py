# -*- coding: utf-8 -*-
# ========================================================

"""
    Author Verification Project:
        models:
            t5_model.py
"""
# ============================ Third Party libs ============================
from typing import List
import pytorch_lightning as pl
import torchmetrics
import torch
import transformers
# ============================ My packages ============================

from .attention import ScaledDotProductAttention


class Convolution(torch.nn.Module):
    """
    Convolution module to extract features from author-specific and topic-specific information

    Attributes:
        n_filters: number of filters
        filter_sizes: list of filter sizes
        max_len: maximum length for each sample
        lm_model: language model
    """

    def __init__(self,
                 n_filters: int,
                 filter_sizes: List[int],
                 max_len: int,
                 lm_model: transformers.T5EncoderModel.from_pretrained):
        super().__init__()
        self.lm_model = lm_model
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels=1,
                            out_channels=n_filters,
                            kernel_size=(fs, self.lm_model.config.d_model))
            for fs in filter_sizes
        ])
        self.max_len = max_len

        self.max_pool = torch.nn.MaxPool1d(self.max_len)
        self.max_pool_info = torch.nn.MaxPool1d(self.max_len // 4)

    def forward(self, batch):
        outputs = self.lm_model(batch).last_hidden_state.unsqueeze(1)

        outputs = [torch.nn.ReLU()(conv(outputs)).squeeze(3) for conv in self.convs]
        # conved_n = [batch_size, n_filters, sent_len - filter_sizes[n] + 1]

        outputs = [torch.nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in outputs]
        # pooled_n = [batch_size, n_filters]

        outputs = torch.cat(outputs, dim=1)
        return outputs


class Classifier(pl.LightningModule):
    """
        Classifier
    """

    def __init__(self,
                 t5_model_path: str,
                 num_classes: int,
                 lr: float,
                 max_len: int,
                 n_filters: int,
                 filter_sizes: List[int]):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy()
        self.f_score_total = torchmetrics.F1(average="weighted", num_classes=num_classes)
        self.learning_rare = lr
        self.max_len = max_len

        self.model = transformers.T5EncoderModel.from_pretrained(t5_model_path)

        self.convolution = Convolution(n_filters=n_filters, filter_sizes=filter_sizes,
                                       max_len=self.max_len, lm_model=self.model)

        self.classifier = torch.nn.Linear(3 * self.model.config.d_model +
                                          len(filter_sizes * n_filters),
                                          num_classes)
        self.attention = ScaledDotProductAttention(3 * self.model.config.d_model)

        self.loss = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, batch):
        # ------------- Text, Information and POS feature extraction -----------
        text_features = self.model(batch["input_ids"]).last_hidden_state
        information_features = self.model(batch["information"]).last_hidden_state
        pos_features = self.model(batch["pos"]).last_hidden_state

        features = torch.cat((text_features, information_features, pos_features), dim=2)

        # ------------------------ Attention Block -------------------------------
        context, attn = self.attention(features, features, features)
        output = context.permute(0, 2, 1)
        features = torch.nn.functional.max_pool1d(output, output.shape[2]).squeeze(2)

        # --------------------------- Convolution block --------------------------
        punctuation_features = self.convolution(batch["punctuation"])

        # ------------------------- Concat features ------------------------------
        features = torch.cat((features, punctuation_features), dim=1)

        # ------------------------- Prediction block ----------------------------
        final_output = self.classifier(features)
        return final_output

    def training_step(self, batch, batch_idx):
        """
        training phase
        Args:
            batch: input batch to processed with
            batch_idx: batch idx

        Returns:
            dictionary contains "loss", "predictions", and "labels"
        """

        label = batch['targets'].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        metric2value = {"train_loss": loss,
                        "train_acc":
                            self.accuracy(torch.softmax(outputs, dim=1), label),
                        "train_total_F1":
                            self.f_score_total(torch.softmax(outputs, dim=1), label)}

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": label}

    def validation_step(self, batch, batch_idx):
        """
        validation phase
        Args:
            batch: input batch to processed with
            batch_idx: batch idx

        Returns:
            loss
        """

        label = batch["targets"].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        metric2value = {"val_loss": loss,
                        "val_acc":
                            self.accuracy(torch.softmax(outputs, dim=1), label),
                        "val_total_F1":
                            self.f_score_total(torch.softmax(outputs, dim=1), label)}

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        test phase
        Args:
            batch: input batch to processed with
            batch_idx: batch idx

        Returns:
            loss
        """
        label = batch["targets"].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        metric2value = {"test_loss": loss,
                        "test_acc":
                            self.accuracy(torch.softmax(outputs, dim=1), label),
                        "test_total_F1":
                            self.f_score_total(torch.softmax(outputs, dim=1), label)}

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """

        :return:
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rare)
        return [optimizer]
