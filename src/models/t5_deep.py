# pylint: disable-msg=no-member
# pylint: disable=too-many-ancestors
# pylint: disable=arguments-differ
# pylint: disable=unused-argument
"""
    AV Project:
        models:
            mt5 encoder finetune
"""
import pytorch_lightning as pl
# ============================ Third Party libs ============================
import torch
import torch.nn.functional as function
import torchmetrics
from torch import nn
# ============================ My packages ============================
from transformers import T5EncoderModel

from .attention import ScaledDotProductAttention


class Classifier(pl.LightningModule):
    """
        Classifier
    """

    def __init__(self, num_classes, t5_model_path, lr, max_len, **kwargs):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy()
        self.f_score = torchmetrics.F1(average='none', num_classes=num_classes)
        self.f_score_total = torchmetrics.F1(average="weighted", num_classes=num_classes)
        self.max_len = max_len
        self.learning_rare = lr

        self.model = T5EncoderModel.from_pretrained(t5_model_path)

        self.classifier = nn.Linear(3 * self.model.config.d_model,
                                    num_classes)
        self.attention = ScaledDotProductAttention(3 * self.model.config.d_model)
        self.max_pool = nn.MaxPool1d(max_len)
        self.max_pool_info = nn.MaxPool1d(max_len // 4)

        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, batch):
        punctuation = self.model(batch["punctuation"]).last_hidden_state  # .permute(0, 2, 1)
        output_encoder = self.model(batch["input_ids"]).last_hidden_state  # .permute(0, 2, 1)
        information = self.model(batch["information"]).last_hidden_state  # .permute(0, 2, 1)
        output = torch.cat([punctuation, output_encoder, information], dim=2)
        # output.size() = [batch_size, sent_len, embedding_dim+2*hid_dim]===>(64,150,1024)

        context, attn = self.attention(output, output, output)
        output = context.permute(0, 2, 1)
        features = function.max_pool1d(output, output.shape[2]).squeeze(2)
        final_output = self.classifier(features)
        return final_output

    def training_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:
        :return:
        """
        label = batch['targets'].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        metric2value = {'train_loss': loss,
                        'train_acc':
                            self.accuracy(torch.softmax(outputs, dim=1), label),
                        'train_f1_first_class':
                            self.f_score(torch.softmax(outputs, dim=1), label)[0],
                        'train_f1_second_class':
                            self.f_score(torch.softmax(outputs, dim=1), label)[1],
                        'train_total_F1':
                            self.f_score_total(torch.softmax(outputs, dim=1), label)}

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'predictions': outputs, 'labels': label}

    def validation_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:
        :return:
        """
        label = batch['targets'].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        metric2value = {'val_loss': loss,
                        'val_acc':
                            self.accuracy(torch.softmax(outputs, dim=1), label),
                        'val_f1_first_class':
                            self.f_score(torch.softmax(outputs, dim=1), label)[0],
                        'val_f1_second_class':
                            self.f_score(torch.softmax(outputs, dim=1), label)[1],
                        'val_total_F1':
                            self.f_score_total(torch.softmax(outputs, dim=1), label)}

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:
        :return:
        """
        label = batch['targets'].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        metric2value = {'test_loss': loss,
                        'test_acc':
                            self.accuracy(torch.softmax(outputs, dim=1), label),
                        'test_f1_first_class':
                            self.f_score(torch.softmax(outputs, dim=1), label)[0],
                        'test_f1_second_class':
                            self.f_score(torch.softmax(outputs, dim=1), label)[1],
                        'test_total_F1':
                            self.f_score_total(torch.softmax(outputs, dim=1), label)}

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """

        :return:
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rare)
        return [optimizer]
