# pylint: disable-msg=no-member
# pylint: disable=too-many-ancestors
# pylint: disable=arguments-differ
# pylint: disable=unused-argument
"""
    AV Project:
        models:
            mt5 encoder finetune
"""
# ============================ Third Party libs ============================
import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics
import torch.nn.functional as F

# ============================ My packages ============================
from transformers import T5EncoderModel


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
        self.punc_embeddings = nn.Embedding(kwargs["vocab_size"],
                                            embedding_dim=kwargs["embedding_dim"],
                                            padding_idx=kwargs["pad_idx"])
        self.punc_embeddings.weight.requires_grad = True
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=kwargs["n_filters"],
                      kernel_size=(fs, kwargs["embedding_dim"]))
            for fs in kwargs["filter_sizes"]
        ])

        self.classifier = nn.Linear(self.model.config.d_model + (len(kwargs["filter_sizes"])*kwargs["n_filters"]),
                                    num_classes)

        self.max_pool = nn.MaxPool1d(max_len)

        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, batch):
        input_ids = batch["input_ids"]
        punctuation = batch["punctuation"]  # .to("cuda:0")

        punctuation = self.punc_embeddings(punctuation)
        # punctuation = [batch_size, sent_len, emb_dim]

        punctuation = punctuation.unsqueeze(1)
        # embedded_cnn = [batch_size, 1, sent_len, emb_dim]

        conved = [torch.nn.ReLU()(conv(punctuation)).squeeze(3) for conv in self.convs]
        # conved_n = [batch_size, n_filters, sent_len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch_size, n_filters]

        cat_cnn = torch.cat(pooled, dim=1)
        # cat_cnn = [batch_size, n_filters * len(filter_sizes)]

        output_encoder = self.model(input_ids).last_hidden_state.permute(0, 2, 1)

        maxed_pool = self.max_pool(output_encoder).squeeze(2)
        features = torch.cat((cat_cnn, maxed_pool), dim=1)

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
