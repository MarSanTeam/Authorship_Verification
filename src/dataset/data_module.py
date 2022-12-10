# -*- coding: utf-8 -*-
# ========================================================

"""
    Author Verification Project:
        dataset:
            data_module.py
"""

# ============================ Third Party libs ============================
import argparse
import pytorch_lightning as pl
import torch
import transformers

# ============================ My packages ============================
from .dataset import ConcatDataset


class DataModule(pl.LightningDataModule):
    """
    DataModule class to create data loader to train model

    Attributes:
        data: all features for each data
        tokenizer: huggingface tokenizer
        config: config arguments
    """

    def __init__(self,
                 data: dict,
                 tokenizer: transformers.AutoTokenizer.from_pretrained,
                 config: argparse.ArgumentParser.parse_args,
                 ):
        super().__init__()
        self.config = config
        self.data = data
        self.tokenizer = tokenizer
        self.customs_dataset = {}

    def setup(self, stage=None):
        """
        method to setup data module

        Returns:
            None
        """
        self.customs_dataset["train_dataset"] = ConcatDataset(
            data=self.data["train_data"], tokenizer=self.tokenizer, max_len=self.config.max_len
        )

        self.customs_dataset["val_dataset"] = ConcatDataset(
            data=self.data["val_data"], tokenizer=self.tokenizer, max_len=self.config.max_len
        )

        self.customs_dataset["test_dataset"] = ConcatDataset(
            data=self.data["test_data"], tokenizer=self.tokenizer, max_len=self.config.max_len
        )

    def train_dataloader(self):
        """
        method to create train dataloader

        Returns:
            Train data loader


        """
        return torch.utils.data.DataLoader(self.customs_dataset["train_dataset"],
                                           batch_size=self.config.batch_size,
                                           shuffle=True, num_workers=self.config.num_workers)

    def val_dataloader(self):
        """
        method to create validation data loader

        Returns:
            Validation data loader

        """
        return torch.utils.data.DataLoader(self.customs_dataset["val_dataset"],
                                           batch_size=self.config.batch_size,
                                           num_workers=self.config.num_workers)

    def test_dataloader(self):
        """
        method to create test data loader

        Returns:
            Test data loader

        """
        return torch.utils.data.DataLoader(self.customs_dataset["test_dataset"],
                                           batch_size=self.config.batch_size,
                                           num_workers=self.config.num_workers)
