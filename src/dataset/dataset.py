# pylint: disable-msg=too-few-public-methods
# pylint: disable-msg=no-member
# pylint: disable-msg=arguments-differ

"""
    AV Project:
        models:
            dataset
"""

# ============================ Third Party libs ============================
from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch


# ==========================================================================


class CustomDataset(ABC, torch.utils.data.Dataset):
    """
        CustomDataset is a abstract class
    """

    def __init__(self, data: dict, tokenizer, max_len: int):
        self.first_text = data["first_text"]
        self.second_text = data["second_text"]
        self.targets = None
        if "targets" in data:
            self.targets = data["targets"]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.first_text)

    @abstractmethod
    def __getitem__(self, item_index):
        """

        :param item_index:
        :return:
        """
        first_text = self.first_text[item_index]
        second_text = self.second_text[item_index]
        if self.targets:
            target = self.targets[item_index]
            return first_text, second_text, target
        return first_text, second_text

    def pair_data_tokenizer(self, first_text, second_text):
        batch = self.tokenizer.encode_plus(text=first_text,
                                           text_pair=second_text,
                                           add_special_tokens=True,
                                           max_length=self.max_len,
                                           return_tensors="pt",
                                           padding="max_length",
                                           truncation=True,
                                           return_token_type_ids=True)
        return batch

    def single_data_tokenizer(self, text):
        batch = self.tokenizer.encode_plus(text=text,
                                           add_special_tokens=True,
                                           max_length=self.max_len,
                                           return_tensors="pt",
                                           padding="max_length",
                                           truncation=True,
                                           return_token_type_ids=True)
        return batch


class SeparateDataset(CustomDataset):
    """
        SeparateDataset
    """

    def __init__(self, data: dict, tokenizer, max_len: int):
        super().__init__(data, tokenizer, max_len)

    def __getitem__(self, item_index):
        first_text, second_text, target = super(SeparateDataset, self).__getitem__(item_index)
        first_text = self.single_data_tokenizer(first_text)
        second_text = self.single_data_tokenizer(second_text)

        first_text = first_text.input_ids.flatten()
        second_text = second_text.input_ids.flatten()

        return {"first_text": first_text,
                "second_text": second_text,
                "targets": torch.tensor(target)}


class ConcatDataset(CustomDataset):
    """
        ConcatDataset
    """

    def __init__(self, data, tokenizer, max_len: int):
        super().__init__(data, tokenizer, max_len)

    def __getitem__(self, item_index):
        first_text, second_text, target = super(ConcatDataset, self).__getitem__(item_index)
        batch = self.pair_data_tokenizer(first_text, second_text)

        input_ids = batch.input_ids.flatten()

        return {"input_ids": input_ids, "targets": torch.tensor(target)}


class GenerationDataset(CustomDataset):
    """
        GenerativeDataset
    """

    def __init__(self, data: dict, tokenizer, max_len: int):
        super().__init__(data, tokenizer, max_len)

    def __getitem__(self, item_index):
        """

        :param item_index:
        :return:
        """
        first_text, second_text, target = super(GenerationDataset, self).__getitem__(item_index)

        input_batch = self.pair_data_tokenizer(first_text, second_text)

        with self.tokenizer.as_target_tokenizer():
            target_batch = self.single_data_tokenizer(str(target))

        inputs_ids = input_batch.input_ids.flatten()
        target_ids = target_batch.input_ids.flatten()

        return {"input_ids": inputs_ids, "targets": torch.tensor(target), "target_ids": target_ids}


class InferenceDataset(CustomDataset):
    """
    dataset to inference  data from model checkpoint
    """

    def __init__(self, data: dict, tokenizer, max_len):
        super(InferenceDataset, self).__init__(data, tokenizer, max_len)

    def __getitem__(self, item_index):
        first_text, second_text = super(InferenceDataset, self).__getitem__(item_index)

        batch = self.pair_data_tokenizer(first_text, second_text)

        input_ids = batch.input_ids.flatten()

        return {"input_ids": input_ids}


class DataModule(pl.LightningDataModule):
    """
        DataModule
    """

    def __init__(self, data: dict,
                 config, tokenizer):
        super().__init__()
        self.config = config
        self.data = data
        self.tokenizer = tokenizer
        self.customs_dataset = {}

    def setup(self):
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
        return torch.utils.data.DataLoader(self.customs_dataset["train_dataset"],
                                           batch_size=self.config.batch_size,
                                           shuffle=True, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.customs_dataset["val_dataset"],
                                           batch_size=self.config.batch_size,
                                           num_workers=self.config.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.customs_dataset["test_dataset"],
                                           batch_size=self.config.batch_size,
                                           num_workers=self.config.num_workers)
