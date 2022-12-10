# -*- coding: utf-8 -*-
# ========================================================

"""
    Author Verification Project:
        dataset:
            dataset.py
"""

# ============================ Third Party libs ============================
from abc import ABC, abstractmethod
import torch
import transformers


# ==========================================================================


class CustomDataset(ABC, torch.utils.data.Dataset):
    """
    Abstract class to  write dataset class for training language comprehension model

    Attributes:
        data: all features for each data
        tokenizer: huggingface tokenizer
        max_len: maximum length for each sample
    """

    def __init__(self, data: dict,
                 tokenizer: transformers.AutoTokenizer.from_pretrained,
                 max_len: int):
        self.first_text = data["first_text"]
        self.second_text = data["second_text"]

        self.first_punctuations = data["first_punctuations"]
        self.second_punctuations = data["second_punctuations"]

        self.first_information = data["first_information"]
        self.second_information = data["second_information"]

        self.first_pos = data["first_pos"]
        self.second_pos = data["second_pos"]

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
        function te return one sample by given index

        Args:
            item_index: random index of sample

        Returns:
            selected sample by given index

        """

        first_text = self.first_text[item_index]
        second_text = self.second_text[item_index]

        first_punctuations = self.first_punctuations[item_index]
        second_punctuations = self.second_punctuations[item_index]

        first_information = self.first_information[item_index]
        second_information = self.second_information[item_index]

        first_pos = self.first_pos[item_index]
        second_pos = self.second_pos[item_index]

        if self.targets:
            target = self.targets[item_index]
        else:
            target = None

        return first_text, second_text, first_punctuations, second_punctuations, \
               first_information, second_information, first_pos, second_pos, target

    def pair_data_tokenizer(self,
                            first_text: str,
                            second_text: str,
                            max_len: int):
        """
        pair data tokenizer
        Args:
            first_text: first text
            second_text: second text
            max_len: maximum length for each sample

        Returns:
            tokenized sample

        """
        batch = self.tokenizer.encode_plus(text=first_text,
                                           text_pair=second_text,
                                           add_special_tokens=True,
                                           max_length=max_len,
                                           return_tensors="pt",
                                           padding="max_length",
                                           truncation="longest_first",
                                           return_token_type_ids=True)
        return batch

    def single_data_tokenizer(self,
                              text: str):
        """
        single data tokenizer
        Args:
            text: text

        Returns:
            tokenized sample

        """
        batch = self.tokenizer.encode_plus(text=text,
                                           add_special_tokens=True,
                                           max_length=self.max_len,
                                           return_tensors="pt",
                                           padding="max_length",
                                           truncation=True,
                                           return_token_type_ids=True)
        return batch


class ConcatDataset(CustomDataset):
    """
    ConcatDataset class to create data for author verification model

    Attributes:
        data: all features for each data
        tokenizer: huggingface tokenizer
        max_len: maximum length for each sample

    """
    def __getitem__(self, item_index):
        first_text, second_text, first_punctuations, second_punctuations, \
        first_information, second_information, first_pos, \
        second_pos, target = super().__getitem__(item_index)

        text = self.pair_data_tokenizer(first_text, second_text, max_len=self.max_len)

        punctuations = self.pair_data_tokenizer(first_punctuations, second_punctuations,
                                                max_len=self.max_len)
        information = self.pair_data_tokenizer(first_information, second_information,
                                               max_len=self.max_len)
        pos = self.pair_data_tokenizer(first_pos, second_pos, max_len=self.max_len)

        input_ids = text.input_ids.flatten()
        punctuations = punctuations.input_ids.flatten()
        information = information.input_ids.flatten()
        pos = pos.input_ids.flatten()

        if target:
            return {"input_ids": input_ids, "punctuation": punctuations, "information": information,
                    "pos": pos, "targets": torch.tensor(target)}
        return {"input_ids": input_ids, "punctuation": punctuations, "information": information,
                "pos": pos}
