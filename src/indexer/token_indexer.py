# -*- coding: utf-8 -*-
"""
    AV Project:
        data preparation:
            index
"""

import os
# ============================ Third Party libs ============================
from typing import List

from data_loader import write_json

# =============================== My packages ==============================
from .indexer import Indexer


# ==========================================================================


class TokenIndexer(Indexer):
    """
        TokenIndexer
    """

    def __init__(self, vocabs: list = None, pad_index: int = 0, unk_index: int = 1):
        super().__init__(vocabs)
        self.pad_index = pad_index
        self.unk_index = unk_index

        self._vocab2idx = None
        self._idx2vocab = None

    def get_idx(self, token: str) -> int:
        """
        get_idx method is written for get index of input word
        :param token:
        :return:
        """
        if not self._vocab2idx:
            self._empty_vocab_handler()
        if token in self._vocab2idx:
            return self._vocab2idx[token]
        return self._vocab2idx["<UNK>"]

    def get_word(self, idx: int) -> str:
        """
        get_word method is written for get word of input index
        :param idx:
        :return:
        """
        if not self._idx2vocab:
            self._empty_vocab_handler()
        if idx in self._idx2vocab:
            return self._idx2vocab[idx]
        return self._idx2vocab[self.unk_index]

    def build_vocab2idx(self):
        """
        build_vocab2idx method is written to build vocab2ix dictionary
        :return:
        """
        self._vocab2idx = {"<PAD>": self.pad_index, "<UNK>": self.unk_index}
        for vocab in self.vocabs:
            self._vocab2idx[vocab] = len(self._vocab2idx)

    def build_idx2vocab(self):
        """
        build_idx2vocab method is written to build idx2vocab dictionary
        :return:
        """
        self._idx2vocab = {self.pad_index: "<PAD>", self.unk_index: "<UNK>"}
        for vocab in self.vocabs:
            self._idx2vocab[len(self._idx2vocab)] = vocab

    def convert_samples_to_char_indexes(self, tokenized_samples: List[list]) -> List[list]:
        """
        :param tokenized_samples: [[word_1, ..., word_n], ..., [word_1, ..., word_m]]
        :return: [[[ch1, ch2, ..], ..., [ch1, ch2, ..]], ..., [[ch1, ch2, ..], ..., [ch1, ch2, ..]]]
        """
        for index, tokenized_sample in enumerate(tokenized_samples):
            for token_index, token in enumerate(tokenized_sample):
                chars = []
                for char in str(token):
                    chars.append(self.get_idx(char))
                tokenized_samples[index][token_index] = chars
        return tokenized_samples
