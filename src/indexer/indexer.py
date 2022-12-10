# -*- coding: utf-8 -*-
"""
    AV Project:
        data preparation:
            indexer
"""
# ============================ Third Party libs ============================
import os
from abc import abstractmethod
from typing import List

# =============================== My packages ==============================
from data_loader import write_json, read_json


# ==========================================================================


class Indexer:
    """
        Indexer Class
    """

    def __init__(self, vocabs: list = None):
        self.vocabs = vocabs
        self._vocab2idx = None
        self._idx2vocab = None
        if self.vocabs:
            self._unitize_vocabs()  # build unique vocab

    def get_vocab2idx(self) -> dict:
        """
        get_vocab2idx method is written to return vocab2idx
        :return:
        """
        if not self._vocab2idx:
            self._empty_vocab_handler()
        return self._vocab2idx

    def get_idx2vocab(self) -> dict:
        """
        get_vocab2idx method is written to return idx2vocab
        :return:
        """
        if not self._idx2vocab:
            self._empty_vocab_handler()
        return self._idx2vocab

    @abstractmethod
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
        print("error handler")
        raise Exception("target is not available")

    @abstractmethod
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

        print("error handler")
        raise Exception("target is not available")

    @abstractmethod
    def build_vocab2idx(self) -> None:
        """
        build_vocab2idx method is written to build vocab2ix dictionary
        :return:
        """
        self._vocab2idx = {}
        for vocab in self.vocabs:
            self._vocab2idx[vocab] = len(self._vocab2idx)

    @abstractmethod
    def build_idx2vocab(self) -> None:
        """
        build_idx2vocab method is written to build idx2vocab dictionary
        :return:
        """
        self._idx2vocab = {}
        for vocab in self.vocabs:
            self._idx2vocab[len(self._idx2vocab)] = vocab

    def _empty_vocab_handler(self):
        """
        _empty_vocab_handler
        :return:
        """
        if not self.vocabs:
            raise Exception("Vocabs is None")
        self.build_vocab2idx()
        self.build_idx2vocab()

    def convert_samples_to_indexes(self, tokenized_samples: List[list]) -> List[list]:
        """
        convert_samples_to_indexes method is written to convert
        samples into their index
        :param tokenized_samples: [[target_1], ..., [target_n]]
        :return: [[target_1_index],...,[target_n_index]]
        """
        for index, tokenized_sample in enumerate(tokenized_samples):
            for token_index, token in enumerate(tokenized_sample):
                tokenized_samples[index][token_index] = self.get_idx(token)
        return tokenized_samples

    def convert_indexes_to_samples(self, indexed_samples: List[list]) -> List[list]:
        """
        convert_indexes_to_samples method is written to convert
        indexes to their tokens
        :param indexed_samples:
        :return:
        """
        for index, indexed_sample in enumerate(indexed_samples):
            for token_index, token in enumerate(indexed_sample):
                indexed_samples[index][token_index] = self.get_word(token)
        return indexed_samples

    def _unitize_vocabs(self) -> None:
        """

        :return:
        """
        self.vocabs = list(set(self.vocabs))

    def load(self, vocab2idx_path: str, idx2vocab_path: str):
        """

        Args:
            vocab2idx_path:
            idx2vocab_path:

        Returns:

        """
        self._vocab2idx = read_json(path=vocab2idx_path)
        self._idx2vocab = read_json(path=idx2vocab_path)

    def save(self, vocab2idx_path: str, idx2vocab_path: str) -> None:
        """

        Args:
            vocab2idx_path:
            idx2vocab_path:

        Returns:

        """
        write_json(data=self.get_vocab2idx(), path=vocab2idx_path)
        write_json(data=self.get_idx2vocab(), path=idx2vocab_path)
