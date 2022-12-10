# -*- coding: utf-8 -*-
# ========================================================

"""
    Author Verification Project:
        data_preparation:
            extract_features.py
"""
# ============================ Third Party libs ============================
import re
import string
from typing import List
import emoji


class AVFeatures:
    """
    extract pos, punctuation, emoji and author-specific and topic-specific information

    Attributes:
        datasets:
        tokenizer: tokenizer object
        pos_tagger: pos tagger object

    """

    def __init__(self, datasets: List[List[str]], tokenizer, pos_tagger):
        self.datasets = datasets
        self.tokenizer = tokenizer
        self.pos_tagger = pos_tagger

    def __call__(self):
        outputs = []
        for data in self.datasets:
            pos = self.extract_pos(data)
            punc = self.extract_punctuation_emoji(data)
            info = self.extract_information(data)
            outputs.append([pos, punc, info])
        return outputs

    def extract_pos(self, documents: List[str]) -> List[List[str]]:
        """
        extract pos from list of documents
        Args:
            documents: list of documents

        Returns:
            List of pos tag for each documents

        """
        return [[word[1] for word in self.pos_tagger(self.tokenizer(text))] for text in
                documents]

    @staticmethod
    def extract_punctuation_emoji(documents) -> List[str]:
        """
        extract punctuation and emojis from input documents
        ARGS:
            documents: list of documents


        Returns:
            List of extracted punctuation from each document

        """
        punctuations = []
        punc = set(string.punctuation)
        emj = set(emoji.UNICODE_EMOJI["en"])
        pattern = r"(?<=\<).*?(?=\>)"
        exclude = punc | emj
        exclude.remove(">")
        exclude.remove("<")
        for doc in documents:
            doc = re.sub(pattern, "", doc)
            punc = " ".join(ch for ch in doc if ch in exclude)
            punctuations.append(punc)
        return punctuations

    @staticmethod
    def extract_information(documents) -> List[str]:
        """
        extract author-specific and topic-specific information

        Args:
            documents: list of documents

        Returns:
            List of extracted author-specific and topic-specific information from each document


        """
        extracted_information = []
        pattern = r"(?<=\<).*?(?=\>)"
        for doc in documents:
            doc = re.findall(pattern, doc)
            extracted_information.append(" ".join(doc))
        return extracted_information
