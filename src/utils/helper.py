# import nltk
# nltk.download('punkt')
# from nltk.tokenize import word_tokenize
# from nltk import pos_tag
import re
import string
from typing import List

import emoji


def extract_pos(texts: List[str]) -> List[str]:
    poses = []
    for text in texts:
        pos_taged = pos_tag(word_tokenize(text))
        pos_taged = [pos_tag(word)[0][1] for word in pos_taged]

        poses.append(pos_taged)
    return poses


def extract_punctuation_emoji(texts: List[str]) -> List[str]:
    """

    :param texts:
    :return:
    """
    punctuations = []
    punc = set(string.punctuation)
    emj = set(emoji.UNICODE_EMOJI['en'])

    pattern = r"(?<=\<).*?(?=\>)"
    exclude = punc | emj
    exclude.remove(">")
    exclude.remove("<")
    for txt in texts:
        txt = re.sub(pattern, "", txt)
        punc = " ".join(ch for ch in txt if ch in exclude)
        punctuations.append(punc)
    return punctuations


def extract_punctuation(texts: List[str]) -> List[str]:
    """

    :param texts:
    :return:
    """
    punctuations = []
    exclude = set(string.punctuation)
    pattern = r"(?<=\<).*?(?=\>)"
    exclude = exclude - {"<", ">"}
    for text in texts:
        text = re.sub(pattern, "", text)
        punc = " ".join(ch for ch in text if ch in exclude)
        punctuations.append(punc)
    return punctuations


def extract_information(texts: List[str]) -> List[str]:
    """

    :param texts:
    :return:
    """
    extracted_information = []
    pattern = r"(?<=\<).*?(?=\>)"
    for text in texts:
        text = re.findall(pattern, text)
        extracted_information.append(" ".join(text))

    return extracted_information


def pad_sequence(texts: List[list], max_length: int, pad_item: str = "[PAD]") -> List[list]:
    """

    :param texts: [["item_1", "item_2", "item_3"], ["item_1", "item_2"]]
    :param max_length: 4
    :param pad_item: pad_item
    :return: [["item_1", "item_2", "item_3", pad_item],
                    ["item_1", "item_2", pad_item, pad_item]]
    """
    for idx, text in enumerate(texts):
        text_length = len(text)
        texts[idx].extend([0] * (max_length - text_length))
    return texts


def truncate_sequence(texts: List[list], max_length: int) -> list:
    """

    :param texts: [["item_1", "item_2", "item_3"], ["item_1", "item_2"]]
    :param max_length: 2
    :return: [["item_1", "item_2"], ["item_1", "item_2"]]
    """
    for idx, text in enumerate(texts):
        if len(text) > max_length:
            texts[idx] = text[: max_length - 1]
            texts[idx].append(29)
    return texts


def create_punc_pair(first_texts: List[list], second_texts: List[list]) -> List[list]:
    """

    :param first_texts:
    :param second_texts:
    :return:
    """
    data = []
    for first_text, second_text in zip(first_texts, second_texts):
        pair_text = first_text + [28] + second_text
        data.append(pair_text)
    data = pad_sequence(data, max_length=100)
    data = truncate_sequence(data, max_length=100)
    return data
