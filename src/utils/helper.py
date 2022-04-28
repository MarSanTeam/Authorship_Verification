from typing import List
import string
import re


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
        texts[idx].extend([pad_item] * (max_length - text_length))
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
            texts[idx].append("[SEP]")
    return texts
