from typing import List
import string


def extract_punctuation(texts: List[str]) -> List[str]:
    """

    :param texts:
    :return:
    """
    punctuations = []
    exclude = set(string.punctuation)
    for text in texts:
        punc = "".join(ch for ch in text if ch in exclude)
        punctuations.append(punc)
    return punctuations
