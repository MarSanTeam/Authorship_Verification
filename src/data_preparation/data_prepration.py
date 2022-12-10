# -*- coding: utf-8 -*-
# ========================================================

"""
    Author Verification Project:
        data_preparation:
            data_preparation.py
"""

# ============================ Third Party libs ============================
import json
from typing import List
from sklearn.model_selection import train_test_split


def prepare_av_data(pair_data_path: str,
                    truth_data_path: str = None) -> [List[str], List[str], List[int]]:
    """
    prepare data for training author verification model
    Args:
        pair_data_path:
        truth_data_path:

    Returns:

    """
    targets = []
    first_authors_texts = []
    second_authors_texts = []
    if truth_data_path:
        for line in open(truth_data_path, encoding="utf8"):
            data = json.loads(line.strip())
            targets.append(int(data["same"]))

    for line in open(pair_data_path, encoding="utf8"):
        data = json.loads(line.strip())
        first_authors_texts.append(data["pair"][0])
        second_authors_texts.append(data["pair"][1])

    return first_authors_texts, second_authors_texts, targets


def split_data(first_texts: List[str],
               second_texts: List[str],
               targets: List[int],
               test_size=0.3) -> [List[str], List[str], List[str], List[str], List[str], List[str],
                                  List[int], List[int], List[int]]:
    """
    split data in train, dev and test categories
    Args:
        first_texts: First texts
        second_texts: Second texts
        targets: targets, which shows whether the author of both texts is the same or not
        test_size: Specifying the percentage of test data from the original data

    Returns:
        List of training data

    """
    train_first_text, test_first_text, train_second_text, test_second_text, train_targets, \
    test_targets = train_test_split(first_texts, second_texts, targets, test_size=test_size,
                                    random_state=1234)

    dev_first_text, test_first_text, dev_second_text, test_second_text, dev_targets, \
    test_targets = train_test_split(test_first_text, test_second_text, test_targets, test_size=0.5,
                                    random_state=1234)
    return train_first_text, dev_first_text, test_first_text, train_second_text, dev_second_text, \
           test_second_text, train_targets, dev_targets, test_targets
