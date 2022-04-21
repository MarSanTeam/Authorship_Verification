from typing import List

import json


def prepare_av_data(pair_data_path: str, truth_data_path: str) -> [List[str], List[str], List[int]]:
    """

    :param truth_data_path:
    :param pair_data_path:
    :return:
    """
    targets = []
    first_authors_texts = []
    second_authors_texts = []
    for line in open(truth_data_path, encoding="utf8"):
        data = json.loads(line.strip())
        targets.append(int(data["same"]))

    for line in open(pair_data_path, encoding="utf8"):
        data = json.loads(line.strip())
        first_authors_texts.append(data["pair"][0])
        second_authors_texts.append(data["pair"][1])

    return first_authors_texts, second_authors_texts, targets
