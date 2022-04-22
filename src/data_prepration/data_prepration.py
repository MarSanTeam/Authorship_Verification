import json
from typing import List


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


if __name__ == '__main__':
    FIRST_AUTHORS_TEXTS, SECOND_AUTHORS_TEXTS, TARGETS = prepare_av_data(
        pair_data_path=os.path.join(CONFIG.raw_data_dir, CONFIG.pair_data),
        truth_data_path=os.path.join(CONFIG.raw_data_dir, CONFIG.truth_data))

    assert len(FIRST_AUTHORS_TEXTS) == len(SECOND_AUTHORS_TEXTS) == len(TARGETS)

    logging.debug("We have {} samples.".format(len(FIRST_AUTHORS_TEXTS)))

    TRAIN_FIRST_AUTHORS, TEST_FIRST_AUTHORS, TRAIN_SECOND_AUTHORS, TEST_SECOND_AUTHORS, \
    TRAIN_TARGETS, TEST_TARGETS = train_test_split(FIRST_AUTHORS_TEXTS,
                                                   SECOND_AUTHORS_TEXTS, TARGETS,
                                                   test_size=0.3, random_state=1234)

    VAL_FIRST_AUTHORS, TEST_FIRST_AUTHORS, VAL_SECOND_AUTHORS, TEST_SECOND_AUTHORS, \
    VAL_TARGETS, TEST_TARGETS = train_test_split(TEST_FIRST_AUTHORS,
                                                 TEST_SECOND_AUTHORS, TEST_TARGETS,
                                                 test_size=0.5, random_state=1234)
    df_train = pd.DataFrame({"first_text": TRAIN_FIRST_AUTHORS,
                             "second_text": TRAIN_SECOND_AUTHORS,
                             "lable": TRAIN_TARGETS})
    df_train.to_csv("train_data.csv", index=False)
    df_val = pd.DataFrame({"first_text": VAL_FIRST_AUTHORS,
                           "second_text": VAL_SECOND_AUTHORS,
                           "lable": VAL_TARGETS})
    df_val.to_csv("val_data.csv", index=False)
    df_test = pd.DataFrame({"first_text": TEST_FIRST_AUTHORS,
                            "second_text": TEST_SECOND_AUTHORS,
                            "lable": TEST_TARGETS})
    df_test.to_csv("test_data.csv", index=False)
