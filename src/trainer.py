# -*- coding: utf-8 -*-
# ========================================================
"""trainer module is written for train model"""
# ========================================================


# ========================================================
# Imports
# ========================================================

import os
import logging
from sklearn.model_selection import train_test_split

from pytorch_lightning.loggers import CSVLogger
from transformers import T5Tokenizer, MT5Tokenizer

from configuration import BaseConfig
from data_prepration import prepare_av_data

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # create config instance
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    # create CSVLogger instance
    LOGGER = CSVLogger(save_dir=CONFIG.saved_model_path, name=CONFIG.model_name)

    # create LM Tokenizer instance
    # TOKENIZER = MT5Tokenizer.from_pretrained(CONFIG.language_model_tokenizer_path)

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

    logging.debug("We have {} train samples.".format(len(TRAIN_FIRST_AUTHORS)))
    logging.debug("We have {} validation samples.".format(len(VAL_FIRST_AUTHORS)))
    logging.debug("We have {} test samples.".format(len(TEST_FIRST_AUTHORS)))

