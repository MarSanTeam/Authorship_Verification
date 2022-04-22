# -*- coding: utf-8 -*-
# ========================================================
"""trainer module is written for train model"""
# ========================================================


# ========================================================
# Imports
# ========================================================

import logging
import os

from pytorch_lightning.loggers import CSVLogger
from transformers import T5Tokenizer, T5EncoderModel

from configuration import BaseConfig
from data_loader import read_csv

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # create config instance
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()
    MT5_TOKENIZER = T5Tokenizer.from_pretrained(ARGS.language_model_tokenizer_path)
    MT5_MODEL = T5EncoderModel.from_pretrained(ARGS.language_model_path)

    # create CSVLogger instance
    LOGGER = CSVLogger(save_dir=ARGS.saved_model_path, name=ARGS.model_name)

    # load raw data
    TRAIN_DATA = read_csv(path=os.path.join(ARGS.processed_data_dir, ARGS.train_file),
                          columns=ARGS.data_headers,
                          names=ARGS.customized_headers).dropna()
    logging.info('train set contain %s sample ...', len(TRAIN_DATA))

    TEST_DATA = read_csv(path=os.path.join(ARGS.processed_data_dir, ARGS.test_file),
                         columns=ARGS.data_headers,
                         names=ARGS.customized_headers).dropna()
    logging.info('test set contain %s sample ...', len(TEST_DATA))

    VALID_DATA = read_csv(path=os.path.join(ARGS.processed_data_dir, ARGS.dev_file),
                          columns=ARGS.data_headers,
                          names=ARGS.customized_headers).dropna()
    logging.info('valid set contain %s sample ...', len(VALID_DATA))

