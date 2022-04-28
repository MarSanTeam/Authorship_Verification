# -*- coding: utf-8 -*-
# ========================================================
"""trainer module is written for train model"""
# ========================================================


# ========================================================
# Imports
# ========================================================

import logging
import os
import itertools

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from transformers import T5Tokenizer, T5EncoderModel

from configuration import BaseConfig
from data_loader import read_csv, write_json
from dataset import DataModule
from indexer import Indexer, TokenIndexer
from models.t5_encoder import Classifier
from utils import extract_punctuation
from data_prepration import word_tokenizer

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # -------------------------------- Create config instance----------------------------------
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()
    T5_TOKENIZER = T5Tokenizer.from_pretrained(ARGS.language_model_tokenizer_path)
    T5_MODEL = T5EncoderModel.from_pretrained(ARGS.language_model_path)

    # create CSVLogger instance
    LOGGER = CSVLogger(save_dir=ARGS.saved_model_path, name=ARGS.model_name)

    # ----------------------------------- Load raw data------------------------------------------
    TRAIN_DATA = read_csv(path=os.path.join(ARGS.processed_data_dir, ARGS.train_file),
                          columns=ARGS.data_headers,
                          names=ARGS.customized_headers).dropna()
    logging.info('train set contain %s sample ...', len(TRAIN_DATA))

    VALID_DATA = read_csv(path=os.path.join(ARGS.processed_data_dir, ARGS.dev_file),
                          columns=ARGS.data_headers,
                          names=ARGS.customized_headers).dropna()
    logging.info('valid set contain %s sample ...', len(VALID_DATA))

    TEST_DATA = read_csv(path=os.path.join(ARGS.processed_data_dir, ARGS.test_file),
                         columns=ARGS.data_headers,
                         names=ARGS.customized_headers).dropna()
    logging.info('test set contain %s sample ...', len(TEST_DATA))

    TRAIN_FIRST_TEXT_PUNCTUATIONS = extract_punctuation(TRAIN_DATA.first_text)
    TRAIN_SECOND_TEXT_PUNCTUATIONS = extract_punctuation(TRAIN_DATA.second_text)
    logging.info("Train punctuations are extracted")

    VALID_FIRST_TEXT_PUNCTUATIONS = extract_punctuation(VALID_DATA.first_text)
    VALID_SECOND_TEXT_PUNCTUATIONS = extract_punctuation(VALID_DATA.second_text)
    logging.info("Valid punctuations are extracted")

    TEST_FIRST_TEXT_PUNCTUATIONS = extract_punctuation(TEST_DATA.first_text)
    TEST_SECOND_TEXT_PUNCTUATIONS = extract_punctuation(TEST_DATA.second_text)
    logging.info("Test punctuations are extracted")

    # --------------------------------- Punctuation tokenization -------------------------------
    TRAIN_FIRST_TEXT_PUNCTUATIONS = word_tokenizer(TRAIN_FIRST_TEXT_PUNCTUATIONS, lambda x: x.split())
    TRAIN_SECOND_TEXT_PUNCTUATIONS = word_tokenizer(TRAIN_SECOND_TEXT_PUNCTUATIONS, lambda x: x.split())

    VALID_FIRST_TEXT_PUNCTUATIONS = word_tokenizer(VALID_FIRST_TEXT_PUNCTUATIONS, lambda x: x.split())
    VALID_SECOND_TEXT_PUNCTUATIONS = word_tokenizer(VALID_SECOND_TEXT_PUNCTUATIONS, lambda x: x.split())

    TEST_FIRST_TEXT_PUNCTUATIONS = word_tokenizer(TEST_FIRST_TEXT_PUNCTUATIONS, lambda x: x.split())
    TEST_SECOND_TEXT_PUNCTUATIONS = word_tokenizer(TEST_SECOND_TEXT_PUNCTUATIONS, lambda x: x.split())

    # ------------------------------------- Indexer --------------------------------------------
    TARGET_INDEXER = Indexer(vocabs=list(TRAIN_DATA.targets))
    TARGET_INDEXER.build_vocab2idx()

    TRAIN_TARGETS_CONVENTIONAL = [[target] for target in list(TRAIN_DATA.targets)]
    TRAIN_INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(TRAIN_TARGETS_CONVENTIONAL)

    TEST_TARGETS_CONVENTIONAL = [[target] for target in list(TEST_DATA.targets)]
    TEST_INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(TEST_TARGETS_CONVENTIONAL)

    VALID_TARGETS_CONVENTIONAL = [[target] for target in list(VALID_DATA.targets)]
    VALID_INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(VALID_TARGETS_CONVENTIONAL)

    PUNCTUATION_VOCABS = list(itertools.chain(*TRAIN_FIRST_TEXT_PUNCTUATIONS + TRAIN_SECOND_TEXT_PUNCTUATIONS))

    PUNCTUATION_INDEXER = TokenIndexer(vocabs=PUNCTUATION_VOCABS)
    PUNCTUATION_INDEXER.build_vocab2idx()

    TRAIN_FIRST_TEXT_PUNCTUATIONS = PUNCTUATION_INDEXER.convert_samples_to_indexes(TRAIN_FIRST_TEXT_PUNCTUATIONS)
    TRAIN_SECOND_TEXT_PUNCTUATIONS = PUNCTUATION_INDEXER.convert_samples_to_indexes(TRAIN_SECOND_TEXT_PUNCTUATIONS)

    VALID_FIRST_TEXT_PUNCTUATIONS = PUNCTUATION_INDEXER.convert_samples_to_indexes(VALID_FIRST_TEXT_PUNCTUATIONS)
    VALID_SECOND_TEXT_PUNCTUATIONS = PUNCTUATION_INDEXER.convert_samples_to_indexes(VALID_SECOND_TEXT_PUNCTUATIONS)

    TEST_FIRST_TEXT_PUNCTUATIONS = PUNCTUATION_INDEXER.convert_samples_to_indexes(TEST_FIRST_TEXT_PUNCTUATIONS)
    TEST_SECOND_TEXT_PUNCTUATIONS = PUNCTUATION_INDEXER.convert_samples_to_indexes(TEST_SECOND_TEXT_PUNCTUATIONS)

    # -------------------------------- Make DataLoader Dict ----------------------------------------
    # TRAIN_COLUMNS2DATA = {'first_text': list(TRAIN_DATA.first_text),
    #                       'second_text': list(TRAIN_DATA.second_text),
    #                       'targets': TRAIN_INDEXED_TARGET}
    #
    # VAL_COLUMNS2DATA = {'first_text': list(VALID_DATA.first_text),
    #                     'second_text': list(VALID_DATA.second_text),
    #                     'targets': VALID_INDEXED_TARGET}
    #
    # TEST_COLUMNS2DATA = {'first_text': list(TEST_DATA.first_text),
    #                      'second_text': list(TEST_DATA.second_text),
    #                      'targets': TEST_INDEXED_TARGET}
    #
    # DATA = {'train_data': TRAIN_COLUMNS2DATA,
    #         'val_data': VAL_COLUMNS2DATA, 'test_data': TRAIN_COLUMNS2DATA}
    #
    # # ----------------------------- Create Data Module ----------------------------------
    # DATA_MODULE = DataModule(data=DATA, config=ARGS, tokenizer=T5_TOKENIZER)
    # DATA_MODULE.setup()
    # CHECKPOINT_CALLBACK = ModelCheckpoint(monitor="val_loss",
    #                                       filename="QTag-{epoch:02d}-{val_loss:.2f}",
    #                                       save_top_k=ARGS.save_top_k,
    #                                       mode="min")
    # # -------------------------------- Instantiate the Model Trainer -----------------------------
    # EARLY_STOPPING_CALLBACK = EarlyStopping(monitor="val_loss", patience=5)
    # TRAINER = pl.Trainer(max_epochs=ARGS.n_epochs, gpus=[0],
    #                      callbacks=[CHECKPOINT_CALLBACK, EARLY_STOPPING_CALLBACK],
    #                      progress_bar_refresh_rate=60, logger=LOGGER)
    # # Create Model
    # STEPS_PER_EPOCH = len(TRAIN_DATA) // ARGS.batch_size
    # MODEL = Classifier(num_classes=len(set(list(TRAIN_DATA.targets))),
    #                    t5_model_path=ARGS.language_model_path,
    #                    lr=ARGS.lr, max_len=ARGS.max_len)
    # # Train and Test Model
    # TRAINER.fit(MODEL, datamodule=DATA_MODULE)
    # TRAINER.test(ckpt_path='best', datamodule=DATA_MODULE)
    #
    # # save best mt5_model_en path
    # write_json(path=ARGS.best_model_path_file,
    #            data={'best_model_path': CHECKPOINT_CALLBACK.best_model_path})
