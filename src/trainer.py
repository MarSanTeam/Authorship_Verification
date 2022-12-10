# -*- coding: utf-8 -*-
# ========================================================

"""
    Author Verification Project:
        src:
            trainer.py
"""

# ============================ Third Party libs ============================
import itertools
import logging
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from transformers import T5Tokenizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# ============================ My packages ============================
from configuration import BaseConfig
from data_loader import write_json, read_pickle, write_pickle
from data_preparation import prepare_av_data, split_data, AVFeatures
from dataset import DataModule
from indexer import Indexer, TokenIndexer
from models import build_checkpoint_callback
from models.t5_model import Classifier

# ==========================================================================


logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # -------------------------------- Create Config Instance---------------------
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    # ------------------------------ Create CSVLogger Instance -------------------
    LOGGER = CSVLogger(save_dir=ARGS.saved_model_path, name=ARGS.model_name)

    # -------------------------------- Load Data----------------------------------
    FIRST_AUTHORS_TEXTS, SECOND_AUTHORS_TEXTS, TARGETS = prepare_av_data(
        pair_data_path=os.path.join(ARGS.raw_data_dir, ARGS.pair_data),
        truth_data_path=os.path.join(ARGS.raw_data_dir, ARGS.truth_data)
    )
    assert len(FIRST_AUTHORS_TEXTS) == len(SECOND_AUTHORS_TEXTS) == len(TARGETS)

    TRAIN_FIRST_TEXT, DEV_FIRST_TEXT, TEST_FIRST_TEXT, TRAIN_SECOND_TEXT, DEV_SECOND_TEXT, \
    TEST_SECOND_TEXT, TRAIN_TARGETS, DEV_TARGETS, TEST_TARGETS = split_data(FIRST_AUTHORS_TEXTS,
                                                                            SECOND_AUTHORS_TEXTS,
                                                                            TARGETS)
    logging.info("train set contain %s sample ...", len(TRAIN_FIRST_TEXT))
    logging.info("validation set contain %s sample ...", len(DEV_FIRST_TEXT))
    logging.info("test set contain %s sample ...", len(TEST_FIRST_TEXT))

    # ------------------------------ Load T5 Tokenizer ---------------------------
    T5_TOKENIZER = T5Tokenizer.from_pretrained(ARGS.language_model_path)

    # ------------------------------ Extract Features -----------------------------------
    # features[0] --> pos
    # features[1] --> punctuation adn emoji
    # features[2] --> author-specific and topic-specific information
    if os.path.exists(os.path.join(ARGS.assets_dir, ARGS.features_file)):
        TRAIN_FIRST_TEXT_FEATURES, TRAIN_SECOND_TEXT_FEATURES, DEV_FIRST_TEXT_FEATURES, \
        DEV_SECOND_TEXT_FEATURES, TEST_FIRST_TEXT_FEATURES, TEST_SECOND_TEXT_FEATURES = read_pickle(
            os.path.join(ARGS.assets_dir, ARGS.features_file))
        logging.info("Features are loaded.")
    else:
        av_features_obj = AVFeatures(
            datasets=[TRAIN_FIRST_TEXT[:10], TRAIN_SECOND_TEXT[:10], DEV_FIRST_TEXT[:10],
                      DEV_SECOND_TEXT[:10],
                      TEST_FIRST_TEXT[:10], TEST_SECOND_TEXT[:10]],
            tokenizer=word_tokenize,
            pos_tagger=pos_tag)
        TRAIN_FIRST_TEXT_FEATURES, TRAIN_SECOND_TEXT_FEATURES, DEV_FIRST_TEXT_FEATURES, \
        DEV_SECOND_TEXT_FEATURES, TEST_FIRST_TEXT_FEATURES, \
        TEST_SECOND_TEXT_FEATURES = av_features_obj()
        write_pickle(
            [TRAIN_FIRST_TEXT_FEATURES, TRAIN_SECOND_TEXT_FEATURES, DEV_FIRST_TEXT_FEATURES,
             DEV_SECOND_TEXT_FEATURES, TEST_FIRST_TEXT_FEATURES, TEST_SECOND_TEXT_FEATURES],
            path=os.path.join(ARGS.assets_dir, ARGS.features_file))
        logging.info("Features are extracted.")

    # --------------------------------- Target Indexer ----------------------------------
    TARGET_INDEXER = Indexer(vocabs=TRAIN_TARGETS)
    TARGET_INDEXER.build_vocab2idx()
    TARGET_INDEXER.save(vocab2idx_path=os.path.join(ARGS.assets_dir, ARGS.target2index_file),
                        idx2vocab_path=os.path.join(ARGS.assets_dir, ARGS.index2target_file))

    TRAIN_TARGETS_CONVENTIONAL = [[target] for target in TRAIN_TARGETS]
    TRAIN_INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(TRAIN_TARGETS_CONVENTIONAL)

    DEV_TARGETS_CONVENTIONAL = [[target] for target in DEV_TARGETS]
    DEV_INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(DEV_TARGETS_CONVENTIONAL)

    TEST_TARGETS_CONVENTIONAL = [[target] for target in TEST_TARGETS]
    TEST_INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(TEST_TARGETS_CONVENTIONAL)
    # ---------------------------------- POS Indexer ----------------------------------
    POS_VOCABS = list(
        itertools.chain(*TRAIN_FIRST_TEXT_FEATURES[0] + TRAIN_SECOND_TEXT_FEATURES[0]))
    POS_INDEXER = TokenIndexer(vocabs=POS_VOCABS)
    POS_INDEXER.build_vocab2idx()
    POS_INDEXER.build_idx2vocab()
    POS_INDEXER.save(vocab2idx_path=os.path.join(ARGS.assets_dir, ARGS.pos2index_file),
                     idx2vocab_path=os.path.join(ARGS.assets_dir, ARGS.index2pos_file))

    TRAIN_FIRST_TEXT_POS = POS_INDEXER.convert_samples_to_indexes(TRAIN_FIRST_TEXT_FEATURES[0])
    TRAIN_SECOND_TEXT_POS = POS_INDEXER.convert_samples_to_indexes(TRAIN_SECOND_TEXT_FEATURES[0])
    DEV_FIRST_TEXT_POS = POS_INDEXER.convert_samples_to_indexes(DEV_FIRST_TEXT_FEATURES[0])
    DEV_SECOND_TEXT_POS = POS_INDEXER.convert_samples_to_indexes(DEV_SECOND_TEXT_FEATURES[0])
    TEST_FIRST_TEXT_POS = POS_INDEXER.convert_samples_to_indexes(TEST_FIRST_TEXT_FEATURES[0])
    TEST_SECOND_TEXT_POS = POS_INDEXER.convert_samples_to_indexes(TEST_SECOND_TEXT_FEATURES[0])

    # ---------------------------- Prepare of aggregated data -------------------------------
    TRAIN_COLUMNS2DATA = {"first_text": TRAIN_FIRST_TEXT[:10],
                          "second_text": TRAIN_SECOND_TEXT[:10],
                          "first_punctuations": TRAIN_FIRST_TEXT_FEATURES[1],
                          "second_punctuations": TRAIN_SECOND_TEXT_FEATURES[1],
                          "first_information": TRAIN_FIRST_TEXT_FEATURES[2],
                          "second_information": TRAIN_SECOND_TEXT_FEATURES[2],
                          "first_pos": TRAIN_FIRST_TEXT_POS,
                          "second_pos": TRAIN_SECOND_TEXT_POS,
                          "targets": TRAIN_INDEXED_TARGET}

    DEV_COLUMNS2DATA = {"first_text": DEV_FIRST_TEXT[:10],
                        "second_text": DEV_SECOND_TEXT[:10],
                        "first_punctuations": DEV_FIRST_TEXT_FEATURES[1],
                        "second_punctuations": DEV_SECOND_TEXT_FEATURES[1],
                        "first_information": DEV_FIRST_TEXT_FEATURES[2],
                        "second_information": DEV_SECOND_TEXT_FEATURES[2],
                        "first_pos": DEV_FIRST_TEXT_POS,
                        "second_pos": DEV_SECOND_TEXT_POS,
                        "targets": DEV_INDEXED_TARGET}

    TEST_COLUMNS2DATA = {"first_text": TEST_FIRST_TEXT[:10],
                         "second_text": TEST_SECOND_TEXT[:10],
                         "first_punctuations": TEST_FIRST_TEXT_FEATURES[1],
                         "second_punctuations": TEST_SECOND_TEXT_FEATURES[1],
                         "first_information": TEST_FIRST_TEXT_FEATURES[2],
                         "second_information": TEST_SECOND_TEXT_FEATURES[2],
                         "first_pos": TEST_FIRST_TEXT_POS,
                         "second_pos": TEST_SECOND_TEXT_POS,
                         "targets": TEST_INDEXED_TARGET}

    DATA = {"train_data": TRAIN_COLUMNS2DATA,
            "val_data": DEV_COLUMNS2DATA, "test_data": TEST_COLUMNS2DATA}

    # ----------------------------- Create Data Module -------------------------------
    DATA_MODULE = DataModule(data=DATA, config=ARGS, tokenizer=T5_TOKENIZER)
    DATA_MODULE.setup()

    # -------------------------- Instantiate the Model Trainer -----------------------
    EARLY_STOPPING_CALLBACK = EarlyStopping(monitor="val_acc", patience=7, mode="max")
    CHECKPOINT_CALLBACK = build_checkpoint_callback(save_top_k=ARGS.save_top_k,
                                                    monitor="val_acc",
                                                    mode="max",
                                                    filename="QTag-{epoch:02d}-{val_acc:.2f}")
    TRAINER = pl.Trainer(max_epochs=ARGS.n_epochs, gpus=[0],
                         callbacks=[CHECKPOINT_CALLBACK, EARLY_STOPPING_CALLBACK],
                         progress_bar_refresh_rate=60, logger=LOGGER)
    # ------------------------------- Create Model -------------------------------
    MODEL = Classifier(num_classes=len(TARGET_INDEXER.get_vocab2idx()),
                       t5_model_path=ARGS.language_model_path, lr=ARGS.lr,
                       max_len=ARGS.max_len, filter_sizes=ARGS.filter_sizes,
                       n_filters=ARGS.n_filters)

    # ---------------------------- Train and Test Model --------------------------
    TRAINER.fit(MODEL, datamodule=DATA_MODULE)
    TRAINER.test(ckpt_path="best", datamodule=DATA_MODULE)

    # --------------------------- save best mt5_model_en path -------------------
    write_json(path=ARGS.best_model_path_file,
               data={"best_model_path": CHECKPOINT_CALLBACK.best_model_path})
