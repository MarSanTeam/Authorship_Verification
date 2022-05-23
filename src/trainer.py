# -*- coding: utf-8 -*-
# ========================================================
"""trainer module is written for train model"""
# ========================================================


# ========================================================
# Imports
# ========================================================

import itertools
import logging
import os
import itertools

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from transformers import T5Tokenizer

from configuration import BaseConfig
<<<<<<< HEAD
from data_loader import read_csv, write_json
from dataset import DataModule
from indexer import Indexer, TokenIndexer
from models.t5_encoder import Classifier
from utils import extract_punctuation, create_punc_pair
from data_prepration import word_tokenizer
=======
from data_loader import read_csv, write_json, read_pickle, write_pickle
from dataset import DataModule
from indexer import Indexer, TokenIndexer
from models import build_checkpoint_callback
from models.t5_encoder_pos import Classifier
from utils import extract_information, extract_punctuation_emoji, extract_pos
>>>>>>> add_architecture

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # -------------------------------- Create config instance----------------------------------
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()
    T5_TOKENIZER = T5Tokenizer.from_pretrained(ARGS.language_model_tokenizer_path)

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
    # ---------------------------------------extract pos---------------------------------

    # TRAIN_FIRST_TEXT_POS = extract_pos(TRAIN_DATA.first_text)
    # TRAIN_SECOND_TEXT_POS = extract_pos(TRAIN_DATA.second_text)
    # logging.info("Train pos are extracted")
    #
    # VALID_FIRST_TEXT_POS = extract_pos(VALID_DATA.first_text)
    # VALID_SECOND_TEXT_POS = extract_pos(VALID_DATA.second_text)
    # logging.info("Valid pos are extracted")
    #
    # TEST_FIRST_TEXT_POS = extract_pos(TEST_DATA.first_text)
    # TEST_SECOND_TEXT_POS = extract_pos(TEST_DATA.second_text)
    # logging.info("Test pos are extracted")
    # ---------------------------- Write POS ------------------------------------
    # write_pickle("../data/POS/TRAIN_FIRST_TEXT_POS.pkl", TRAIN_FIRST_TEXT_POS)
    # write_pickle("../data/POS/TRAIN_SECOND_TEXT_POS.pkl", TRAIN_SECOND_TEXT_POS)
    # logging.info("Train pos are saved")
    #
    # write_pickle("../data/POS/VALID_FIRST_TEXT_POS.pkl", VALID_FIRST_TEXT_POS)
    # write_pickle("../data/POS/VALID_SECOND_TEXT_POS.pkl", VALID_SECOND_TEXT_POS)
    # logging.info("Valid pos are saved")
    #
    # write_pickle("../data/POS/TEST_FIRST_TEXT_POS.pkl", TEST_FIRST_TEXT_POS)
    # write_pickle("../data/POS/TEST_SECOND_TEXT_POS.pkl", TEST_SECOND_TEXT_POS)
    # logging.info("Test pos are saved")
    # ----------------------------- Read POS --------------------------------------------
    TRAIN_FIRST_TEXT_POS = read_pickle("../data/POS/TRAIN_FIRST_TEXT_POS.pkl")
    TRAIN_SECOND_TEXT_POS = read_pickle("../data/POS/TRAIN_SECOND_TEXT_POS.pkl")
    logging.info("Train pos are extracted")
    VALID_FIRST_TEXT_POS = read_pickle("../data/POS/VALID_FIRST_TEXT_POS.pkl")
    VALID_SECOND_TEXT_POS = read_pickle("../data/POS/VALID_SECOND_TEXT_POS.pkl")
    logging.info("Valid pos are extracted")
    TEST_FIRST_TEXT_POS = read_pickle("../data/POS/TEST_FIRST_TEXT_POS.pkl")
    TEST_SECOND_TEXT_POS = read_pickle("../data/POS/TEST_SECOND_TEXT_POS.pkl")
    logging.info("Test pos are extracted")
    # --------------------------extract_punctuation and emoji-----------------------------

    TRAIN_FIRST_TEXT_PUNCTUATIONS = extract_punctuation_emoji(TRAIN_DATA.first_text)
    TRAIN_SECOND_TEXT_PUNCTUATIONS = extract_punctuation_emoji(TRAIN_DATA.second_text)

    logging.info("Train punctuations are extracted")

    VALID_FIRST_TEXT_PUNCTUATIONS = extract_punctuation_emoji(VALID_DATA.first_text)
    VALID_SECOND_TEXT_PUNCTUATIONS = extract_punctuation_emoji(VALID_DATA.second_text)
    logging.info("Valid punctuations are extracted")

    TEST_FIRST_TEXT_PUNCTUATIONS = extract_punctuation_emoji(TEST_DATA.first_text)
    TEST_SECOND_TEXT_PUNCTUATIONS = extract_punctuation_emoji(TEST_DATA.second_text)
    logging.info("Test punctuations are extracted")
    # --------------------------extract_punctuation-----------------------------

    # TRAIN_FIRST_TEXT_PUNCTUATIONS = extract_punctuation(TRAIN_DATA.first_text)
    # TRAIN_SECOND_TEXT_PUNCTUATIONS = extract_punctuation(TRAIN_DATA.second_text)
    # logging.info("Train punctuations are extracted")
    #
    # VALID_FIRST_TEXT_PUNCTUATIONS = extract_punctuation(VALID_DATA.first_text)
    # VALID_SECOND_TEXT_PUNCTUATIONS = extract_punctuation(VALID_DATA.second_text)
    # logging.info("Valid punctuations are extracted")
    #
    # TEST_FIRST_TEXT_PUNCTUATIONS = extract_punctuation(TEST_DATA.first_text)
    # TEST_SECOND_TEXT_PUNCTUATIONS = extract_punctuation(TEST_DATA.second_text)
    # logging.info("Test punctuations are extracted")

    TRAIN_FIRST_TEXT_INFORMATION = extract_information(TRAIN_DATA.first_text)
    TRAIN_SECOND_TEXT_INFORMATION = extract_information(TRAIN_DATA.second_text)
    logging.info("Train information are extracted")
    #
    VALID_FIRST_TEXT_INFORMATION = extract_information(VALID_DATA.first_text)
    VALID_SECOND_TEXT_INFORMATION = extract_information(VALID_DATA.second_text)
    logging.info("Valid information are extracted")

    TEST_FIRST_TEXT_INFORMATION = extract_information(TEST_DATA.first_text)
    TEST_SECOND_TEXT_INFORMATION = extract_information(TEST_DATA.second_text)
    logging.info("Test information are extracted")

    # --------------------------------- Punctuation tokenization -------------------------------
    # TRAIN_FIRST_TEXT_PUNCTUATIONS = word_tokenizer(TRAIN_FIRST_TEXT_PUNCTUATIONS, lambda x: x.split())
    # TRAIN_SECOND_TEXT_PUNCTUATIONS = word_tokenizer(TRAIN_SECOND_TEXT_PUNCTUATIONS, lambda x: x.split())
    #
    # VALID_FIRST_TEXT_PUNCTUATIONS = word_tokenizer(VALID_FIRST_TEXT_PUNCTUATIONS, lambda x: x.split())
    # VALID_SECOND_TEXT_PUNCTUATIONS = word_tokenizer(VALID_SECOND_TEXT_PUNCTUATIONS, lambda x: x.split())
    #
    # TEST_FIRST_TEXT_PUNCTUATIONS = word_tokenizer(TEST_FIRST_TEXT_PUNCTUATIONS, lambda x: x.split())
    # TEST_SECOND_TEXT_PUNCTUATIONS = word_tokenizer(TEST_SECOND_TEXT_PUNCTUATIONS, lambda x: x.split())

<<<<<<< HEAD
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
    # TRAIN_FIRST_TEXT_PUNCTUATIONS = word_tokenizer(TRAIN_FIRST_TEXT_PUNCTUATIONS, lambda x: x.split())
    # TRAIN_SECOND_TEXT_PUNCTUATIONS = word_tokenizer(TRAIN_SECOND_TEXT_PUNCTUATIONS, lambda x: x.split())
    #
    # VALID_FIRST_TEXT_PUNCTUATIONS = word_tokenizer(VALID_FIRST_TEXT_PUNCTUATIONS, lambda x: x.split())
    # VALID_SECOND_TEXT_PUNCTUATIONS = word_tokenizer(VALID_SECOND_TEXT_PUNCTUATIONS, lambda x: x.split())
    #
    # TEST_FIRST_TEXT_PUNCTUATIONS = word_tokenizer(TEST_FIRST_TEXT_PUNCTUATIONS, lambda x: x.split())
    # TEST_SECOND_TEXT_PUNCTUATIONS = word_tokenizer(TEST_SECOND_TEXT_PUNCTUATIONS, lambda x: x.split())

=======
>>>>>>> add_architecture
    # ------------------------------------- Indexer --------------------------------------------
    TARGET_INDEXER = Indexer(vocabs=list(TRAIN_DATA.targets))
    TARGET_INDEXER.build_vocab2idx()

    TRAIN_TARGETS_CONVENTIONAL = [[target] for target in list(TRAIN_DATA.targets)]
    TRAIN_INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(TRAIN_TARGETS_CONVENTIONAL)

    TEST_TARGETS_CONVENTIONAL = [[target] for target in list(TEST_DATA.targets)]
    TEST_INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(TEST_TARGETS_CONVENTIONAL)

    VALID_TARGETS_CONVENTIONAL = [[target] for target in list(VALID_DATA.targets)]
    VALID_INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(VALID_TARGETS_CONVENTIONAL)
<<<<<<< HEAD

    # PUNCTUATION_VOCABS = list(itertools.chain(*TRAIN_FIRST_TEXT_PUNCTUATIONS + TRAIN_SECOND_TEXT_PUNCTUATIONS))
    #
=======
    # ------------------------------------- Token Indexer -----------------------------------------

    # PUNCTUATION_VOCABS = list(itertools.chain(*TRAIN_FIRST_TEXT_PUNCTUATIONS + TRAIN_SECOND_TEXT_PUNCTUATIONS))
    # PUNCTUATION_INDEXER = TokenIndexer(vocabs=PUNCTUATION_VOCABS)
    # PUNCTUATION_INDEXER.build_vocab2idx()

    # TRAIN_FIRST_TEXT_PUNCTUATIONS = PUNCTUATION_INDEXER.convert_samples_to_indexes(TRAIN_FIRST_TEXT_PUNCTUATIONS)
    # TRAIN_SECOND_TEXT_PUNCTUATIONS = PUNCTUATION_INDEXER.convert_samples_to_indexes(TRAIN_SECOND_TEXT_PUNCTUATIONS)
    # VALID_FIRST_TEXT_PUNCTUATIONS = PUNCTUATION_INDEXER.convert_samples_to_indexes(VALID_FIRST_TEXT_PUNCTUATIONS)
    # VALID_SECOND_TEXT_PUNCTUATIONS = PUNCTUATION_INDEXER.convert_samples_to_indexes(VALID_SECOND_TEXT_PUNCTUATIONS)
    # TEST_FIRST_TEXT_PUNCTUATIONS = PUNCTUATION_INDEXER.convert_samples_to_indexes(TEST_FIRST_TEXT_PUNCTUATIONS)
    # TEST_SECOND_TEXT_PUNCTUATIONS = PUNCTUATION_INDEXER.convert_samples_to_indexes(TEST_SECOND_TEXT_PUNCTUATIONS)

    POS_VOCABS = list(itertools.chain(*TRAIN_FIRST_TEXT_POS + TRAIN_SECOND_TEXT_POS))
    POS_INDEXER = TokenIndexer(vocabs=POS_VOCABS)
    POS_INDEXER.build_vocab2idx()

    TRAIN_FIRST_TEXT_POS = POS_INDEXER.convert_samples_to_indexes(TRAIN_FIRST_TEXT_POS)
    TRAIN_SECOND_TEXT_POS = POS_INDEXER.convert_samples_to_indexes(TRAIN_SECOND_TEXT_POS)
    VALID_FIRST_TEXT_POS = POS_INDEXER.convert_samples_to_indexes(VALID_FIRST_TEXT_POS)
    VALID_SECOND_TEXT_POS = POS_INDEXER.convert_samples_to_indexes(VALID_SECOND_TEXT_POS)
    TEST_FIRST_TEXT_POS = POS_INDEXER.convert_samples_to_indexes(TEST_FIRST_TEXT_POS)
    TEST_SECOND_TEXT_POS = POS_INDEXER.convert_samples_to_indexes(TEST_SECOND_TEXT_POS)

    # PUNCTUATION_VOCABS = list(itertools.chain(*TRAIN_FIRST_TEXT_PUNCTUATIONS + TRAIN_SECOND_TEXT_PUNCTUATIONS))

>>>>>>> add_architecture
    # PUNCTUATION_INDEXER = TokenIndexer(vocabs=PUNCTUATION_VOCABS)
    # PUNCTUATION_INDEXER.build_vocab2idx()
    #
    # TRAIN_FIRST_TEXT_PUNCTUATIONS = PUNCTUATION_INDEXER.convert_samples_to_indexes(TRAIN_FIRST_TEXT_PUNCTUATIONS)
    # TRAIN_SECOND_TEXT_PUNCTUATIONS = PUNCTUATION_INDEXER.convert_samples_to_indexes(TRAIN_SECOND_TEXT_PUNCTUATIONS)
    #
    # VALID_FIRST_TEXT_PUNCTUATIONS = PUNCTUATION_INDEXER.convert_samples_to_indexes(VALID_FIRST_TEXT_PUNCTUATIONS)
    # VALID_SECOND_TEXT_PUNCTUATIONS = PUNCTUATION_INDEXER.convert_samples_to_indexes(VALID_SECOND_TEXT_PUNCTUATIONS)
    #
    # TEST_FIRST_TEXT_PUNCTUATIONS = PUNCTUATION_INDEXER.convert_samples_to_indexes(TEST_FIRST_TEXT_PUNCTUATIONS)
    # TEST_SECOND_TEXT_PUNCTUATIONS = PUNCTUATION_INDEXER.convert_samples_to_indexes(TEST_SECOND_TEXT_PUNCTUATIONS)
    #
    # TRAIN_PUNCTUATIONS = create_punc_pair(TRAIN_FIRST_TEXT_PUNCTUATIONS, TRAIN_SECOND_TEXT_PUNCTUATIONS)
    # VALID_PUNCTUATIONS = create_punc_pair(VALID_FIRST_TEXT_PUNCTUATIONS, VALID_SECOND_TEXT_PUNCTUATIONS)
    # TEST_PUNCTUATIONS = create_punc_pair(TEST_FIRST_TEXT_PUNCTUATIONS, TEST_SECOND_TEXT_PUNCTUATIONS)
<<<<<<< HEAD

=======
    # TRAIN_POS = create_sample_pair(ARGS, TRAIN_FIRST_TEXT_POS, TRAIN_SECOND_TEXT_POS)
    # VALID_POS = create_sample_pair(ARGS, VALID_FIRST_TEXT_POS, VALID_SECOND_TEXT_POS)
    # TEST_POS = create_sample_pair(ARGS, TEST_FIRST_TEXT_POS, TEST_SECOND_TEXT_POS)
>>>>>>> add_architecture
    # -------------------------------- Make DataLoader Dict ----------------------------------------
    TRAIN_COLUMNS2DATA = {"first_text": list(TRAIN_DATA.first_text),
                          "second_text": list(TRAIN_DATA.second_text),
                          "first_punctuations": TRAIN_FIRST_TEXT_PUNCTUATIONS,
                          "second_punctuations": TRAIN_SECOND_TEXT_PUNCTUATIONS,
<<<<<<< HEAD
=======
                          "first_information": TRAIN_FIRST_TEXT_INFORMATION,
                          "second_information": TRAIN_SECOND_TEXT_INFORMATION,
                          "first_pos": TRAIN_FIRST_TEXT_POS,
                          "second_pos": TRAIN_SECOND_TEXT_POS,
>>>>>>> add_architecture
                          "targets": TRAIN_INDEXED_TARGET}

    VAL_COLUMNS2DATA = {"first_text": list(VALID_DATA.first_text),
                        "second_text": list(VALID_DATA.second_text),
                        "first_punctuations": VALID_FIRST_TEXT_PUNCTUATIONS,
                        "second_punctuations": VALID_SECOND_TEXT_PUNCTUATIONS,
<<<<<<< HEAD
=======
                        "first_information": VALID_FIRST_TEXT_INFORMATION,
                        "second_information": VALID_SECOND_TEXT_INFORMATION,
                        "first_pos": VALID_FIRST_TEXT_POS,
                        "second_pos": VALID_SECOND_TEXT_POS,
>>>>>>> add_architecture
                        "targets": VALID_INDEXED_TARGET}

    TEST_COLUMNS2DATA = {"first_text": list(TEST_DATA.first_text),
                         "second_text": list(TEST_DATA.second_text),
                         "first_punctuations": TEST_FIRST_TEXT_PUNCTUATIONS,
                         "second_punctuations": TEST_SECOND_TEXT_PUNCTUATIONS,
<<<<<<< HEAD
=======
                         "first_information": TEST_FIRST_TEXT_INFORMATION,
                         "second_information": TEST_SECOND_TEXT_INFORMATION,
                         "first_pos": TEST_FIRST_TEXT_POS,
                         "second_pos": TEST_SECOND_TEXT_POS,
>>>>>>> add_architecture
                         "targets": TEST_INDEXED_TARGET}

    DATA = {"train_data": TRAIN_COLUMNS2DATA,
            "val_data": VAL_COLUMNS2DATA, "test_data": TEST_COLUMNS2DATA}

    # ----------------------------- Create Data Module ----------------------------------
    DATA_MODULE = DataModule(data=DATA, config=ARGS, tokenizer=T5_TOKENIZER)
    DATA_MODULE.setup()
    CHECKPOINT_CALLBACK = build_checkpoint_callback(save_top_k=ARGS.save_top_k,
                                                    monitor="val_acc",
                                                    mode="max",
                                                    filename="QTag-{epoch:02d}-{val_acc:.2f}")
    # -------------------------------- Instantiate the Model Trainer -----------------------------
<<<<<<< HEAD
    EARLY_STOPPING_CALLBACK = EarlyStopping(monitor="val_loss", patience=5)
    TRAINER = pl.Trainer(max_epochs=ARGS.n_epochs, gpus=[1],
=======
    EARLY_STOPPING_CALLBACK = EarlyStopping(monitor="val_acc", patience=7, mode="max")
    TRAINER = pl.Trainer(max_epochs=ARGS.n_epochs, gpus=[0],
>>>>>>> add_architecture
                         callbacks=[CHECKPOINT_CALLBACK, EARLY_STOPPING_CALLBACK],
                         progress_bar_refresh_rate=60, logger=LOGGER)
    # Create Model
    STEPS_PER_EPOCH = len(TRAIN_DATA) // ARGS.batch_size
    MODEL = Classifier(num_classes=len(set(list(TRAIN_DATA.targets))),
                       t5_model_path=ARGS.language_model_path, lr=ARGS.lr,
                       max_len=ARGS.max_len, embedding_dim=ARGS.embedding_dim,
<<<<<<< HEAD
                       vocab_size=10,#len(PUNCTUATION_INDEXER.get_vocab2idx())+2,
                       pad_idx=0, filter_sizes=ARGS.filter_sizes, n_filters=ARGS.n_filters)
=======
                       pad_idx=0, filter_sizes=ARGS.filter_sizes, n_filters=ARGS.n_filters)
    # MODEL = Classifier(args=ARGS, num_classes=len(set(list(TRAIN_DATA.targets))))

>>>>>>> add_architecture
    # Train and Test Model
    TRAINER.fit(MODEL, datamodule=DATA_MODULE)
    TRAINER.test(ckpt_path="best", datamodule=DATA_MODULE)

    # save best mt5_model_en path
    write_json(path=ARGS.best_model_path_file,
               data={"best_model_path": CHECKPOINT_CALLBACK.best_model_path})
