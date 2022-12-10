# -*- coding: utf-8 -*-
# ========================================================

"""
    Author Verification Project:
        src:
            inferencer.py
"""

# ============================ Third Party libs ============================
import logging
import os
import numpy as np
import torch
from transformers import T5Tokenizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
# ============================ My packages ============================
from configuration import BaseConfig
from indexer import TokenIndexer
from data_preparation import prepare_av_data, AVFeatures
from dataset import ConcatDataset
from models.t5_model import Classifier

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    # -------------------------------- Load Data----------------------------------

    FIRST_TEXT, SECOND_TEXT, _ = prepare_av_data(
        pair_data_path=os.path.join(ARGS.raw_data_dir, ARGS.pair_data))
    logging.info("test set contain %s sample ...", len(FIRST_TEXT))

    # ------------------------------ Load T5 Tokenizer ---------------------------
    T5_TOKENIZER = T5Tokenizer.from_pretrained(ARGS.language_model_path)

    # ----------------------------- Load Model -----------------------------------
    MODEL = Classifier.load_from_checkpoint(ARGS.best_model_path)

    # ------------------------------ Extract Features -----------------------------
    # features[0] --> pos
    # features[1] --> punctuation adn emoji
    # features[2] --> author-specific and topic-specific information
    av_features_obj = AVFeatures(
        datasets=[FIRST_TEXT[:10], SECOND_TEXT[:10]],
        tokenizer=word_tokenize,
        pos_tagger=pos_tag)
    FIRST_TEXT_FEATURES, SECOND_TEXT_FEATURES = av_features_obj()
    logging.info("Features are extracted.")

    POS_INDEXER = TokenIndexer()
    POS_INDEXER.load(vocab2idx_path=os.path.join(ARGS.assets_dir, ARGS.pos2index_file),
                     idx2vocab_path=os.path.join(ARGS.assets_dir, ARGS.index2pos_file))

    FIRST_TEXT_FEATURES_POS = POS_INDEXER.convert_samples_to_indexes(FIRST_TEXT_FEATURES[0])
    SECOND_TEXT_FEATURES_POS = POS_INDEXER.convert_samples_to_indexes(SECOND_TEXT_FEATURES[0])

    # ---------------------------- Prepare of aggregated data -------------------------------
    COLUMNS2DATA = {"first_text": FIRST_TEXT[:10],
                    "second_text": SECOND_TEXT[:10],
                    "first_punctuations": FIRST_TEXT_FEATURES[1],
                    "second_punctuations": SECOND_TEXT_FEATURES[1],
                    "first_information": FIRST_TEXT_FEATURES[2],
                    "second_information": SECOND_TEXT_FEATURES[2],
                    "first_pos": FIRST_TEXT_FEATURES_POS,
                    "second_pos": SECOND_TEXT_FEATURES_POS}

    # ------------------------- Create dataloader -----------------------------------
    DATASET = ConcatDataset(data=COLUMNS2DATA,
                            tokenizer=T5_TOKENIZER,
                            max_len=ARGS.max_len)

    DATALOADER = torch.utils.data.DataLoader(DATASET, batch_size=ARGS.batch_size,
                                             shuffle=False, num_workers=ARGS.num_workers)

    # ------------------------- Make Prediction -------------------------------------

    PREDICTIONS = []

    for i_batch, sample_batched in enumerate(DATALOADER):
        OUTPUT = MODEL(sample_batched)
        OUTPUT = torch.softmax(OUTPUT, dim=1)
        TARGETS = np.argmax(OUTPUT.detach().numpy(), axis=1)
        PREDICTIONS.extend(TARGETS)
    print(PREDICTIONS)
