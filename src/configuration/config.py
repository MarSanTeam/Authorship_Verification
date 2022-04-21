# -*- coding: utf-8 -*-
# pylint: disable-msg=too-few-public-methods
# ========================================================
"""config module is written for write parameters."""
# ========================================================


# ========================================================
# Imports
# ========================================================

import argparse
from pathlib import Path


class BaseConfig:
    """
    BaseConfig class is written to write configs in it
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--model_name", type=str, default="Q_dec")

        self.parser.add_argument("--raw_data_dir", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/Raw")

        self.parser.add_argument("--processed_data_dir", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/Q_dec")

        self.parser.add_argument("--assets_dir", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets/")

        self.parser.add_argument("--saved_model_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets/saved_models/"),

        self.parser.add_argument("--language_model_path", type=str,
                                 default="/home/maryam.najafi/LanguageModels/mt5_en_large/",
                                 help="Path of the multilingual lm model dir")
        self.parser.add_argument("--language_model_tokenizer_path", type=str,
                                 default=Path(__file__).parents[2].__str__()
                                         + "/assets/pretrained_models/mt5_tokenizer")
        self.parser.add_argument("--roberta_model_path", type=str,
                                 default=Path(__file__).parents[2].__str__()
                                         + "/assets/pretrained_models/xlm_roberta_large")

        self.parser.add_argument("--bpemb_model_path", type=str,
                                 default=Path(__file__).parents[2].__str__()
                                         + "/assets/pretrained_models/bpemb/dutch/nl.wiki.bpe.vs200000.model")
        self.parser.add_argument("--bpemb_vocab_path", type=str,
                                 default=Path(__file__).parents[2].__str__()
                                         + "/assets/pretrained_models/bpemb/dutch/nl.wiki.bpe.vs200000.d300.w2v.bin")

        self.parser.add_argument("--csv_logger_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets")

        self.parser.add_argument("--pair_data", type=str, default="pairs.jsonl")
        self.parser.add_argument("--truth_data", type=str, default="truth.jsonl")

        self.parser.add_argument("--train_data", type=str, default="train.txt")
        self.parser.add_argument("--test_data", type=str, default="test.txt")
        self.parser.add_argument("--dev_data", type=str, default="val.txt")

        self.parser.add_argument("--data_headers", type=list, default=["text", "label"])
        self.parser.add_argument("--customized_headers", type=list, default=["text", "label"])

        self.parser.add_argument("--save_top_k", type=int, default=1, help="...")

        self.parser.add_argument("--num_workers", type=int,
                                 default=10,
                                 help="...")

        self.parser.add_argument("--n_epochs", type=int,
                                 default=100,
                                 help="...")

        self.parser.add_argument("--batch_size", type=int,
                                 default=32,
                                 help="...")

        self.parser.add_argument("--lr", default=2e-5,
                                 help="...")

        self.parser.add_argument("--lstm_units", type=int,
                                 default=128,
                                 help="...")
        self.parser.add_argument("--lstm_layers", type=int,
                                 default=2,
                                 help="...")
        self.parser.add_argument("--bidirectional", type=bool,
                                 default=True,
                                 help="...")
        self.parser.add_argument("--dropout", type=float,
                                 default=0.15,
                                 help="...")
        self.parser.add_argument("--embedding_dim", type=int,
                                 default=256,
                                 help="...")
        self.parser.add_argument("--alpha", type=float,
                                 default=50.0,
                                 help="...")
        self.parser.add_argument("--alpha_warmup_ratio", type=float,
                                 default=0.1,
                                 help="...")

    def get_config(self):
        """

        :return:
        """
        return self.parser.parse_args()
