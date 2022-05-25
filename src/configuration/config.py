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
        self.parser.add_argument("--model_name", type=str, default="Author_Verification")

        self.parser.add_argument("--raw_data_dir", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/Raw")

        self.parser.add_argument("--processed_data_dir", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/Processed")

        self.parser.add_argument("--assets_dir", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets/")

        self.parser.add_argument("--saved_model_path", type=str,
           default=Path(__file__).parents[2].__str__() + "/assets/saved_models/"),

        self.parser.add_argument("--language_model_path", type=str,
                                 default=Path(__file__).parents[3].__str__()
                                         + "/LanguageModels/t5_en_large/",
                                 help="Path of the multilingual lm model dir")
        self.parser.add_argument("--language_model_tokenizer_path", type=str,
                                 default=Path(__file__).parents[3].__str__()
                                         + "/LanguageModels/t5_en_large/")
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
        self.parser.add_argument("--best_model_path", type=str,
                                 default="Author_Verification/version_115/checkpoints/QTag-epoch=59-val_acc=0.92.ckpt")
        self.parser.add_argument("--pair_data", type=str, default="pairs.jsonl")
        self.parser.add_argument("--truth_data", type=str, default="truth.jsonl")

        self.parser.add_argument("--train_file", type=str, default="train_data.csv")
        self.parser.add_argument("--test_file", type=str, default="test_data.csv")
        self.parser.add_argument("--dev_file", type=str, default="val_data.csv")

        self.parser.add_argument("--data_headers",
                                 type=list,
                                 default=["first_text", "second_text", "lable"])
        self.parser.add_argument("--customized_headers",
                                 type=list,
                                 default=["first_text", "second_text", "targets"])

        self.parser.add_argument("--save_top_k", type=int, default=1, help="...")

        self.parser.add_argument("--num_workers", type=int,
                                 default=10,
                                 help="...")

        self.parser.add_argument("--n_epochs", type=int,
                                 default=100,
                                 help="...")

        self.parser.add_argument("--batch_size", type=int,
                                 default=4,
                                 help="...")
        self.parser.add_argument("--max_len", type=int, default=350,
                                 help="Maximum length of inputs")

        self.parser.add_argument("--lr", default=2e-5,
                                 help="...")

        self.parser.add_argument("--n_filters", type=int,
                                 default=128,
                                 help="...")
        self.parser.add_argument("--filter_sizes", type=int,
                                 default=[1, 2, 3],
                                 help="...")
        self.parser.add_argument("--dropout", type=float,
                                 default=0.15,
                                 help="...")
        self.parser.add_argument("--embedding_dim", type=int,
                                 default=50,
                                 help="...")

    def get_config(self):
        """

        :return:
        """
        return self.parser.parse_args()
