import os
from configuration import BaseConfig
from transformers import T5Tokenizer
from models.t5_encoder_pos import Classifier
from utils import prepare_test_data

if __name__ == "__main__":
    # ----------------------------------- instantiation -------------------------------
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    MODEL_PATH = ""

    first_authors_texts, second_authors_texts = prepare_test_data(path="")

    T5_TOKENIZER = T5Tokenizer.from_pretrained(ARGS.language_model_tokenizer_path)
    MODEL = Classifier.load_from_checkpoint(MODEL_PATH)


