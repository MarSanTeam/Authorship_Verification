import logging
from torch.utils.data import DataLoader
import numpy as np

from configuration import BaseConfig
from data_loader import read_json
from transformers import T5Tokenizer
from models.t5_encoder_pos import Classifier
from utils import prepare_test_data, extract_punctuation_emoji, extract_information, \
    extract_pos, handle_pos_tags
from dataset import InferenceDataset

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # ----------------------------------- instantiation -------------------------------
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    MODEL_PATH = "../assets/saved_models/version_115/" \
                 "checkpoints/QTag-epoch=55-val_acc=0.92.ckpt"
    DATA_PATH = "../data/Raw/pairs.jsonl"

    FIRST_AUTHORS_TEXTS, SECOND_AUTHORS_TEXTS = prepare_test_data(path=DATA_PATH)
    FIRST_AUTHORS_TEXTS = FIRST_AUTHORS_TEXTS[:10]
    SECOND_AUTHORS_TEXTS = SECOND_AUTHORS_TEXTS[:10]

    POS_VOCAB2IDX = read_json(path="../assets/token_vocab2idx.json")
    logging.info("Data was loaded")
    logging.info("We have %d data", len(FIRST_AUTHORS_TEXTS))

    T5_TOKENIZER = T5Tokenizer.from_pretrained(ARGS.language_model_tokenizer_path)
    logging.info("Tokenizer was loaded")
    MODEL = Classifier.load_from_checkpoint(MODEL_PATH, map_location="cuda:0").to("cuda:0")
    logging.info("Model was loaded")

    FIRST_TEXT_PUNCTUATIONS = extract_punctuation_emoji(FIRST_AUTHORS_TEXTS)
    SECOND_TEXT_PUNCTUATIONS = extract_punctuation_emoji(SECOND_AUTHORS_TEXTS)
    logging.info("Punctuations and emojis are extracted")

    FIRST_TEXT_INFORMATION = extract_information(FIRST_AUTHORS_TEXTS)
    SECOND_TEXT_INFORMATION = extract_information(SECOND_AUTHORS_TEXTS)
    logging.info("Information are extracted")

    FIRST_TEXT_POS = extract_pos(FIRST_AUTHORS_TEXTS)
    SECOND_TEXT_POS = extract_pos(SECOND_AUTHORS_TEXTS)
    FIRST_TEXT_POS = handle_pos_tags(FIRST_TEXT_POS, POS_VOCAB2IDX)
    SECOND_TEXT_POS = handle_pos_tags(SECOND_TEXT_POS, POS_VOCAB2IDX)
    logging.info("Pos are extracted")

    DATA = {"first_text": FIRST_AUTHORS_TEXTS,
            "second_text": SECOND_AUTHORS_TEXTS,
            "first_punctuations": FIRST_TEXT_PUNCTUATIONS,
            "second_punctuations": SECOND_TEXT_PUNCTUATIONS,
            "first_information": FIRST_TEXT_INFORMATION,
            "second_information": SECOND_TEXT_INFORMATION,
            "first_pos": FIRST_TEXT_POS,
            "second_pos": SECOND_TEXT_POS}

    DATASET = InferenceDataset(data=DATA, tokenizer=T5_TOKENIZER,
                               max_len=ARGS.max_len)

    DATALOADER = DataLoader(DATASET, batch_size=4,
                            shuffle=False, num_workers=4)

    PREDICTIONS = []
    for i_batch, sample_batched in enumerate(DATALOADER):
        sample_batched["input_ids"] = sample_batched["input_ids"].to("cuda:0")
        sample_batched["punctuation"] = sample_batched["punctuation"].to("cuda:0")
        sample_batched["information"] = sample_batched["information"].to("cuda:0")
        sample_batched["pos"] = sample_batched["pos"].to("cuda:0")
        OUTPUT = MODEL(sample_batched)
        OUTPUT = np.argmax(OUTPUT.cpu().detach().numpy(), axis=1)

        PREDICTIONS.extend(OUTPUT)
    print(PREDICTIONS)
