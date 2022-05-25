import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer

from configuration import BaseConfig
from data_loader import read_json
from dataset import InferenceDataset
from models.t5_encoder_pos import Classifier
from utils import prepare_test_data, extract_punctuation_emoji, extract_information, \
    extract_pos, handle_pos_tags
from utils.helper import progress_bar

logging.basicConfig(level=logging.DEBUG)


def main(ARGS):
    np.set_printoptions(suppress=True)
    np.set_printoptions(formatter={'all': lambda x: str(x)})
    # ----------------------------------- instantiation -------------------------------

    # ARGS.best_model_path = "../assets/saved_models/version_115/checkpoints/QTag-epoch=55-val_acc=0.92.ckpt"
    # ARGS.input_data_path = "../data/Raw/pairs.jsonl"
    # TRUE_PATH = "../data/Raw/truth.jsonl"

    FIRST_AUTHORS_TEXTS, SECOND_AUTHORS_TEXTS, SAMPLE_ID = prepare_test_data(path=ARGS.input_data_path)
    FIRST_AUTHORS_TEXTS = FIRST_AUTHORS_TEXTS[:10]
    SECOND_AUTHORS_TEXTS = SECOND_AUTHORS_TEXTS[:10]
    SAMPLE_ID = SAMPLE_ID[:10]

    POS_VOCAB2IDX = read_json(path="../assets/token_vocab2idx.json")
    logging.info("Data was loaded")
    logging.info("We have %d data", len(FIRST_AUTHORS_TEXTS))

    T5_TOKENIZER = T5Tokenizer.from_pretrained(ARGS.language_model_tokenizer_path)
    logging.info("Tokenizer was loaded")
    MODEL = Classifier.load_from_checkpoint(os.path.join(ARGS.best_model_path, ARGS.saved_model_path),
                                            map_location="cuda:0").to("cuda:0")
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

    DATALOADER = DataLoader(DATASET, batch_size=1,
                            shuffle=False, num_workers=4)

    PREDICTIONS = []
    outf = open('av.jsonl', 'w')

    for i_batch, sample_batched in enumerate(DATALOADER):
        sample_batched["input_ids"] = sample_batched["input_ids"].to("cuda:0")
        sample_batched["punctuation"] = sample_batched["punctuation"].to("cuda:0")
        sample_batched["information"] = sample_batched["information"].to("cuda:0")
        sample_batched["pos"] = sample_batched["pos"].to("cuda:0")
        OUTPUT = MODEL(sample_batched)
        OUTPUT = torch.softmax(OUTPUT, dim=1)
        SIM_SCORES = OUTPUT.cpu().detach().numpy()
        SIM_SCORES = (SIM_SCORES[0])
        # print(SIM_SCORES)
        # SIM_SCORES = "{:f}".format(SIM_SCORES[1])
        SIM_SCORES = SIM_SCORES[1]
        print(SIM_SCORES)
        OUTPUT = np.argmax(OUTPUT.cpu().detach().numpy(), axis=1)
        progress_bar(i_batch, len(DATALOADER), "testing ....")
        PREDICTIONS.append(SIM_SCORES)
        print()
    radius = 0.01
    for u_id, pred in zip(SAMPLE_ID, PREDICTIONS):
        if pred >= 0.9999:
            r = {"id": u_id, "value": 0.9999}
        elif 0.5 - radius <= pred <= 0.5 + radius:
            r = {"id": u_id, "value": 0.5}
        else:
            pred = float("{:.4f}".format(pred))
            r = {"id": u_id, "value": pred}
        outf.write(json.dumps(r) + '\n')

    # report = classification_report(y_true=list(get_true_target(TRUE_PATH)[:10]), y_pred=PREDICTIONS,
    #                                target_names=["1", "0"])
    # print(report)
    # print(PREDICTIONS)
    # print(get_true_target(TRUE_PATH))
    # results = evaluate_all(get_true_target(TRUE_PATH), PREDICTIONS)
    # print(results)


if __name__ == '__main__':
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()
    ARGS.input_data_path = os.path.join(ARGS.raw_data_dir, ARGS.pair_data)
    ARGS.output_data_path = os.path.join(ARGS.saved_model_path)
    main(ARGS)
