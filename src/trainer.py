# -*- coding: utf-8 -*-
# ========================================================
"""trainer module is written for train model"""
# ========================================================


# ========================================================
# Imports
# ========================================================

import os
import logging
import json

from configuration import BaseConfig
from data_prepration import prepare_av_data

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # create config instance
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    FIRST_AUTHORS_TEXTS, SECOND_AUTHORS_TEXTS, TARGETS = prepare_av_data(
        pair_data_path=os.path.join(CONFIG.raw_data_dir, CONFIG.pair_data),
        truth_data_path=os.path.join(CONFIG.raw_data_dir, CONFIG.truth_data))

    assert len(FIRST_AUTHORS_TEXTS) == len(SECOND_AUTHORS_TEXTS) == len(TARGETS)
