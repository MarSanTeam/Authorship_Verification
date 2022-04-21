
# -*- coding: utf-8 -*-
# pylint: disable-msg=import-error
# ========================================================
"""
    FAQ Project:
        data loader:
            loading data
"""

# ============================ Third Party libs ============================
import json
import pickle
import json5
import pandas as pd
# ==========================================================================


def read_csv(path: str, columns: list = None, names: list = None) -> pd.DataFrame:
    """
    read_csv function for reading csv files
    :param path:
    :param columns:
    :param names:
    :return:
    """
    dataframe = pd.read_csv(path, usecols=columns) if columns else pd.read_csv(path)
    return dataframe.rename(columns=dict(zip(columns, names))) if names else dataframe


def read_excel(path: str, columns: list = None, names: list = None) -> pd.DataFrame:
    """
    read_excel function for reading excel files
    :param path:
    :param columns:
    :param names:
    :return:
    """

    dataframe = pd.read_excel(path, usecols=columns) if columns else pd.read_excel(path)
    return dataframe.rename(columns=dict(zip(columns, names))) if names else dataframe


def read_json(path: str) -> json:
    """
    read_json function for  reading json file
    :param path:
    :return:
    """
    with open(path, encoding="utf-8") as json_file:
        return json.load(json_file)


def read_json5(path: str):
    """
    read_json5 function for  reading json file
    :param path:
    :return:
    """
    with open(path, "r", encoding="utf-8") as file:
        data = json5.load(file)
    return data


def read_text(path: str) -> list:
    """
    read_text function for  reading text file
    :param path:
    :return:
    """
    with open(path, "r", encoding="utf8") as file:
        data = file.readlines()
    return data


def read_pickle(path: str) -> list:
    """
    read_pickle function for  reading pickle file
    :param path:
    :return:
    """
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data
