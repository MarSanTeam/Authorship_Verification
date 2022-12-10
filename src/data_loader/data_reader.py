# -*- coding: utf-8 -*-
# ========================================================

"""
    Author Verification Project:
        data_loader:
                data_reader.py
"""

# ============================ Third Party libs ============================
import json
from typing import List
import pickle
import json5
import pandas as pd


# ==========================================================================


def read_csv(path: str,
             columns: List[str] = None,
             names: List[str] = None) -> pd.DataFrame:
    """
    read_csv function for reading csv files

    Args:
        path: path of CSV file
        columns: list of columns name
        names: list of new columns name

    Returns:
        loaded dataFrame

    """
    dataframe = pd.read_csv(path, usecols=columns) if columns else pd.read_csv(path)
    return dataframe.rename(columns=dict(zip(columns, names))) if names else dataframe


def read_excel(path: str,
               columns: List[str] = None,
               names: List[str] = None) -> pd.DataFrame:
    """
    read_excel function for reading excel files

    Args:
        path: path of EXCEL file
        columns: list of columns name
        names: list of new columns name

    Returns:
        loaded dataFrame

    """
    dataframe = pd.read_excel(path, usecols=columns) if columns else pd.read_excel(path)
    return dataframe.rename(columns=dict(zip(columns, names))) if names else dataframe


def read_json(path: str) -> json:
    """
    read_json function for  reading json file

    Args:
        path: path of JSON file

    Returns:
        loaded json file

    """
    with open(path, encoding="utf-8") as json_file:
        return json.load(json_file)


def read_json5(path: str) -> json5:
    """
    read_json5 function for  reading json file

    Args:
        path: path of JSON file

    Returns:
        loaded json file

    """
    with open(path, "r", encoding="utf-8") as file:
        data = json5.load(file)
    return data


def read_text(path: str) -> list:
    """
    read_text function for  reading text file

    Args:
        path: path of TEXT file


    Returns:
        loaded text file

    """
    with open(path, "r", encoding="utf8") as file:
        data = file.readlines()
    return data


def read_pickle(path: str) -> list:
    """
    read_pickle function for  reading pickle file

    Args:
        path: path of PICKLE file


    Returns:
        loaded pickle file

    """
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data
