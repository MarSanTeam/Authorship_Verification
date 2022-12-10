# -*- coding: utf-8 -*-
# ==========================================================================

"""
    Author Verification Project:
        data_loader:
            data_writer.py
"""

# ============================ Third Party libs ============================
import json
import pickle

import pandas as pd


# ==========================================================================


def write_json(data: list or dict,
               path: str) -> None:
    """
    write_json function is written for write in json files

    Args:
        data: data to save in json file
        path: path of JSON file

    Returns:
        None

    """
    with open(path, "w", encoding="utf8") as outfile:
        json.dump(data, outfile, separators=(",", ":"), indent=4, ensure_ascii=False)


def write_pickle(data: list or dict,
                 path: str) -> None:
    """
    write_pickle function is written for write data in pickle file

    Args:
        data: data to save in PICKLE file
        path: path of PICKLE file

    Returns:
        None

    """
    with open(path, "wb") as outfile:
        pickle.dump(data, outfile)


def write_text(data: list,
               path: str) -> None:
    """
    save_text function is written for write in text files

    Args:
        data: data to save in TEXT file
        path: path of TEXT file

    Returns:
        None

    """
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(data))


def write_excel(dataframe: pd.DataFrame,
                path: str, ) -> None:
    """
    write_excel function is written for write in excel files
    Args:
        dataframe: data to save in EXCEL file
        path: path of EXCEL file

    Returns:
        None

    """
    dataframe.to_excel(path, index=False)


def write_csv(dataframe: pd.DataFrame, path: str) -> None:
    """
    write_csv function is written for write in csv files

        dataframe: data to save in CSV file
        path: path of CSV file

    Returns:
        None

    """
    dataframe.to_csv(path, index=False)
