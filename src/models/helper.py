# -*- coding: utf-8 -*-
# ========================================================

"""
    Author Verification Project:
        models:
            helper.py
"""

# ============================ Third Party libs ============================

from pytorch_lightning.callbacks import ModelCheckpoint


def build_checkpoint_callback(save_top_k,
                              filename="QTag-{epoch:02d}-{val_loss:.2f}",
                              monitor="val_loss",
                              mode="min"):
    """
    :param save_top_k: save top k model that has low loss
    :param filename: the name that checkpoint is saved.
    :param monitor: how to monitor val loss
    :param mode:
    """
    # saves a file like: input/QTag-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,  # monitored quantity
        filename=filename,
        save_top_k=save_top_k,  # save the top k models
        mode=mode,  # mode of the monitored quantity for optimization
    )
    return checkpoint_callback
