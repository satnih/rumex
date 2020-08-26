# This code is based on code from Stanford cs230 class
# https://github.com/cs230-stanford/cs230-code-examples
import json
import logging
import os
import shutil
import torch
import pandas as pd


def save_ckpt(state_dict, pred_dict, metrics_dict, is_best, ckpt_dir):
    """Saves model and training parameters at ckpt_dir + 'last.pth.tar'. If is_best==True, also saves
    ckpt_dir + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, 
        is_best: (bool) True if it is the best model seen till now
        ckpt_dir: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(ckpt_dir, 'last.pth.tar')
    if not os.path.exists(ckpt_dir):
        print("Checkpoint Directory does not exist! Making directory {}".format(ckpt_dir))
        os.mkdir(ckpt_dir)
    torch.save(state_dict, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(ckpt_dir, 'best.pth.tar'))
        save_to_json(pred_dict, os.path.join(ckpt_dir, 'best_pred.json'))
        save_to_json(metrics_dict, os.path.join(ckpt_dir, 'best_metrics.json'))


def load_ckpt(ckpt_dir, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is 
    provided, loads state_dict of optimizer assuming it is present in ckpt_dir.

    Args:
        ckpt_dir: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from ckpt_dir
    """
    if not os.path.exists(ckpt_dir):
        raise("File doesn't exist {}".format(ckpt_dir))
    ckpt_dir = torch.load(ckpt_dir)
    model.load_state_dict(ckpt_dir['state_dict'])

    if optimizer:
        optimizer.load_state_dict(ckpt_dir['optim_dict'])

    return ckpt_dir


def save_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
