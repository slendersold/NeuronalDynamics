import os
import sys

import traceback

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

import zarr
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import time

def create_save_dir(base_path: str, folder_name: str) -> str:
    """Создаёт директорию с именем, основанным на названии эксперимента,текущей дате и времени."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    save_dir = os.path.join(base_path, f"{folder_name} on {timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def fit_predict(zarrnik, channel, trial_names, model, labelencoder):
    # Используем только лейблы из трейна
    X, y = {}, {}
    for key, value in trial_names.items():
        X[key] = np.concatenate(
            [zarrnik[n]["embedding"][channel] for n in value], axis=0
        )
        y[key] = np.concatenate(
            [zarrnik[n]["labels"] for n in value], axis=0
        )

    # Обучение модели
    model.fit(X["train"], labelencoder.transform(y["train"][:, 1]))

    y_hat = model.predict(X["test"])

    return labelencoder.transform(y["test"][:, 1]), y_hat #true, predict

import zarr
import numpy as np
from numcodecs import Blosc, JSON
import json

def get_model_string(model_name, params):
    return f"{model_name}|{json.dumps(params, sort_keys=True)}"

def get_index(mapping, key):
    """Get index in mapping, add if not present."""
    if key not in mapping:
        mapping[key] = len(mapping)
    return mapping[key]

class ZarrLogWriter:
    def __init__(self, zarr_path, n_folds=5):
        self.zarr_path = zarr_path
        self.n_folds = n_folds
        self.store = zarr.open_group(zarr_path, mode='a')
        self._init_arrays()
        self.sub_chan_map = {}  # "sub-R1001P_ch0" -> row index
        self.model_map = {}     # "SVC|{...}" -> col index

    def _init_arrays(self):
        compressor = Blosc(cname="zstd", clevel=5)

        self.interfold = self.store.require_group("interfold")
        self.final = self.store.require_group("final")
        self.code = self.store.require_group("code")

        if "metrics" not in self.code:
            self.code.create_dataset("metrics", data=["accuracy", "precision", "recall", "f1", "time"],
                                    dtype=object, object_codec=JSON())

        self.interfold.create_dataset("metrics", shape=(0, 0, self.n_folds, 5), chunks=(1, 1, self.n_folds, 5),
                                    dtype="float32", compressor=compressor, overwrite=True)

        self.final.create_dataset("metrics", shape=(0, 0, 5), chunks=(1, 1, 5),
                                dtype="float32", compressor=compressor, overwrite=True)

        self.code.create_dataset("sub_chan", shape=(0,), chunks=(1,), dtype=object,
                                object_codec=JSON(), compressor=compressor, overwrite=True)

        self.code.create_dataset("model_plus_params", shape=(0,), chunks=(1,), dtype=object,
                                object_codec=JSON(), compressor=compressor, overwrite=True)

    def _resize_if_needed(self, row, col):
        interfold = self.interfold["metrics"]
        final = self.final["metrics"]

        if row >= interfold.shape[0]:
            interfold.resize((row + 1, interfold.shape[1], interfold.shape[2], interfold.shape[3]))
            final.resize((row + 1, final.shape[1], final.shape[2]))
            self.code["sub_chan"].resize((row + 1,))

        if col >= interfold.shape[1]:
            interfold.resize((interfold.shape[0], col + 1, interfold.shape[2], interfold.shape[3]))
            final.resize((final.shape[0], col + 1, final.shape[2]))
            self.code["model_plus_params"].resize((col + 1,))

    def log_fold(self, sub, channel, model_name, params, fold_idx, acc, prec, rec, f1, elapsed):
        key_row = f"{sub}_ch{channel}"
        key_col = get_model_string(model_name, params)

        row = get_index(self.sub_chan_map, key_row)
        col = get_index(self.model_map, key_col)

        self._resize_if_needed(row, col)

        self.interfold["metrics"][row, col, fold_idx, :] = [acc, prec, rec, f1, elapsed]
        self.code["sub_chan"][row] = key_row
        self.code["model_plus_params"][col] = key_col

    def log_final(self, sub, channel, model_name, params, acc, prec, rec, f1, mean_elapsed):
        key_row = f"{sub}_ch{channel}"
        key_col = get_model_string(model_name, params)

        row = get_index(self.sub_chan_map, key_row)
        col = get_index(self.model_map, key_col)

        self._resize_if_needed(row, col)

        self.final["metrics"][row, col, :] = [acc, prec, rec, f1, mean_elapsed]

    def flush(self):
        # В данном случае ничего не требуется — все изменения сохраняются итеративно.
        pass
if __name__ == "__main__":
    preprocessed_files = "/trinity/home/asma.benachour/processed_files/"
    pdf_output_dir = "/trinity/home/asma.benachour/PDF/"
    subs = [
        "sub-R1001P",
        "sub-R1002P",
        "sub-R1003P",
        "sub-R1010J",
        # "sub-R1015J", 4 trials - too small experiment itself
        "sub-R1020J",
        "sub-R1026D",
        "sub-R1031M",
        "sub-R1032D",
        "sub-R1035M",
    ]
    # Создание KFold с перемешиванием
    kf = KFold(n_splits=5, shuffle=True)

    channel = 0

    labelencoder = LabelEncoder()
    labelencoder.fit(['rec', 'not-rec'])
    # experimentation = input("Enter experimentation name without spaces:")
    experimentation = "memory_class"
    save_dir = create_save_dir(pdf_output_dir, experimentation)
    log_path = os.path.join(save_dir, f"log.txt")

    # Модели для классификации
    classification_models = {
        "SVC": {
            "model": SVC(probability=True),
            "param_grid": {
                "C": [1, 50, 100],
                "gamma": ["scale", "auto"],
                "kernel": ["rbf"]
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(class_weight="balanced", random_state=42),
            "param_grid": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [5, 10, 15, 20]
            }
        },
        "LogisticRegression": {
            "model": LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=5000),
            "param_grid": {
                "C": [1, 50, 100],
                "penalty": ["l1", "l2"]
            }
        }
    }
    
    labelencoder = LabelEncoder()
    labelencoder.fit(['rec', 'not-rec'])
    logger = ZarrLogWriter(f'{save_dir}/evaluation.zarr')

    for sub in subs:
        zarrnik = zarr.open(f"{preprocessed_files}/{sub}/WORD.zarr", mode="r")
        keys = list(zarrnik.group_keys())
        # for channel in range(zarrnik[keys[0]]["embedding"].shape[0]):
        for model_name, model_info in classification_models.items():
            model_proto = model_info["model"]
            param_grid = model_info.get("param_grid", [{}])  # [{}] if no grid specified
            for params in ParameterGrid(param_grid):
                model = clone(model_proto).set_params(**params)
                y_true_arr = []
                y_pred_arr = []
                start = time.time()
                for fold_idx, (train_ix, test_ix) in enumerate(kf.split(keys)):
                    try:
                        trial_names = {
                            "train": [keys[i] for i in train_ix],
                            "test": [keys[i] for i in test_ix],
                        }

                        y_true, y_pred = fit_predict(zarrnik, channel, trial_names, model, labelencoder)

                        end = time.time()
                        length = end - start

                        y_true_arr.append(y_true)
                        y_pred_arr.append(y_pred)

                        acc = accuracy_score(y_true, y_pred)
                        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
                        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
                        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

                        # with open(log_path, "a") as f:
                        #     f.write(
                        #         f"Patient: {sub}. Channel {channel}. Model name: {model_name}. Params: {params}. Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}. Time: {length:.2f}s\n"
                        #     )
                        logger.log_fold(sub, channel, model_name, params, fold_idx, acc, prec, rec, f1, length)

                    except Exception as e:
                        with open(log_path, "a") as f:
                            f.write(
                                f"==============================================\n"
                                f"Error during CV for {sub}, model={model_name}, params={params}\n"
                                f"train_ix={train_ix}\n"
                                f"test_ix={test_ix}\n"
                                f"Error: {e}\n"
                                f"{traceback.format_exc()}\n"
                                f"==============================================\n"
                            )

                # Оценка метрик
                if len(y_true_arr) == 0 or len(y_pred_arr) == 0:
                    acc = prec = rec = f1 = 0.0
                    with open(log_path, "a") as f:
                        f.write(
                            f"No predictions made for {sub} channel {channel} with {model_name} and params={params}. Returning zeros.\n"
                        )
                else:
                    y_true_all = np.concatenate(y_true_arr)
                    y_pred_all = np.concatenate(y_pred_arr)

                    acc = accuracy_score(y_true_all, y_pred_all)
                    prec = precision_score(y_true_all, y_pred_all, average="macro", zero_division=0)
                    rec = recall_score(y_true_all, y_pred_all, average="macro", zero_division=0)
                    f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)

                # with open(log_path, "a") as f:
                #     f.write(
                #         f"Final metrics for {sub}, channel {channel}, model={model_name}, params={params}:\n"
                #         f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}\n\n"
                #     )
                length = end - start
                logger.log_final(sub, channel, model_name, params, acc, prec, rec, f1, length)