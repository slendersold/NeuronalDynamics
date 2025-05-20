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

def permutation_test_cv_f1(y_true_list, y_pred_list,
                           n_permutations=1000,
                           average="macro",
                           random_state=None):
    rng = np.random.RandomState(random_state)

    # 1) наблюдаемое значение
    f1_obs = np.mean([
        f1_score(y_true, y_pred, average=average)
        for y_true, y_pred in zip(y_true_list, y_pred_list)
    ])

    # 2) подготовка под пермутации
    f1_perm = np.zeros(n_permutations, dtype=float)

    for i in range(n_permutations):
        f1s = []
        for y_true, y_pred in zip(y_true_list, y_pred_list):
            y_pred_shuffled = rng.permutation(y_pred)
            f1s.append(f1_score(y_true, y_pred_shuffled, average=average))
        f1_perm[i] = np.mean(f1s)

    # 3) p-value
    p_value = (np.sum(f1_perm >= f1_obs) + 1) / (n_permutations + 1)

    return f1_obs, p_value, f1_perm


def create_save_dir(base_path: str, folder_name: str) -> str:
    """Создаёт директорию с именем, основанным на названии эксперимента,текущей дате и времени."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(base_path, f"{timestamp} as {folder_name}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def fit_predict_classic(zarrnik, channel, trial_names, model, labelencoder, preprocessing_type,**kwargs):
    # Используем только лейблы из трейна
    X, y = {}, {}
    for split, names in trial_names.items():
        # конкатенация вдоль временной оси
        X[split] = np.concatenate([
            np.nan_to_num(
                zarrnik[trial]["embedding"][channel], 
                nan=0, posinf=5, neginf=0
            )
            for trial in names
        ], axis=0)
        # читаем лейблы из VLenUnicode
        labels_concat = np.concatenate([
            zarrnik[trial]["labels"][:]   # dtype=object, строки или пара строк
            for trial in names
        ], axis=0)
        if preprocessing_type == "WORD":
            # labels_concat.shape = (N,2): [word, verdict]
            y[split] = labelencoder.transform(labels_concat[:,1])
        else:
            # labels_concat.shape = (N,)
            y[split] = labelencoder.transform(labels_concat)

    # Обучение / предсказание как раньше
    model.fit(X["train"], y["train"])
    y_pred = model.predict(X["test"])
    return y["test"], y_pred

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# === 1. Определение модели ===
class FeedforwardClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(FeedforwardClassifier, self).__init__()
        self.hidden_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_hidden=False):
        hidden = self.hidden_layer(x)
        out = self.output_layer(hidden)
        return (out, hidden) if return_hidden else out

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def fit_predict_neural(
    zarrnik,
    channel,
    trial_names,
    model,
    labelencoder,
    preprocessing_type,
    **kwargs
):
    """
    Обучает PyTorch модель на основе эмбеддингов из Zarr и возвращает предсказания и истинные метки.

    Аргументы:
    ----------
    zarrnik : zarr.Group
        Zarr-группа, содержащая эмбеддинги и метки для одного пациента/эксперимента.
    channel : int
        Индекс канала, из которого брать эмбеддинги.
    trial_names : dict
        Словарь с ключами 'train' и 'test', значениями являются списки ключей Zarr-группы (названия испытаний).
    model_proto : nn.Module
        Инициализированная модель PyTorch (например, от model_factory(...)).
    labelencoder : sklearn.preprocessing.LabelEncoder
        Кодировщик меток.
    preprocessing_type : str
        Строка, определяющая тип задачи ('WORD' — бинарная, иначе — мульткласс).
    kwargs : dict
        Параметры обучения:
            - batch_size_train: int (по умолчанию 64)
            - batch_size_test: int (по умолчанию 128)
            - lr: float (по умолчанию 1e-3)
            - epochs: int (по умолчанию 10)
            - device: str (по умолчанию 'cuda' if available else 'cpu')

    Возвращает:
    -----------
    y_true : np.ndarray
        Истинные метки тестовой выборки.
    y_pred : np.ndarray
        Предсказанные моделью метки.
    """
    # === 1. Параметры из kwargs с дефолтами ===
    batch_size_train = kwargs.get('batch_size_train', 64)
    batch_size_test = kwargs.get('batch_size_test', 128)
    lr = kwargs.get('lr', 1e-3)
    epochs = kwargs.get('epochs', 10)
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # === 2. Загрузка и подготовка данных ===
    X, y = {}, {}
    for key, value in trial_names.items():
        X[key] = np.concatenate([
            np.nan_to_num(zarrnik[n]["embedding"][channel], nan=0, posinf=5, neginf=0)
            for n in value
        ], axis=0)
        y[key] = np.concatenate([zarrnik[n]["labels"] for n in value], axis=0)

    # Выбор правильного столбца лейблов
    if preprocessing_type == "WORD":
        y_train = labelencoder.transform(y["train"][:, 1])
        y_test = labelencoder.transform(y["test"][:, 1])
    else:
        y_train = labelencoder.transform(y["train"])
        y_test = labelencoder.transform(y["test"])

    # === 3. Torch DataLoader ===
    X_train_tensor = torch.tensor(X["train"], dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                              batch_size=batch_size_train, shuffle=True)

    X_test_tensor = torch.tensor(X["test"], dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor),
                             batch_size=batch_size_test)

    # === 4. Обучение модели ===
    model = model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    # === 5. Предсказания ===
    model.eval()
    y_pred = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            pred_labels = torch.argmax(logits, dim=1)
            y_pred.extend(pred_labels.cpu().numpy())

    return y_test, np.array(y_pred)

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
        # Метрики; список имён можно менять
        self.metric_names = ["pval", "precision", "recall", "f1", "time"]
        self._init_arrays()
        self.sub_chan_map = {}  # "sub-R1001P_ch0" -> row index
        self.model_map    = {}  # "SVC|{...}"         -> col index
        
    def _init_arrays(self):
        compressor = Blosc(cname="zstd", clevel=5)
        self.interfold = self.store.require_group("interfold")
        self.final     = self.store.require_group("final")
        self.code      = self.store.require_group("code")

        if "metrics" not in self.code:
            self.code.create_dataset(
                "metrics",
                data=self.metric_names,
                dtype=object,
                object_codec=JSON()
            )

        M = len(self.metric_names)
        # интерфолд: (rows, cols, n_folds, M)
        self.interfold.create_dataset(
            "metrics",
            shape=(0, 0, self.n_folds, M),
            chunks=(1, 1, self.n_folds, M),
            dtype="float32",
            compressor=compressor,
            overwrite=True
        )
        # финал: (rows, cols, M)
        self.final.create_dataset(
            "metrics",
            shape=(0, 0, M),
            chunks=(1, 1, M),
            dtype="float32",
            compressor=compressor,
            overwrite=True
        )
        # кодовые массивы
        self.code.create_dataset(
            "sub_chan", shape=(0,), chunks=(1,),
            dtype=object, object_codec=JSON(), compressor=compressor,
            overwrite=True
        )
        self.code.create_dataset(
            "model_plus_params", shape=(0,), chunks=(1,),
            dtype=object, object_codec=JSON(), compressor=compressor,
            overwrite=True
        )

    def _resize_if_needed(self, row, col):
        inter = self.interfold["metrics"]
        fin   = self.final["metrics"]

        # расширяем строки
        if row >= inter.shape[0]:
            inter.resize((row + 1, inter.shape[1], inter.shape[2], inter.shape[3]))
            fin.resize((row + 1, fin.shape[1], fin.shape[2]))
            self.code["sub_chan"].resize((row + 1,))

        # расширяем столбцы
        if col >= inter.shape[1]:
            inter.resize((inter.shape[0], col + 1, inter.shape[2], inter.shape[3]))
            fin.resize((fin.shape[0], col + 1, fin.shape[2]))
            self.code["model_plus_params"].resize((col + 1,))

    def _log(self, sub, channel, model_name, params, values, fold_idx=None):
        key_row = f"{sub}_ch{channel}"
        key_col = get_model_string(model_name, params)

        row = get_index(self.sub_chan_map, key_row)
        col = get_index(self.model_map, key_col)

        self._resize_if_needed(row, col)

        # записываем кодовые массивы
        self.code["sub_chan"][row]          = key_row
        self.code["model_plus_params"][col] = key_col

        if fold_idx is None:
            self.final["metrics"][row, col, :] = values
        else:
            self.interfold["metrics"][row, col, fold_idx, :] = values

    def log_fold(self, *, sub, channel, model_name, params, fold_idx, metrics):
        """
        sub, channel, model_name, params, fold_idx — передаются как ключевые аргументы,
        metrics — список/кортеж из M чисел, где M = число метрик в вашей инициализации.
        """
        self._log(sub, channel, model_name, params,
                  values=metrics, fold_idx=fold_idx)

    def log_final(self, *, sub, channel, model_name, params, metrics):
        """
        То же самое для итоговых метрик, без fold_idx.
        """
        self._log(sub, channel, model_name, params,
                  values=metrics, fold_idx=None)

if __name__ == "__main__":
    preprocessed_files = "/trinity/home/asma.benachour/processed_files_ver2/"
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
    # experimentation = input("Enter experimentation name without spaces:")
    experimentation = "fullscale_cross"
    save_dir = create_save_dir(pdf_output_dir, experimentation)
    log_path = os.path.join(save_dir, f"log.txt")
    for preprocessing_type in ["partial_intervals","WORD"]:
    # for preprocessing_type in ["WORD"]:
        
        # Создание KFold с перемешиванием
        kf = KFold(n_splits=5, shuffle=True)

        channel = 0

        labelencoder = LabelEncoder()
        if preprocessing_type == "WORD":
            labelencoder.fit(['rec', 'not-rec'])
        elif preprocessing_type == "partial_intervals":
            labelencoder.fit(["COUNTDOWN", "MATH", "OTHER", "REC_WORD", "WORD"])
        num_classes = len(labelencoder.classes_)

        # Модели для классификации
        classification_models = {
            "FeedForward": {
                "model": lambda input_dim=768, hidden_dim=256, output_dim=2, dropout=0.3:
                    FeedforwardClassifier(input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        output_dim=output_dim,
                                        dropout=dropout),
                "param_grid": {
                    "hidden_dim": [128, 256],
                    "dropout": [0.2, 0.3]
                }
            },
            "SVC": {
                "model": SVC(probability=True),
                "param_grid": {
                    "C": [1, 50],
                    "gamma": ["scale", "auto"],
                    "kernel": ["rbf"]
                    # "C": [1],
                    # "gamma": ["scale", "auto"],
                    # "kernel": ["rbf"]
                }
            },
            "RandomForest": {
                "model": RandomForestClassifier(class_weight="balanced", random_state=42),
                "param_grid": {
                    "n_estimators": [50, 100, 200, 300],
                    "max_depth": [5, 10, 15, 20]
                    # "n_estimators": [150, 200, 250],
                    # "max_depth": [5]
                }
            },
            "LogisticRegression": {
                "model": LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=5000),
                "param_grid": {
                    "C": [1, 50, 100],
                    "penalty": ["l1", "l2"]
                    # "C": [1],
                    # "penalty": ["l1"]
                }
            }
        }
        logger = ZarrLogWriter(f'{save_dir}/evaluation_{preprocessing_type}.zarr')

        for sub in subs:
            zarrnik = zarr.open(f"{preprocessed_files}/{sub}/{preprocessing_type}.zarr", mode="r")
            keys = [k for k in zarrnik.group_keys() if k.startswith("trial_")]

            for model_name, model_info in classification_models.items():
                model_proto = model_info["model"]
                param_grid = model_info.get("param_grid", [{}])  # [{}] if no grid specified
                for params in ParameterGrid(param_grid):
                    if model_name == "FeedForward":
                        model = model_proto(input_dim=768,
                                     hidden_dim=params['hidden_dim'],
                                     output_dim=num_classes,
                                     dropout=params['dropout'])
                        fit_predict = fit_predict_neural
                    else:
                        model = clone(model_proto).set_params(**params)
                        fit_predict = fit_predict_classic
                    
                    y_true_arr = []
                    y_pred_arr = []
                    start = time.time()
                    for fold_idx, (train_ix, test_ix) in enumerate(kf.split(keys)):
                        try:
                            trial_names = {
                                "train": [keys[i] for i in train_ix],
                                "test": [keys[i] for i in test_ix],
                            }

                            y_true, y_pred = fit_predict(
                                zarrnik=zarrnik,
                                channel=channel,
                                trial_names=trial_names,
                                model=model,
                                labelencoder=labelencoder,
                                preprocessing_type=preprocessing_type,
                                lr=1e-4,
                                epochs=15,
                                batch_size_train=32,
                                batch_size_test=64
                            )

                            end = time.time()
                            length = end - start

                            y_true_arr.append(y_true)
                            y_pred_arr.append(y_pred)

                            # acc = accuracy_score(y_true, y_pred)

                            prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
                            rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
                            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

                            # with open(log_path, "a") as f:
                            #     f.write(
                            #         f"Patient: {sub}. Channel {channel}. Model name: {model_name}. Params: {params}. Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}. Time: {length:.2f}s\n"
                            #     )
                            logger.log_fold(
                                sub=sub,
                                channel=channel,
                                model_name=model_name,
                                params=params,
                                fold_idx=fold_idx,
                                metrics=[0.0, prec, rec, f1, length]
                            )

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
                        p_val = prec = rec = f1 = 0.0
                        with open(log_path, "a") as f:
                            f.write(
                                f"No predictions made for {sub} channel {channel} with {model_name} and params={params}. Returning zeros.\n"
                            )
                    else:
                        y_true_all = np.concatenate(y_true_arr)
                        y_pred_all = np.concatenate(y_pred_arr)

                        # acc = accuracy_score(y_true_all, y_pred_all)
                        f1, p_val, f1_null = permutation_test_cv_f1(
                            y_true_arr,
                            y_pred_arr,
                            n_permutations=10000,
                            average="macro"
                        )
                        prec = precision_score(y_true_all, y_pred_all, average="macro", zero_division=0)
                        rec = recall_score(y_true_all, y_pred_all, average="macro", zero_division=0)
                        # f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)

                    # with open(log_path, "a") as f:
                    #     f.write(
                    #         f"Final metrics for {sub}, channel {channel}, model={model_name}, params={params}:\n"
                    #         f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}\n\n"
                    #     )
                    end = time.time()
                    length = end - start
                    logger.log_final(
                                sub=sub,
                                channel=channel,
                                model_name=model_name,
                                params=params,
                                metrics=[p_val, prec, rec, f1, length]
                            )