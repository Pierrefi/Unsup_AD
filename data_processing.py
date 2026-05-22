#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function

import json
import torch 
import logging 
import os 
import pickle 

import pandas as pd 
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np 

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

random.seed(42)
batch_size = 32


def set_binary_label(dataframe, indomain, col='label'):
    if indomain:
        dataframe[col].values[:] = 0
    else:
        dataframe[col].values[:] = 1

def load_dataset_clinc(data_name, data_type='full'):
    with open('./dataset/CLINIC150/data_full.json', 'r') as f:
        data = json.load(f)
    field = "_".join(data_name.split("_")[1:])
    dataset = data[field]
    data_df = pd.DataFrame(dataset, columns=['text', 'label'])  # labels are not used for training
    return data_df

def load_extra_dataset(file_path ,drop_index=False, label=0):
    df = pd.read_csv(file_path, sep='\t', header=0)
    df['label'] = label
    df.rename(columns = {'sentence': 'text'}, inplace=True)
    if drop_index:
        df.drop(columns='index', inplace=True)
    df.dropna(inplace=True)
    return df

def save_features(id_train, id_test, ood_test, model_name, dataset_name, data = None):

    if data == 'agnews' :
        base_dir = f'/initial_framework/embeddings_files/{model_name}/agnews/{dataset_name}/'       
    else :  
        base_dir = f'/initial_framework/embeddings_files/{model_name}/{dataset_name}/' 


    os.makedirs(os.path.dirname(base_dir), exist_ok=True)

    features_map = {
        "train_features": id_train,
        "test_features": id_test,
        "ood_features": ood_test
    }

    for feature_name, features in features_map.items():
        file_name = os.path.join(base_dir, f"{feature_name}.pkl")
        with open(file_name, 'wb') as f:
            pickle.dump(features, f)

def load_features(dataset_name, model_name):

    base_dir = f'/home/ids/fihey-23/new_code_contamination/embeddings_files/{model_name}/{dataset_name}/outlier_detection/' 

    features_map = {}
    
    for feature_name in ["train_features", "test_features", "ood_features"]:
        file_name = os.path.join(base_dir, f"{feature_name}.pkl")
        
        if os.path.exists(file_name):  # Vérifie que le fichier existe
            with open(file_name, 'rb') as f:
                features_map[feature_name] = pickle.load(f)
                print(f"{feature_name} chargé depuis {file_name}.")
        else:
            raise FileNotFoundError(f"Fichier introuvable : {file_name}")
    
    return features_map


def create_datasets_df(dataset, id_labels, ood_labels, test_size=0.3, random_state=42, sample = False, with_contamination=0.0):
    train_df_base = pd.DataFrame(dataset['train'])
    split_name = "test" if "test" in dataset else "validation"

    test_df = pd.DataFrame(dataset[split_name])

    if sample == True : 
        train_df_base = train_df_base.groupby('label').apply(lambda x: x.sample(n=15000, random_state=42)).reset_index(drop=True)

    for df in [train_df_base, test_df]:
        df.drop(columns=[col for col in df.columns if col not in ['text', 'label']], inplace=True)
    
    train_df_id = train_df_base[train_df_base['label'].isin(id_labels)]
    test_df_id = test_df[test_df['label'].isin(id_labels)]
    
    for df in [train_df_id, test_df_id]:
        set_binary_label(df, indomain=True)
    
    # Créer les datasets OOD
    ood_df_train = train_df_base[train_df_base['label'].isin(ood_labels)]
    ood_df = test_df[test_df['label'].isin(ood_labels)]
    set_binary_label(ood_df, indomain=False)

    if with_contamination > 0:

        n_id = len(train_df_id)
        n_ood = int((with_contamination / (1 - with_contamination)) * n_id)  
         
        ood_train_contamination = ood_df_train.sample(n=n_ood, random_state=random_state, replace=False)  

        train_df_id = pd.concat([train_df_id, ood_train_contamination], ignore_index=True)

    test_df = pd.concat([test_df_id, ood_df], ignore_index=True)
    
    return train_df_id, test_df, ood_df, test_df_id

def _build_nyt_text(row: pd.Series) -> str:
    """
    Construit un champ 'text' à partir des champs NYT disponibles.
    Tu peux ajuster si tu veux inclure/exclure certains champs.
    """
    parts = []
    for k in ["headline", "abstract", "caption"]:
        v = row.get(k, None)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return "\n\n".join(parts).strip()

def create_datasets_nytimes(
    id_label: str,
    label_col: str = "section",
    test_size: float = 0.3,
    random_state: int = 42,
    with_contamination: float = 0.0,
):

    dataset_json_path="/initial_framework/dataset/nytimes_dataset.json"
    with open(dataset_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)
    df["label"] = df[label_col].astype(str)
    df["text"] = df.apply(_build_nyt_text, axis=1)

    # 3) Split train/test stratifié si possible
    stratify = df["label"] if df["label"].nunique() > 1 else None
    train_df_base, test_df_base = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # 4) ID vs OOD
    id_labels = [id_label]
    all_labels = set(df["label"].unique().tolist())
    ood_labels = sorted(list(all_labels - set(id_labels)))

    train_df_id = train_df_base[train_df_base["label"].isin(id_labels)].copy()
    test_df_id  = test_df_base[test_df_base["label"].isin(id_labels)].copy()

    # binaire ID/OOD (en supposant que tu as déjà cette fonction)
    for d in (train_df_id, test_df_id):
        set_binary_label(d, indomain=True)

    # OOD train pool + OOD test
    ood_df_train_pool = train_df_base[train_df_base["label"].isin(ood_labels)].copy()
    ood_df = test_df_base[test_df_base["label"].isin(ood_labels)].copy()
    set_binary_label(ood_df, indomain=False)

    # 5) Contamination éventuelle : on injecte de l'OOD dans le train ID
    if with_contamination > 0:
        if not (0.0 < with_contamination < 1.0):
            raise ValueError("with_contamination must be in (0, 1).")

        n_id = len(train_df_id)
        n_ood = int((with_contamination / (1 - with_contamination)) * n_id)

        ood_train_contamination = ood_df_train_pool.sample(n=n_ood, random_state=random_state, replace=False)
        set_binary_label(ood_train_contamination, indomain=False)

        train_df_id = pd.concat([train_df_id, ood_train_contamination], ignore_index=True)

    test_df = pd.concat([test_df_id, ood_df], ignore_index=True)

    return train_df_id, test_df, ood_df, test_df_id