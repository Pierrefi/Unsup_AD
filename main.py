#!/usr/bin/env python3
import os
import sys 
import time

# Ajouter le dossier parent au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ad_algorithms_outlier import ood_detection_every_combination_outlier
import torch
import pandas as pd
from datasets import load_dataset
from data_processing import save_features, load_features, create_datasets_df, create_datasets_nytimes
from ad_algorithms import ood_detection_every_combination
import numpy as np
import pickle
import argparse, time

device = torch.device('cuda:0') 

def dataloaders_creation(name, contamination): 

    if name == 'agnews_world':
        dataset = load_dataset("SetFit/ag_news")
        id_labels, ood_labels =  [0], list(range(1, 4))

        train_df_id, test_df, ood_df, test_df_id  = create_datasets_df(dataset, id_labels, ood_labels, sample = True, with_contamination = contamination)

    elif name == 'agnews_sports':
        dataset = load_dataset("SetFit/ag_news")
        id_labels, ood_labels =  [1], [0, 2, 3]

        train_df_id, test_df, ood_df, test_df_id  = create_datasets_df(dataset, id_labels, ood_labels, sample = True, with_contamination = contamination)

    elif name == 'agnews_business':
        dataset = load_dataset("SetFit/ag_news")
        id_labels, ood_labels =  [2], [0, 1, 3]

        train_df_id, test_df, ood_df, test_df_id  = create_datasets_df(dataset, id_labels, ood_labels, sample = True, with_contamination = contamination)
    
    elif name == 'agnews_scitech':
        dataset = load_dataset("SetFit/ag_news")
        id_labels, ood_labels =  [3], [0, 1, 2]

        train_df_id, test_df, ood_df, test_df_id  = create_datasets_df(dataset, id_labels, ood_labels, sample = True, with_contamination = contamination)

    elif name == 'bbc_tech':
        dataset = load_dataset("SetFit/bbc-news")
        id_labels, ood_labels =  [0], [1, 2, 3, 4]

        train_df_id, test_df, ood_df, test_df_id  = create_datasets_df(dataset, id_labels, ood_labels, sample = False, with_contamination = contamination)

    elif name == 'bbc_business':
        dataset = load_dataset("SetFit/bbc-news")
        id_labels, ood_labels =  [1], [0, 2, 3, 4]

        train_df_id, test_df, ood_df, test_df_id  = create_datasets_df(dataset, id_labels, ood_labels, sample = False, with_contamination = contamination)

    elif name == 'bbc_sport':
        dataset = load_dataset("SetFit/bbc-news")
        id_labels, ood_labels =  [2], [0, 1, 3, 4]

        train_df_id, test_df, ood_df, test_df_id  = create_datasets_df(dataset, id_labels, ood_labels, sample = False, with_contamination = contamination)

    elif name == 'bbc_entertainment':
        dataset = load_dataset("SetFit/bbc-news")
        id_labels, ood_labels =  [3], [0, 1, 2, 4]

        train_df_id, test_df, ood_df, test_df_id  = create_datasets_df(dataset, id_labels, ood_labels, sample = False, with_contamination = contamination)

    elif name == 'bbc_politics':
        dataset = load_dataset("SetFit/bbc-news")
        id_labels, ood_labels =  [4], [0, 1, 2, 3]

        train_df_id, test_df, ood_df, test_df_id  = create_datasets_df(dataset, id_labels, ood_labels, sample = False, with_contamination = contamination)

    elif name == "n24_health":
        train_df_id, test_df, ood_df, test_df_id = create_datasets_nytimes(id_label="Health", with_contamination=contamination)

    elif name == "n24_science":
        train_df_id, test_df, ood_df, test_df_id = create_datasets_nytimes(id_label="Science", with_contamination=contamination)

    elif name == "n24_sports":
        train_df_id, test_df, ood_df, test_df_id = create_datasets_nytimes(id_label="Sports", with_contamination=contamination)

    elif name == "n24_technology":
        train_df_id, test_df, ood_df, test_df_id = create_datasets_nytimes(id_label="Technology", with_contamination=contamination)

    elif name == "n24_movies":
        train_df_id, test_df, ood_df, test_df_id = create_datasets_nytimes(id_label="Movies", with_contamination=contamination)

    elif name == '20ng_alt' :
        dataset = load_dataset("SetFit/20_newsgroups")
        id_labels = [0]
        ood_labels = [label for label in range(20) if label not in id_labels]

        train_df_id, test_df, ood_df, test_df_id  = create_datasets_df(dataset, id_labels, ood_labels, with_contamination = contamination)

    elif name == '20ng_comp' :
        dataset = load_dataset("SetFit/20_newsgroups")
        id_labels = list(range(1, 6))
        ood_labels = [label for label in range(20) if label not in id_labels]

        train_df_id, test_df, ood_df, test_df_id  = create_datasets_df(dataset, id_labels, ood_labels, with_contamination = contamination)

    elif name == '20ng_rec' :
        dataset = load_dataset("SetFit/20_newsgroups")
        id_labels = list(range(7, 11))
        ood_labels = [label for label in range(20) if label not in id_labels]

        train_df_id, test_df, ood_df, test_df_id  = create_datasets_df(dataset, id_labels, ood_labels, with_contamination = contamination)
    
    elif name == '20ng_sci' :
        dataset = load_dataset("SetFit/20_newsgroups")
        id_labels = list(range(11, 15))
        ood_labels = [label for label in range(20) if label not in id_labels]

        train_df_id, test_df, ood_df, test_df_id  = create_datasets_df(dataset, id_labels, ood_labels, with_contamination = contamination)

    elif name == '20ng_rel' :
        dataset = load_dataset("SetFit/20_newsgroups")
        id_labels =  [15, 19]
        ood_labels = [label for label in range(20) if label not in id_labels]

        train_df_id, test_df, ood_df, test_df_id  = create_datasets_df(dataset, id_labels, ood_labels, with_contamination = contamination)

    elif name == '20ng_misc' :
        dataset = load_dataset("SetFit/20_newsgroups")
        id_labels =  [6]
        ood_labels = [label for label in range(20) if label not in id_labels]

        train_df_id, test_df, ood_df, test_df_id  = create_datasets_df(dataset, id_labels, ood_labels, with_contamination = contamination)

    elif name == '20ng_pol' :
        dataset = load_dataset("SetFit/20_newsgroups")
        id_labels =  list(range(16, 19))
        ood_labels = [label for label in range(20) if label not in id_labels]

        train_df_id, test_df, ood_df, test_df_id  = create_datasets_df(dataset, id_labels, ood_labels, with_contamination = contamination)

    return train_df_id, test_df_id, ood_df

state_dicts = {
}

def task_extract_features(dataset_names, contamination, model_name, data_tag, device):

    for dataset_name in dataset_names : 

        train_df_id, test_df_id, ood_df = dataloaders_creation(dataset_name, contamination)
        features_id_train, features_id_test, features_ood = feature_extract(train_df_id, test_df_id, ood_df, device, model_name) 
        save_features(features_id_train, features_id_test, features_ood, f'{model_name}_{contamination}', dataset_name, data = '')
        print(f"[OK] Features sauvées → model={model_name} dataset={dataset_name} data_tag={data_tag}")


def task_do_test(dataset_names, model_name, methods, seed, post_proc_method = ''):
    """
    Charge les features sauvegardées et lance les tests OOD pour chaque méthode.
    """
    rng = np.random.default_rng(seed)

    for dataset_name in dataset_names :

        print('Dataset : ', dataset_name)
        features = load_features(dataset_name, model_name)
        train_features = features["train_features"]
        test_features = features["test_features"]
        ood_features = features["ood_features"]

        idx_test = np.random.choice(len(test_features), size=min(10000,len(test_features)) , replace=False)
        idx_ood = np.random.choice(len(ood_features), size=min(1000,len(ood_features)), replace=False)

        test_features = test_features[idx_test]
        ood_features = ood_features[idx_ood]

        print(f"Train features size: {train_features.shape[0]}")
        print(f"Validation features size: {test_features.shape[0]}")
        print(f"OOD features size: {ood_features.shape[0]}")

        for method in methods : 
            start_time = time.perf_counter()
            if post_proc_method == '':
                ood_detection_every_combination('', train_features, test_features, ood_features, 'no_post_processing', method, dataset_name, model_name, language = 'english')
            else :
                ood_detection_every_combination('', train_features, test_features, ood_features, post_proc_method, method, dataset_name, model_name, language = 'english')
            end_time = time.perf_counter()
            duration = end_time - start_time
            print(f"⏱️ Temps écoulé : {duration/60:.2f} min ({duration:.1f} s)")

def task_do_test_outlier(dataset_names, model_name, methods, seed, post_proc_method=""):

    base_dir = "/embeddings_files"

    def load_pkl(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    for dataset_name in dataset_names:
        print("Dataset:", dataset_name)
        root = os.path.join(
            base_dir,
            model_name,
            dataset_name,
            "outlier_detection"
        )
        train_features = np.asarray(
            load_pkl(os.path.join(root, "train_features.pkl"))
        )
        y_train_values = np.asarray(
            load_pkl(os.path.join(root, "train_labels.pkl"))
        )
        print(f"Train features size: {train_features.shape[0]}")
        print(f"Train labels size:   {y_train_values.shape[0]}")
        print(f"Contamination:       {y_train_values.mean():.4f}")

        for method in methods:
            start_time = time.perf_counter()
            pp = post_proc_method if post_proc_method != "" else "no_post_processing"
            ood_detection_every_combination_outlier(
                "",
                train_features,
                y_train_values,
                pp,
                method,
                dataset_name,
                model_name,
            )
            duration = time.perf_counter() - start_time
            print(f"Temps écoulé : {duration/60:.2f} min ({duration:.1f} s)")

def parse_list(arg: str):
    """Transforme 'a,b,c' -> ['a','b','c'] en ignorant espaces/vides."""
    if arg is None or arg == "":
        return []
    return [x.strip() for x in arg.split(",") if x.strip()]

def pick_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    p = argparse.ArgumentParser(description="Runner simple pour extract_features / do_test (arguments minimalistes).")
    p.add_argument("--dataset_names", required=True,
                   help="Liste de datasets séparés par des virgules (ex: 'banking,clinc').")
    p.add_argument("--contamination", type=float, default=0.0,
                   help="Taux de contamination pour dataloaders_creation (uniquement pour extract_features).")
    p.add_argument("--model_name", required=True,
                   help="Nom du modèle pour sauvegarde/chargement des features (ex: 'sbert_cont_40').")
    p.add_argument("--methods", default="",
                   help="Liste de méthodes séparées par des virgules pour do_test (ex: 'knn,ocsvm,projection_depth').")
    p.add_argument("--post_proc", default="")
    p.add_argument("--seed", type=int, default=42, help="Graine aléatoire pour l'échantillonnage.")

    args = p.parse_args()

    dataset_names = parse_list(args.dataset_names)
    methods = parse_list(args.methods)

    device = pick_device()
    print(f"[info] device={device}")

    task_do_test_outlier(
        dataset_names=dataset_names,
        model_name=args.model_name,
        methods=methods,
        seed=args.seed,
        post_proc_method=args.post_proc
    )

if __name__ == "__main__":
    main()

