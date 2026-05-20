#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import argparse
import numpy as np


BASE_DIR = "/home/ids/fihey-23/new_code_contamination/embeddings_files"


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_feature_triplet(model_name, dataset_name, base_dir=BASE_DIR):
    root = os.path.join(base_dir, model_name, dataset_name)

    train_features = np.asarray(load_pkl(os.path.join(root, "train_features.pkl")))
    test_features = np.asarray(load_pkl(os.path.join(root, "test_features.pkl")))
    ood_features = np.asarray(load_pkl(os.path.join(root, "ood_features.pkl")))

    return train_features, test_features, ood_features

def make_contaminated_train(train_features, ood_features, contamination=0.1, seed=42):
    rng = np.random.default_rng(seed)

    X_id = np.asarray(train_features)
    X_ood_pool = np.asarray(ood_features)

    n_id = len(X_id)

    if contamination <= 0.0:
        y_train = np.zeros(n_id, dtype=np.int64)
        return X_id.copy(), y_train, np.array([], dtype=np.int64)

    n_ood = int(round((contamination / (1.0 - contamination)) * n_id))
    n_ood = min(n_ood, len(X_ood_pool))

    idx_ood_used = rng.choice(len(X_ood_pool), size=n_ood, replace=False)
    X_ood_cont = X_ood_pool[idx_ood_used]

    X_train = np.concatenate([X_id, X_ood_cont], axis=0)
    y_train = np.concatenate([
        np.zeros(len(X_id), dtype=np.int64),
        np.ones(len(X_ood_cont), dtype=np.int64),
    ])

    perm = rng.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    return X_train, y_train, idx_ood_used


def save_novelty_detection_folder(root_dir, train_features, test_features, ood_features):
    os.makedirs(root_dir, exist_ok=True)
    save_pkl(train_features, os.path.join(root_dir, "train_features.pkl"))
    save_pkl(test_features, os.path.join(root_dir, "test_features.pkl"))
    save_pkl(ood_features, os.path.join(root_dir, "ood_features.pkl"))


def save_outlier_detection_folder(root_dir, train_features, train_labels):
    os.makedirs(root_dir, exist_ok=True)
    save_pkl(train_features, os.path.join(root_dir, "train_features.pkl"))
    save_pkl(train_labels, os.path.join(root_dir, "train_labels.pkl"))


def create_contaminated_feature_set(
    dataset_name,
    source_model_name="e5",
    contamination=0.1,
    seed=42,
    base_dir=BASE_DIR,
):
    train_features, test_features, ood_features = load_feature_triplet(
        source_model_name, dataset_name, base_dir=base_dir
    )

    print(f"\nDataset: {dataset_name}")
    print(f"Original train: {len(train_features)}")
    print(f"Original test : {len(test_features)}")
    print(f"OOD pool       : {len(ood_features)}")

    train_features_cont, y_train_outlier, idx_ood_used = make_contaminated_train(
    train_features=train_features,
    ood_features=ood_features,
    contamination=contamination,
    seed=seed,
)

    mask_ood_test = np.ones(len(ood_features), dtype=bool)
    mask_ood_test[idx_ood_used] = False
    ood_features_test = ood_features[mask_ood_test]

    new_model_name = f"{source_model_name}_{contamination:.1f}"
    dataset_root = os.path.join(base_dir, new_model_name, dataset_name)

    novelty_root = os.path.join(dataset_root, "novelty_detection")
    
    save_novelty_detection_folder(
    root_dir=novelty_root,
    train_features=train_features_cont,
    test_features=test_features,
    ood_features=ood_features_test,
)

    outlier_root = os.path.join(dataset_root, "outlier_detection")
    save_outlier_detection_folder(
        root_dir=outlier_root,
        train_features=train_features_cont,
        train_labels=y_train_outlier,
    )

    final_cont = y_train_outlier.mean()

    print(f"[OK] saved -> {dataset_root}")
    print(f"[INFO] contamination requested = {contamination:.4f}")
    print(f"[INFO] contamination obtained  = {final_cont:.4f}")
    print(f"[INFO] final train size        = {len(train_features_cont)}")

    return {
        "new_model_name": new_model_name,
        "dataset_name": dataset_name,
        "dataset_root": dataset_root,
        "novelty_root": novelty_root,
        "outlier_root": outlier_root,
    }


def get_all_datasets_for_model(model_name, base_dir=BASE_DIR):
    model_dir = os.path.join(base_dir, model_name)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    dataset_names = []
    for name in sorted(os.listdir(model_dir)):
        dataset_dir = os.path.join(model_dir, name)
        if not os.path.isdir(dataset_dir):
            continue

        needed = [
            os.path.join(dataset_dir, "train_features.pkl"),
            os.path.join(dataset_dir, "test_features.pkl"),
            os.path.join(dataset_dir, "ood_features.pkl"),
        ]
        if all(os.path.isfile(p) for p in needed):
            dataset_names.append(name)

    return dataset_names


def create_for_all_datasets(
    source_model_name="e5",
    contamination=0.1,
    seed=42,
    base_dir=BASE_DIR,
):
    dataset_names = get_all_datasets_for_model(source_model_name, base_dir=base_dir)

    if not dataset_names:
        raise ValueError(f"No valid datasets found for model {source_model_name}")

    print(f"[INFO] Found {len(dataset_names)} datasets for model {source_model_name}")
    for d in dataset_names:
        print(f"  - {d}")

    results = []
    for dataset_name in dataset_names:
        res = create_contaminated_feature_set(
            dataset_name=dataset_name,
            source_model_name=source_model_name,
            contamination=contamination,
            seed=seed,
            base_dir=base_dir,
        )
        results.append(res)

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create contaminated novelty/outlier datasets for all datasets of a given embedding model."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Source model name, e.g. e5",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        required=True,
        help="Final contamination rate in the train set, e.g. 0.1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=BASE_DIR,
        help="Base embeddings directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not (0.0 <= args.contamination < 1.0):
        raise ValueError("--contamination must be in [0, 1)")

    create_for_all_datasets(
        source_model_name=args.model_name,
        contamination=args.contamination,
        seed=args.seed,
        base_dir=args.base_dir,
    )


if __name__ == "__main__":
    main()