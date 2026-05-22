#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function

import os 
from scipy.signal import savgol_filter, argrelextrema
from sklearn.preprocessing import normalize
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import MinCovDet
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import pairwise_distances
from scipy import stats
from sklearn.metrics import roc_auc_score
from tqdm import tqdm 

from pyod.models.lunar import LUNAR
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.lof import LOF
from pyod.models.gmm import GMM
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from skdim.id import TwoNN

import torch
import torch.nn.functional as F
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
#from depth.multivariate import *
from depth.model import DepthEucl

import plotly.express as px

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
import calculate_log as callog
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from joblib import Parallel, delayed
from scipy.sparse.csgraph import dijkstra, shortest_path

device = torch.device('cuda:0') 
random_state = 53

def precompute_train_geodesics_landmarks(
    X_train,
    n_neighbors=50,
    metric='cosine',
    n_landmarks=256,
    landmark_strategy='random',
    seed=0,
    dtype=np.float32,
):

    n = X_train.shape[0]
    rng = np.random.default_rng(seed)

    # 1) Graphe kNN sparse (poids = distance)
    G = kneighbors_graph(X_train, n_neighbors=n_neighbors, mode='distance', metric=metric)
    G = 0.5 * (G + G.T)  # symétrise
    landmarks = rng.choice(n, size=n_landmarks, replace=False)

    D_lm = dijkstra(csgraph=G, directed=False, indices=landmarks, unweighted=False)
    D_lm = D_lm.astype(dtype, copy=False)  # (L, n)

    # 4) NN pour attacher les queries
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(X_train)

    return G, nn, landmarks, D_lm

@torch.no_grad()
def depth_landmarks_gpu(
    X_query_np,        # (m,d) numpy float32
    X_lm_np,           # (L,d) numpy float32
    D_lm_np,           # (L,n) numpy float32
    agg="median",
    batch_size=256,
    chunk_size=5000,   # chunk sur n_train
    device="cuda",
):
    # to torch
    Q = torch.from_numpy(X_query_np).to(device)
    Xlm = torch.from_numpy(X_lm_np).to(device)
    Dlm = torch.from_numpy(D_lm_np).to(device)

    # cosine => normalisation
    Q = torch.nn.functional.normalize(Q, p=2, dim=1)
    Xlm = torch.nn.functional.normalize(Xlm, p=2, dim=1)

    m = Q.shape[0]
    L, n = Dlm.shape
    depths = torch.empty(m, device=device, dtype=torch.float32)

    for b0 in range(0, m, batch_size):
        b1 = min(b0 + batch_size, m)
        Qb = Q[b0:b1]  # (B,d)

        # (B,L): cosine distance = 1 - dot
        d_q_lm = 1.0 - (Qb @ Xlm.T)

        # approx distances (B,n) en chunks
        # on stocke pour median exacte
        approx = torch.empty((Qb.shape[0], n), device=device, dtype=torch.float32)

        for j0 in range(0, n, chunk_size):
            j1 = min(j0 + chunk_size, n)
            Dchunk = Dlm[:, j0:j1]  # (L,C)
            # (B,L,1) + (1,L,C) -> min over L => (B,C)
            tmp = torch.min(d_q_lm[:, :, None] + Dchunk[None, :, :], dim=1).values
            approx[:, j0:j1] = tmp

        if agg == "median":
            s = approx.median(dim=1).values
        else:
            s = approx.mean(dim=1)

        depths[b0:b1] = 1.0 / (1.0 + s)

    return depths.detach().cpu().numpy()


def local_l2_depth_keep_ratio(X_query, X_train, keep_ratio=0.01, agg='median'):

    X_train = np.asarray(X_train, dtype=np.float64)
    X_query = np.asarray(X_query, dtype=np.float64)

    n_train = X_train.shape[0]
    k = max(1, int(keep_ratio * n_train))

    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(X_train)

    dists, _ = nn.kneighbors(X_query, return_distance=True)  # (n_query, k)

    if agg == 'median':
        s = np.median(dists, axis=1)
    else:
        s = np.mean(dists, axis=1)

    depths = (1.0 / (1.0 + s)).astype(np.float32)
    return depths

def l2_depth_global(X_query, X_train, agg='median', metric='cosine'):

    X_query = np.asarray(X_query, dtype=np.float64)
    X_train = np.asarray(X_train, dtype=np.float64)

    D = pairwise_distances(X_query, X_train, metric=metric)  # (n_query, n_train)

    if agg == 'median':
        s = np.median(D, axis=1)
    else:
        s = np.mean(D, axis=1)

    return (1.0 / (1.0 + s)).astype(np.float32)

def detection_performance(scores, Y, outf, tag='TMP'):
    import os 
    os.makedirs(outf, exist_ok=True)
    num_samples = scores.shape[0]
    l1 = open('%s/confidence_%s_In.txt'%(outf, tag), 'w')
    l2 = open('%s/confidence_%s_Out.txt'%(outf, tag), 'w')
    y_pred = scores 

    for i in range(num_samples):
        if Y[i] == 0:
            l1.write("{}\n".format(-y_pred[i]))
        else:
            l2.write("{}\n".format(-y_pred[i]))
    l1.close()
    l2.close()
    results = callog.metric(outf, [tag])
    return results


def append_to_latex_table(file_path, model_name, results, mtypes=None):
    if mtypes is None:
        mtypes = ['AUROC', 'DTACC', 'AUIN', 'AUOUT']
    print(os)
    dir_name = os.path.dirname(file_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Remove the last few lines (\hline and \end{tabular}) so we can append new data
        lines = lines[:-3]

        # Append new row to the LaTeX table
        with open(file_path, 'w') as f:
            f.writelines(lines)  # Write back all the lines except \hline and \end{tabular}
            f.write(f'\\textbf{{{model_name}}} & ' + ' & '.join([f'{100.*results["TMP"][mtype]:.2f}' for mtype in mtypes]) + ' \\\\\n')
            f.write('\\hline\n')
            f.write('    \\end{tabular}\n')
            f.write('\\caption{Metrics Results}\n')
            f.write('\\label{tab:metrics}\n')
            f.write('\\end{table}\n')
    else:
        # Create a new LaTeX table with headers
        with open(file_path, 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("    \\centering\n")
            f.write("    \\begin{tabular}{|c|" + "c|"*len(mtypes) + "}\n")
            f.write("        \\hline\n")
            f.write("         Model & " + ' & '.join([f'\\textbf{{{mtype}}}' for mtype in mtypes]) + " \\\\\n")
            f.write("         \\hline\n")
            f.write(f'\\textbf{{{model_name}}} & ' + ' & '.join([f'{100.*results["TMP"][mtype]:.2f}' for mtype in mtypes]) + ' \\\\\n')
            f.write('        \\hline\n')
            f.write('    \\end{tabular}\n')
            f.write('\\caption{Metrics Results}\n')
            f.write('\\label{tab:metrics}\n')
            f.write('\\end{table}\n')

    print(f"LaTeX table updated and saved to {file_path}")


def precompute_train_geodesics(X_train, n_neighbors=10, metric='euclidean'):
    """
    Construit le graphe kNN pondéré sur le train et calcule les plus courts chemins entre
    tous les points d'entraînement (APSP). Retourne aussi un kNN pour connecter les requêtes.
    """
    G = kneighbors_graph(X_train, n_neighbors=n_neighbors, mode='distance', metric=metric)
    G = 0.5 * (G + G.T)  # symétrise
    D_train = shortest_path(G, directed=False, unweighted=False)  # (n_train, n_train)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(X_train)
    return D_train, nn

def geodesic_l2_depth_queries(X_query, X_train, D_train, nn, k_attach=10, agg='median'):
    depths = np.empty(len(X_query), dtype=np.float32)
    for idx, q in enumerate(X_query):
        d, I = nn.kneighbors(q[None, :], n_neighbors=k_attach, return_distance=True)
        approx_geo = np.min(d[0][:, None] + D_train[I[0], :], axis=0)
        s = np.median(approx_geo) if agg == 'median' else np.mean(approx_geo)
        depths[idx] = 1.0 / (1.0 + float(s))
    return depths

def whitening(X):

    mu = np.mean(X, axis=0)
    X_centered = X - mu
    cov = np.cov(X_centered, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    W = vectors @ np.diag(1.0 / np.sqrt(values + 1e-5)) @ vectors.T

    return X_centered @ W

def compute_whitening_params(X):

    mu = np.mean(X, axis=0)
    X_centered = X - mu
    cov = np.cov(X_centered, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    W = vectors @ np.diag(1.0 / np.sqrt(values + 1e-5)) @ vectors.T  
    return mu, W

def apply_whitening(X, mu, W):

    X_centered = X - mu
    return X_centered @ W

def geodesic_l2_depth(
    X_query, X_train, n_neighb,
    metric="cosine", agg="median",           
    landmarks=False,
    n_landmarks=256,
    seed=0,
    device="cuda",
    batch_size=256,
    chunk_size=5000,
):
    n = X_train.shape[0]
    n_neighbors = n_neighb 
    k_attach  = n_neighbors 

    if not landmarks:
        D_train, nn = precompute_train_geodesics(X_train, n_neighbors=n_neighbors, metric=metric)
        return geodesic_l2_depth_queries(X_query, X_train, D_train, nn, k_attach=k_attach, agg=agg)

    _, _, lm_idx, D_lm = precompute_train_geodesics_landmarks(
        X_train, n_neighbors=n_neighbors, metric=metric,
        n_landmarks=min(n_landmarks, n), seed=seed, dtype=np.float32
    )
    X_lm = X_train[lm_idx].astype(np.float32, copy=False)
    return depth_landmarks_gpu(
        X_query.astype(np.float32, copy=False), X_lm, D_lm,
        agg=agg, batch_size=batch_size, chunk_size=chunk_size, device=device
    )


def ood_detection_every_combination(loss_fn_name, features_id_train, features_id_val, features_ood, post_proc_method, method, dataset_name, model_type, language, support_fraction = 0.6): 

    if post_proc_method == 'train_and_test_whitening_with_train_params':

        mu_train, W_train = compute_whitening_params(features_id_train)
        features_id_train = apply_whitening(features_id_train, mu_train, W_train)

        features_combined = np.concatenate((features_id_val, features_ood), axis=0)
        features_combined = apply_whitening(features_combined, mu_train, W_train)
        num_id_val = features_id_val.shape[0]
        features_id_val = features_combined[:num_id_val]
        features_ood = features_combined[num_id_val:num_id_val + features_ood.shape[0]]

        loss_fn_name += '_with_train_and_test_whitening_with_train_params'

    if method == 'ocsvm' : 

        ocsvm_model = OCSVM()  
        ocsvm_model.fit(features_id_train)

        test_scores = ocsvm_model.decision_function(features_id_val)  
        ood_scores = ocsvm_model.decision_function(features_ood)

        X_scores = np.concatenate((ood_scores, test_scores))
        ood_labels = np.ones_like(ood_scores)
        test_labels = np.zeros_like(test_scores)
        Y_test = np.concatenate((ood_labels, test_labels))

        best_results = detection_performance(X_scores, Y_test, 'feats_logs', tag='TMP')

        file_path = f'/results_tables_anomaly/{dataset_name}_results/{model_type}/{model_type}_ocsvm.tex'
  
        append_to_latex_table(file_path, loss_fn_name, best_results)

        mtypes = ['AUROC', 'DTACC', 'AUIN', 'AUOUT']
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*best_results['TMP']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*best_results['TMP']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*best_results['TMP']['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*best_results['TMP']['AUOUT']), end='')
    
    if method == 'lof' : 

        lof_model = LOF()  
        lof_model.fit(features_id_train)

        test_scores = lof_model.decision_function(features_id_val)  
        ood_scores = lof_model.decision_function(features_ood)

        X_scores = np.concatenate((ood_scores, test_scores))
        ood_labels = np.ones_like(ood_scores)
        test_labels = np.zeros_like(test_scores)
        Y_test = np.concatenate((ood_labels, test_labels))

        best_results = detection_performance(X_scores, Y_test, 'feats_logs', tag='TMP')
        file_path = f'/results_tables_anomaly/{dataset_name}_results/{model_type}/{model_type}_lof.tex'
 
        append_to_latex_table(file_path, loss_fn_name, best_results)

        mtypes = ['AUROC', 'DTACC', 'AUIN', 'AUOUT']
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*best_results['TMP']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*best_results['TMP']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*best_results['TMP']['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*best_results['TMP']['AUOUT']), end='')
    
    if method == 'isolation_forest':

        if_model = IForest()  
        if_model.fit(features_id_train)

        test_scores = if_model.decision_function(features_id_val)  
        ood_scores = if_model.decision_function(features_ood)

        X_scores = np.concatenate((ood_scores, test_scores))
        ood_labels = np.ones_like(ood_scores)
        test_labels = np.zeros_like(test_scores)
        Y_test = np.concatenate((ood_labels, test_labels))

        best_results = detection_performance(X_scores, Y_test, 'feats_logs', tag='TMP')
        file_path = f'/results_tables_anomaly/{dataset_name}_results/{model_type}/{model_type}_isolation_forest.tex'

        append_to_latex_table(file_path, loss_fn_name, best_results)

        mtypes = ['AUROC', 'DTACC', 'AUIN', 'AUOUT']
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*best_results['TMP']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*best_results['TMP']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*best_results['TMP']['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*best_results['TMP']['AUOUT']), end='')

    if method == 'knn': 

        knn_model = KNN()  
        knn_model.fit(features_id_train)

        # Get anomaly scores for validation and OOD datasets
        test_scores = knn_model.decision_function(features_id_val)  
        ood_scores = knn_model.decision_function(features_ood)

        # Combine scores and labels
        X_scores = np.concatenate((ood_scores, test_scores))
        ood_labels = np.ones_like(ood_scores)
        test_labels = np.zeros_like(test_scores)
        Y_test = np.concatenate((ood_labels, test_labels))

        # Evaluate detection performance
        best_results = detection_performance(X_scores, Y_test, 'feats_logs', tag='TMP')

        file_path = f'/results_tables_anomaly/{dataset_name}_results/{model_type}/{model_type}_knn.tex'

        append_to_latex_table(file_path, loss_fn_name, best_results)

        # Affichage des meilleurs résultats
        mtypes = ['AUROC', 'DTACC', 'AUIN', 'AUOUT']
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*best_results['TMP']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*best_results['TMP']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*best_results['TMP']['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*best_results['TMP']['AUOUT']), end='')

    if method == 'gmm': 

        gmm_model = GMM()  
        gmm_model.fit(features_id_train)

        # Get anomaly scores for validation and OOD datasets
        test_scores = gmm_model.decision_function(features_id_val)  
        ood_scores = gmm_model.decision_function(features_ood)

        # Combine scores and labels
        X_scores = np.concatenate((ood_scores, test_scores))
        ood_labels = np.ones_like(ood_scores)
        test_labels = np.zeros_like(test_scores)
        Y_test = np.concatenate((ood_labels, test_labels))

        # Evaluate detection performance
        best_results = detection_performance(X_scores, Y_test, 'feats_logs', tag='TMP')
        
        file_path = f'/results_tables_anomaly/{dataset_name}_results/{model_type}/{model_type}_gmm.tex'

        append_to_latex_table(file_path, loss_fn_name, best_results)
        mtypes = ['AUROC', 'DTACC', 'AUIN', 'AUOUT']
        for mtype in mtypes:
            print(f' {mtype:6s}', end='')
        print(f'\n{best_results["TMP"]["AUROC"] * 100:6.2f}', end='')
        print(f' {best_results["TMP"]["DTACC"] * 100:6.2f}', end='')
        print(f' {best_results["TMP"]["AUIN"] * 100:6.2f}', end='')
        print(f' {best_results["TMP"]["AUOUT"] * 100:6.2f}\n', end='')

    if method == 'lunar':

        lunar_model = LUNAR()  
        lunar_model.fit(features_id_train)

        # Get anomaly scores for validation and OOD datasets
        test_scores = lunar_model.decision_function(features_id_val)  # Higher scores mean more abnormal
        ood_scores = lunar_model.decision_function(features_ood)

        # Combine scores and labels
        X_scores = np.concatenate((ood_scores, test_scores))
        ood_labels = np.ones_like(ood_scores)
        test_labels = np.zeros_like(test_scores)
        Y_test = np.concatenate((ood_labels, test_labels))

        # Evaluate detection performance
        results = detection_performance(X_scores, Y_test, 'feats_logs', tag='TMP')
        
        file_path = f'/results_tables_anomaly/{dataset_name}_results/{model_type}/{model_type}_lunar.tex'

        append_to_latex_table(file_path, loss_fn_name, results)

        # Print the best metrics
        mtypes = ['AUROC', 'DTACC', 'AUIN', 'AUOUT']
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results['TMP']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['TMP']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['TMP']['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results['TMP']['AUOUT']), end='')

    if method == 'autoencoder':
            
        autoencoder = AutoEncoder()        
        autoencoder.fit(features_id_train)

        # Obtenir les scores d'anomalie (erreur de reconstruction)
        test_scores = autoencoder.decision_function(features_id_val)
        ood_scores = autoencoder.decision_function(features_ood)

        # Combinaison des scores et des étiquettes
        X_scores = np.concatenate((ood_scores, test_scores))
        ood_labels = np.ones_like(ood_scores)
        test_labels = np.zeros_like(test_scores)
        Y_test = np.concatenate((ood_labels, test_labels))

        # Évaluation des performances de détection
        results = detection_performance(X_scores, Y_test, 'feats_logs', tag='TMP')
        file_path = f'/results_tables_anomaly/{dataset_name}_results/{model_type}/{model_type}_autoencoder.tex'

        append_to_latex_table(file_path, loss_fn_name, results)

        # Affichage des meilleurs résultats
        mtypes = ['AUROC', 'DTACC', 'AUIN', 'AUOUT']
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results['TMP']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['TMP']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['TMP']['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results['TMP']['AUOUT']), end='')

    if method == 'projection_depth': 

        model = DepthEucl().load_dataset(features_id_train, CUDA=True)
        depth_test = model.projection(features_id_val, NRandom = 10000, solver = 'refinedrandom', CUDA = True, output_option='lowest_depth')
        depth_ood  = model.projection(features_ood, NRandom = 10000, solver = 'refinedrandom', CUDA = True, output_option='lowest_depth')

        test_scores = -depth_test
        ood_scores  = -depth_ood

        X_scores = np.concatenate((ood_scores, test_scores))
        Y_test   = np.concatenate((np.ones_like(ood_scores), np.zeros_like(test_scores)))

        results    = detection_performance(X_scores, Y_test, 'feats_logs', tag='TMP')
        neg_results= detection_performance(-X_scores, Y_test, 'feats_logs', tag='TMP')

        
        file_path = (f'/results_tables_anomaly/{dataset_name}_results/{model_type}/{model_type}_projection_depth.tex')
        append_to_latex_table(file_path, loss_fn_name, results)

        # 5) Affichage (identique à ton bloc AE)
        mtypes = ['AUROC', 'DTACC', 'AUIN', 'AUOUT']
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results['TMP']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['TMP']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['TMP']['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results['TMP']['AUOUT']), end='')

    if method == 'l2_depth': 

        model = DepthEucl().load_dataset(features_id_train)

        depth_test = model.L2(features_id_val)
        depth_ood  = model.L2(features_ood)

        test_scores = -depth_test
        ood_scores  = -depth_ood

        X_scores = np.concatenate((ood_scores, test_scores))
        Y_test   = np.concatenate((np.ones_like(ood_scores), np.zeros_like(test_scores)))

        results    = detection_performance(X_scores, Y_test, 'feats_logs', tag='TMP')

        file_path = (f'/results_tables_anomaly/{dataset_name}_results/{model_type}/{model_type}_l2_depth.tex')
        append_to_latex_table(file_path, loss_fn_name, results)

        # 5) Affichage (identique à ton bloc AE)
        mtypes = ['AUROC', 'DTACC', 'AUIN', 'AUOUT']
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results['TMP']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['TMP']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['TMP']['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results['TMP']['AUOUT']), end='')

    if method == "geodesic_l2_depth":
        use_landmarks = False  # <- switch ici
        n = features_id_train.shape[0]
        n_neighbors = int(0.15 * n)

        depth_test = geodesic_l2_depth(features_id_val, features_id_train, n_neighb = n_neighbors,  landmarks=use_landmarks)
        depth_ood  = geodesic_l2_depth(features_ood,    features_id_train, n_neighb = n_neighbors,  landmarks=use_landmarks)

        test_scores = -depth_test
        ood_scores  = -depth_ood
        X_scores = np.concatenate((ood_scores, test_scores))
        Y_test   = np.concatenate((np.ones_like(ood_scores), np.zeros_like(test_scores)))

        results = detection_performance(X_scores, Y_test, "feats_logs", tag="TMP")

        suffix = "landmarks" if use_landmarks else "exact"
        file_path = (
            f"/results_tables_anomaly/{dataset_name}_results/{model_type}/"
            f"{model_type}_Geodesic_L2_depth_{suffix}.tex"
        )
        append_to_latex_table(file_path, loss_fn_name, results)
        # 5) Affichage (identique à ton bloc AE)
        mtypes = ['AUROC', 'DTACC', 'AUIN', 'AUOUT']
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results['TMP']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['TMP']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['TMP']['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results['TMP']['AUOUT']), end='')

    if method == 'l2_depth_global':
    
        depth_test = l2_depth_global(features_id_val, features_id_train, agg = 'median', metric = 'euclidean')
        depth_ood  = l2_depth_global(features_ood, features_id_train, agg = 'median', metric = 'euclidean')

        test_scores = -depth_test; ood_scores = -depth_ood
        X_scores = np.concatenate((ood_scores, test_scores))
        Y_test   = np.concatenate((np.ones_like(ood_scores), np.zeros_like(test_scores)))
        results     = detection_performance(X_scores,  Y_test, 'feats_logs', tag='TMP')

        file_path = (
            f"/results_tables_anomaly/{dataset_name}_results/{model_type}/"
            f"{model_type}_L2_depth_global_median.tex")
        append_to_latex_table(file_path, loss_fn_name, results)

        for mtype in ['AUROC','DTACC','AUIN','AUOUT']: print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results['TMP']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['TMP']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['TMP']['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results['TMP']['AUOUT']), end='')