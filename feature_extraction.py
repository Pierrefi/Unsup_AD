#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function

import sys, os
sys.path.append("/home/ids/fihey-23/new_project/pytorch-bertflow")
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import MinCovDet
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoConfig, AutoModelForCausalLM
import torch.nn.functional as F

# --- BERT-Flow (depuis le repo bohanli/BERT-flow) ---
from tflow_utils import TransformerGlow, AdamWeightDecayOptimizer

device = torch.device('cuda:0') 

def mean_pooling(last_hidden_state, attention_mask):

    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())    
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)    
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    
    return sum_embeddings / sum_mask

def feature_extract(train_df_id, test_df_id, ood_df, device, model_type): 
    
    if model_type == 'qwen3':
        model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', trust_remote_code=True).to(device)
    
        def extract_features(texts, model, batch_size=1):
            return model.encode(texts,batch_size=batch_size,convert_to_tensor=True, show_progress_bar=True,device=device).cpu().numpy()

        features_id_train = extract_features(train_df_id['text'].tolist(), model)
        features_id_val = extract_features(test_df_id['text'].tolist(), model)
        features_ood = extract_features(ood_df['text'].tolist(), model)

        return features_id_train, features_id_val, features_ood
    
    if model_type == 'e5':
        model = SentenceTransformer("intfloat/e5-base-v2").to(device)
    
        def extract_features(texts, model, batch_size=8):
            return model.encode(texts,batch_size=batch_size,convert_to_tensor=True, show_progress_bar=True,device=device).cpu().numpy()

        features_id_train = extract_features(train_df_id['text'].tolist(), model)
        features_id_val = extract_features(test_df_id['text'].tolist(), model)
        features_ood = extract_features(ood_df['text'].tolist(), model)

        return features_id_train, features_id_val, features_ood

    if model_type == 'llama':

        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
        model = AutoModel.from_pretrained('meta-llama/Llama-3.2-1B').to(device)

    if model_type == "qwen3llm":

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").to(device)

        def _sanitize_batch(batch_texts, tokenizer):
            clean = []
            for t in batch_texts:
                if t is None:
                    t = ""
                if not isinstance(t, str):
                    t = str(t)
                t = t.strip()
                if t == "":
                    t = tokenizer.eos_token if tokenizer.eos_token is not None else " "
                clean.append(t)
            return clean

        def extract_features(df, batch_size=2):
            texts = df["text"].tolist()
            embeddings = []

            model.eval()
            with torch.no_grad():
                for i in tqdm(range(0, len(texts), batch_size)):
                    batch_texts = _sanitize_batch(texts[i:i + batch_size], tokenizer)

                    tokens = tokenizer(
                        batch_texts,
                        padding=True,
                        return_tensors="pt",
                        truncation=True,
                        max_length=1024,
                        add_special_tokens=True,
                    )
                    tokens = {k: v.to(device) for k, v in tokens.items()}

                    # sécurité ultime: éviter seq_len=0
                    if tokens["input_ids"].shape[1] == 0:
                        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                        tokens["input_ids"] = torch.tensor([[eos_id]], device=device)
                        tokens["attention_mask"] = torch.tensor([[1]], device=device)

                    outputs = model(**tokens, output_hidden_states=True)
                    last_hidden = outputs.hidden_states[-1]  # (B, L, D)

                    pooled = mean_pooling(last_hidden, tokens["attention_mask"])
                    pooled = F.normalize(pooled, p=2, dim=1)
                    embeddings.append(pooled.cpu())

            return torch.cat(embeddings, dim=0).numpy()

        features_id_train = extract_features(train_df_id)
        features_id_val   = extract_features(test_df_id)
        features_ood      = extract_features(ood_df)

        return features_id_train, features_id_val, features_ood

    def extract_features_from_df(df):
        texts = df['text'].tolist()
        batch_size = 8
        embeddings = []
        model.eval()

        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i:i+batch_size]
                tokens = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
                outputs = model(**tokens)
                pooled = mean_pooling(outputs.last_hidden_state, tokens['attention_mask'])
                pooled = F.normalize(pooled, p=2, dim=1)
                embeddings.append(pooled)

        return torch.cat(embeddings, dim=0).cpu().numpy()

    features_id_train = extract_features_from_df(train_df_id)
    features_id_val = extract_features_from_df(test_df_id)
    features_id_ood = extract_features_from_df(ood_df)

    return features_id_train, features_id_val, features_id_ood

    