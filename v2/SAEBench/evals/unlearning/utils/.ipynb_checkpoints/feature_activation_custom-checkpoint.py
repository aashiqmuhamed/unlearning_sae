from datasets import load_dataset
import json
import einops
from tqdm import tqdm
import torch
from torch import Tensor
from jaxtyping import Float
import gc
import numpy as np
import random
import os

from sae_lens import SAE
from transformer_lens import HookedTransformer

from sae_bench_utils.activation_collection import get_feature_activation_sparsity
import sae_bench_utils.dataset_utils as dataset_utils

FORGET_FILENAME = "feature_sparsity_forget.txt"
RETAIN_FILENAME = "feature_sparsity_retain.txt"

SPARSITIES_DIR = "results/sparsities"


def get_forget_retain_data(
    forget_corpora: str = "bio-forget-corpus",
    retain_corpora: str = "wikitext",
    min_len: int = 50,
    max_len: int = 2000,
    batch_size: int = 4,
) -> tuple[list[str], list[str]]:
    retain_dataset = []
    if retain_corpora == "wikitext":
        raw_retain = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        for x in raw_retain:
            if len(x["text"]) > min_len:
                retain_dataset.append(str(x["text"]))
    else:
        raise Exception("Unknown retain corpora")

    forget_dataset = []
    for num, line in enumerate(open(f"./evals/unlearning/data/{forget_corpora}.jsonl", "r")):
        if "bio-forget-corpus" in forget_corpora:
            try:
                
                raw_text = json.loads(line)["text"]
            except Exception as e:
                import pdb; pdb.set_trace()
            #     raw_text = line
        else:
            raw_text = line
        if len(raw_text) > min_len:
            forget_dataset.append(str(raw_text))

    return forget_dataset, retain_dataset


def get_shuffled_forget_retain_tokens(
    model: HookedTransformer,
    forget_corpora: str = "bio-forget-corpus",
    retain_corpora: str = "wikitext",
    batch_size: int = 2048,
    seq_len: int = 1024,
):
    """
    get shuffled forget tokens and retain tokens, with given batch size and sequence length
    note: wikitext has less than 2048 batches with seq_len=1024
    """
    forget_dataset, retain_dataset = get_forget_retain_data(forget_corpora, retain_corpora)

    print(len(forget_dataset), len(forget_dataset[0]))
    print(len(retain_dataset), len(retain_dataset[0]))

    shuffled_forget_dataset = random.sample(forget_dataset, min(batch_size, len(forget_dataset)))

    forget_tokens = dataset_utils.tokenize_and_concat_dataset(
        model.tokenizer, shuffled_forget_dataset, seq_len=seq_len
    ).to("cuda")
    retain_tokens = dataset_utils.tokenize_and_concat_dataset(
        model.tokenizer, retain_dataset, seq_len=seq_len
    ).to("cuda")

    print(forget_tokens.shape, retain_tokens.shape)
    shuffled_forget_tokens = forget_tokens[torch.randperm(forget_tokens.shape[0])]
    shuffled_retain_tokens = retain_tokens[torch.randperm(retain_tokens.shape[0])]

    return shuffled_forget_tokens[:batch_size], shuffled_retain_tokens[:batch_size]


def gather_residual_activations(model: HookedTransformer, target_layer: int, inputs):
    target_act = None

    def gather_target_act_hook(mod, inputs, outputs):
        nonlocal target_act  # make sure we can modify the target_act from the outer scope
        target_act = outputs[0]
        return outputs

    handle = model.model.layers[target_layer].register_forward_hook(gather_target_act_hook)
    _ = model.forward(inputs)
    handle.remove()
    return target_act


# def get_top_features(forget_score, retain_score, retain_threshold=0.01):
#     # criteria for selecting features: retain score < 0.01 and then sort by forget score
#     # high_retain_score_features = np.where(retain_score >= retain_threshold)[0]
#     # modified_forget_score = forget_score.copy()
#     # modified_forget_score[high_retain_score_features] = 0
#     # top_features = modified_forget_score.argsort()[::-1]
#     # # print(top_features[:20])

#     # n_non_zero_features = np.count_nonzero(modified_forget_score)
#     # top_features_non_zero = top_features[:n_non_zero_features]

#     # return top_features_non_zero

#     difference_score = forget_score - retain_score
    
#     # Sort features by difference score in descending order
#     top_features = difference_score.argsort()[::-1]
    
#     # Get only features with positive difference 
#     # (i.e., forget score > retain score)
#     positive_diff_mask = difference_score > 0
#     n_positive_features = np.count_nonzero(positive_diff_mask)
#     top_features_positive = top_features[:n_positive_features]
    
#     return top_features_positive


import numpy as np

def get_top_features(
    forget_score: np.ndarray,
    retain_score: np.ndarray,
    retain_threshold=0.01,
    method: str = "pareto",
    top_k: int = None,
    alpha: float = 1.0,
    eps: float = 1e-9
):
    """
    Select top features using one of several methods:
      1) 'difference':  score[i] = forget_score[i] - retain_score[i]
         - Filter to only those with score > 0 (i.e., forget_score > retain_score).
      
      2) 'ratio':       score[i] = forget_score[i] / (retain_score[i] + eps)
         - Filter to only those with score > 1.0 (i.e., forget_score > retain_score).
      
      3) 'rank':        Aggregate rank approach:
         - Rf[i] = rank of i in descending order of forget_score
         - Rr[i] = rank of i in ascending order of retain_score
         - score[i] = Rf[i] + Rr[i]
         - Lower is better. We'll invert the sort so that features with lowest score come first.
         - No hard cut, but you can pick top_k if desired.
         
      4) 'pareto':      Approximate Pareto frontier in the 2D space:
         - (forget_score[i], -retain_score[i]) 
           i.e., maximize forget_score, minimize retain_score
         - This returns only features on (or near) the Pareto frontier,
           then sorts them by forget_score descending for convenience.
         
    Args:
      forget_score (np.ndarray): 1D array of forget scores.
      retain_score (np.ndarray): 1D array of retain scores.
      method (str): Which selection method to use: 
                    {'difference', 'ratio', 'rank', 'pareto'}.
      top_k (int): How many top features to return. If None, return all.
      alpha (float): Optional weight for advanced difference-based methods, 
                     e.g. forget_score - alpha * retain_score. 
                     (Used only in 'difference' if desired.)
      eps (float): Small value to avoid division by zero (used in 'ratio').

    Returns:
      np.ndarray: Indices of the selected top features, sorted in the "best-first" order.
    """
    n_features = len(forget_score)

    if method == "difference":
        # Score = forget_score - alpha * retain_score (alpha defaults to 1.0)
        diff_score = forget_score - alpha * retain_score
        
        # Sort by descending difference
        sorted_idx = np.argsort(diff_score)[::-1]
        
        # Filter to keep only features with diff_score > 0
        positive_mask = diff_score > 0
        sorted_idx_pos = sorted_idx[positive_mask[sorted_idx]]
        
        if top_k is not None:
            return sorted_idx_pos[:top_k]
        else:
            return sorted_idx_pos

    elif method == "ratio":
        # Score = forget_score / (retain_score + eps)
        ratio_score = forget_score / (retain_score + eps)
        
        # Sort by descending ratio
        sorted_idx = np.argsort(ratio_score)[::-1]
        
        # Filter to keep only ratio > 1.0 (forget_score > retain_score)
        above_one_mask = ratio_score > 1.0
        sorted_idx_above_one = sorted_idx[above_one_mask[sorted_idx]]
        
        if top_k is not None:
            return sorted_idx_above_one[:top_k]
        else:
            return sorted_idx_above_one

    elif method == "rank":
        # Rf = rank in descending order of forget_score
        forget_desc_idx = np.argsort(forget_score)[::-1]  # highest forget_score first
        Rf = np.empty(n_features, dtype=int)
        Rf[forget_desc_idx] = np.arange(n_features)
        
        # Rr = rank in ascending order of retain_score
        retain_asc_idx = np.argsort(retain_score)  # lowest retain_score first
        Rr = np.empty(n_features, dtype=int)
        Rr[retain_asc_idx] = np.arange(n_features)
        
        # combined_rank = Rf[i] + Rr[i], smaller is "better"
        combined_rank = Rf + Rr
        
        # Sort by ascending combined rank
        # (Features with smallest combined rank come first)
        sorted_idx = np.argsort(combined_rank)
        
        if top_k is not None:
            return sorted_idx[:top_k]
        else:
            return sorted_idx

    elif method == "pareto":
        # Approximate the Pareto frontier:
        # We want to maximize forget_score and minimize retain_score.
        # We'll treat 'retain_score' with a negative sign to turn it into "higher is better."
        # Then find points on the frontier. A point (f_i, -r_i) is dominated if
        # there's another point (f_j, -r_j) with f_j >= f_i and -r_j >= -r_i
        # (meaning r_j <= r_i), with at least one strict inequality.

        # We'll store (forget_score[i], -retain_score[i]) for each feature i
        points = np.stack([forget_score, -retain_score], axis=1)
        indices = np.arange(n_features)
        
        # A simple approach: sort by one dimension (forget_score descending),
        # then do a linear scan to find non-dominated points by the other dimension.
        # This is an O(n log n) approach for 2D.
        
        # Sort features by forget_score descending, tie-break by retain_score ascending.
        sort_idx = np.lexsort((retain_score, -forget_score))  # primary: -forget, secondary: retain
        # This yields ascending in last key, so we do a trick: pass negative forget_score as the key.

        frontier = []
        # We'll track the best (lowest) retain_score encountered so far as we move.
        best_retain = np.inf
        
        for i in sort_idx[::-1]:
            # sort_idx[::-1] ensures we go from highest forget_score to lowest
            current_retain = retain_score[i]
            if current_retain < best_retain:
                # This point is on the Pareto frontier
                frontier.append(i)
                best_retain = current_retain
        
        # frontier now has indices on the Pareto frontier, but in descending forget_score order
        frontier = np.array(frontier)
        
        # Optionally, you might want to sort the frontier again by forget_score descending
        # or leave it as is, depending on your preference. We'll do it for clarity.
        frontier_sorted = frontier[np.argsort(forget_score[frontier])[::-1]]
        
        if top_k is not None:
            return frontier_sorted[:top_k]
        else:
            return frontier_sorted

    else:
        raise ValueError(f"Unknown method: {method}")



def check_existing_results(artifacts_folder: str, sae_name) -> bool:
    forget_path = os.path.join(artifacts_folder, sae_name, SPARSITIES_DIR, FORGET_FILENAME)
    retain_path = os.path.join(artifacts_folder, sae_name, SPARSITIES_DIR, RETAIN_FILENAME)
    return os.path.exists(forget_path) and os.path.exists(retain_path)


def calculate_sparsity(
    model: HookedTransformer, sae: SAE, forget_tokens, retain_tokens, batch_size: int
):
    feature_sparsity_forget = (
        get_feature_activation_sparsity(
            forget_tokens,
            model,
            sae,
            batch_size=batch_size,
            layer=sae.cfg.hook_layer,
            hook_name=sae.cfg.hook_name,
            mask_bos_pad_eos_tokens=True,
        )
        .cpu()
        .numpy()
    )
    feature_sparsity_retain = (
        get_feature_activation_sparsity(
            retain_tokens,
            model,
            sae,
            batch_size=batch_size,
            layer=sae.cfg.hook_layer,
            hook_name=sae.cfg.hook_name,
            mask_bos_pad_eos_tokens=True,
        )
        .cpu()
        .numpy()
    )
    return feature_sparsity_forget, feature_sparsity_retain


def save_results(
    artifacts_folder: str, sae_name: str, feature_sparsity_forget, feature_sparsity_retain
):
    output_dir = os.path.join(artifacts_folder, sae_name, SPARSITIES_DIR)
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, FORGET_FILENAME), feature_sparsity_forget, fmt="%f")
    np.savetxt(os.path.join(output_dir, RETAIN_FILENAME), feature_sparsity_retain, fmt="%f")


def load_sparsity_data(artifacts_folder: str, sae_name: str) -> tuple[np.ndarray, np.ndarray]:
    forget_sparsity = np.loadtxt(
        os.path.join(artifacts_folder, sae_name, SPARSITIES_DIR, FORGET_FILENAME), dtype=float
    )
    retain_sparsity = np.loadtxt(
        os.path.join(artifacts_folder, sae_name, SPARSITIES_DIR, RETAIN_FILENAME), dtype=float
    )
    return forget_sparsity, retain_sparsity


def save_feature_sparsity(
    model: HookedTransformer,
    sae: SAE,
    artifacts_folder: str,
    sae_name: str,
    dataset_size: int,
    seq_len: int,
    batch_size: int,
):
    if check_existing_results(artifacts_folder, sae_name):
        print(f"Sparsity calculation for {sae_name} is already done")
        return

    forget_tokens, retain_tokens = get_shuffled_forget_retain_tokens(
        model, batch_size=dataset_size, seq_len=seq_len
    )

    feature_sparsity_forget, feature_sparsity_retain = calculate_sparsity(
        model, sae, forget_tokens, retain_tokens, batch_size
    )

    save_results(artifacts_folder, sae_name, feature_sparsity_forget, feature_sparsity_retain)
