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

# from sae_bench_utils.activation_collection_dynamic import get_feature_activation_sparsity
from sae_bench_utils.activation_collection_dynamic import get_feature_activation_sparsity, get_feature_activation_sparsity_old
import sae_bench_utils.dataset_utils as dataset_utils

FORGET_FILENAME = "feature_sparsity_forget.txt"
RETAIN_FILENAME = "feature_sparsity_retain.txt"

SPARSITIES_DIR = "results/sparsities"

def collate_fn(batch):
    texts = []
    for item in batch:
        #import pdb; pdb.set_trace()
        if 'choices' in item:
            task_description = "The following are multiple choice questions (with answers).\n\n"            
            prompts = []#
            prompt = task_description + f"{item['question']}\nA: {item['choices'][0]}\nB: {item['choices'][1]}\nC: {item['choices'][2]}\nD: {item['choices'][3]}\nAnswer:"
            texts.append(prompt)
        elif 'text' in item:
            texts.append(item['text'])
        else:
            texts.append(item)
    # tokenized = tokenizer(texts, add_special_tokens=True, padding=False, truncation=True)
    # tokenized['input_ids'] = torch.tensor(pad_sequences(tokenized['input_ids']))
    # tokenized['attention_mask'] = torch.tensor(pad_sequences(tokenized['attention_mask']))
    return texts
 
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
        #############################
        if forget_corpora != "bio-forget-corpus":
            concat_el = load_dataset("cais/wmdp", "wmdp-chem",split='test')
            list_of_str = collate_fn(concat_el)
            #retain_dataset = retain_dataset + list_of_str#[:50]

    elif retain_corpora == 'books':
        raw_retain = load_dataset("muse-bench/MUSE-Books", "raw",split='retain1')
        for x in raw_retain:
            if len(x["text"]) > min_len:
                retain_dataset.append(str(x["text"]))
    elif retain_corpora == 'news':
        raw_retain = load_dataset("muse-bench/MUSE-News", "raw",split='retain1')
        for x in raw_retain:
            if len(x["text"]) > min_len:
                retain_dataset.append(str(x["text"]))
    else:
        raise Exception("Unknown retain corpora")

    forget_dataset = []
    if "bio-forget-corpus" in forget_corpora:
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

    elif "cyber-forget-corpus" in forget_corpora:
        forget_dataset = load_dataset("cais/wmdp-corpora", "cyber-forget-corpus",split='train')
        forget_dataset = forget_dataset.filter(lambda x: len(x["text"])>min_len)
        forget_dataset.shuffle(seed=42)
        forget_dataset = forget_dataset["text"]
    elif forget_corpora == 'books':
        forget_dataset = load_dataset("muse-bench/MUSE-Books", "raw",split='forget')
        forget_dataset = forget_dataset.filter(lambda x: len(x["text"])>min_len)
        forget_dataset.shuffle(seed=42)
        forget_dataset = forget_dataset["text"]
    elif forget_corpora == 'news':
        forget_dataset = load_dataset("muse-bench/MUSE-News", "raw",split='forget')
        #forget_dataset = load_dataset("muse-bench/MUSE-News", "scal",split='forget_4')
        #forget_dataset = load_dataset("muse-bench/MUSE-News", "sust",split='forget_3')
        forget_dataset = forget_dataset.filter(lambda x: len(x["text"])>min_len)
        forget_dataset.shuffle(seed=42)
        forget_dataset = forget_dataset["text"]
    return forget_dataset, retain_dataset


def get_shuffled_forget_retain_tokens(
    model: HookedTransformer,
    forget_corpora: str = "bio-forget-corpus",
    retain_corpora: str = "wikitext",
    batch_size: int = 2048,
    seq_len: int = 1024,
    dataset_fraction: int = 100
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
    batch_size_cmn = min(int(batch_size*dataset_fraction/100),min(int(shuffled_forget_tokens.shape[0]*dataset_fraction/100),int(shuffled_retain_tokens.shape[0]*dataset_fraction/100)))
    
    print('tokens size: ',batch_size_cmn)#[:batch_size_cmn*2]
    return shuffled_forget_tokens[:batch_size_cmn], shuffled_retain_tokens[:batch_size_cmn]


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


def get_top_features(forget_score, retain_score, retain_threshold=0.01):
    # criteria for selecting features: retain score < 0.01 and then sort by forget score

    
    high_retain_score_features = np.where(retain_score >= retain_threshold)[0]
    modified_forget_score = forget_score.copy()
    modified_forget_score[high_retain_score_features] = 0
    top_features = modified_forget_score.argsort()[::-1]
    # print(top_features[:20])

    n_non_zero_features = np.count_nonzero(modified_forget_score)
    top_features_non_zero = top_features[:n_non_zero_features]

    return top_features_non_zero

def get_top_features_percentile(
    forget_activations: np.ndarray,
    retain_activations: np.ndarray,
    forget_percentile: float = 5,  # Select features in top 5% of forget importance
    retain_percentile: float = 100,  # Filter out features in top 50% of retain importance
    ratio_percentile: float = 90,    # Select features with forget/retain ratio in top 10%
    n_features_lst:list = [10,20,30],
    folder_name=None,

) -> np.ndarray:
    """
    Selects features using percentile-based criteria instead of fixed thresholds.
    
    The selection process considers three criteria:
    1. How important the feature is for forgetting (forget_activations)
    2. How important it is for retention (retain_activations)
    3. The ratio of forget/retain importance
    
    Args:
        forget_activations: Squared feature activations on forget dataset
        retain_activations: Squared feature activations on retain dataset
        forget_percentile: Minimum percentile for forget importance
        retain_percentile: Maximum percentile for retain importance
        ratio_percentile: Minimum percentile for forget/retain ratio
    """
    # Square activations to match FIM-style importance
    forget_score = forget_activations ** 2
    retain_score = retain_activations ** 2
    
    # Calculate importance ratio
    importance_ratio = forget_score / (retain_score + 1e-21)
    
    # Calculate percentile thresholds
    
    forget_threshold = np.percentile(forget_score, forget_percentile)
    retain_threshold = np.percentile(retain_score, retain_percentile)
    ratio_threshold = np.percentile(importance_ratio, ratio_percentile)
    # import pickle
    # with open('/home/jb/Documents/unlearning_sae/test_filtro.pkl', 'wb') as f:
    #     pickle.dump([forget_score,retain_score,importance_ratio],f)
    # print(SAVED)
    # Select features that meet all criteria:
    selected_features = np.where(
        (forget_score >= forget_threshold) &        # High forget importance
        (retain_score <= retain_threshold) &        # Low retain importance
        (importance_ratio >= ratio_threshold)       # High forget/retain ratio
    )[0]
    sel_ind_out = selected_features[np.argsort(-forget_score[selected_features])]
    forget_score_align = np.sort(-forget_score[selected_features])
    # plot_colored_points(retain_score,forget_score, sel_ind)
    # import pdb; pdb.set_trace()
    # Sort by forget importance
    # load activations
    import pickle
    with open(folder_name+'/act_fgt.pkl', 'rb') as f:
        act_fgt = pickle.load(f)
    with open(folder_name+'/act_ret.pkl', 'rb') as f:
        act_ret = pickle.load(f)
    percentile_95 = {}    

    for n in n_features_lst:
        sel_ind = sel_ind_out[:n]
        print(forget_score_align[:n])
        # distrib = []
        # for el in act_fgt:
        #     buff  = el[:,:,sel_ind]
        #     buff = buff>0
        #     buff2 = (buff.sum(axis=2)>0)
        #     #print(buff.shape,buff2.sum()/buff2.shape[1])
        #     distrib.append(buff.sum(axis=(1, 2)) / (buff.shape[1] * buff.shape[2]))
        # distrib = np.asarray(distrib)
        # print('percentili 1 e 5 fgt: ',np.percentile(distrib,1),np.percentile(distrib,5))    
        
        distrib = []

        for el in act_ret:
            buff  = el[:,:,sel_ind]
            buff = buff>0
            buff2 = (buff.sum(axis=2)>0)
            #print(buff2.sum()/buff2.shape[1])
            distrib.append(buff2.sum()/buff2.shape[1])
            #distrib.append(buff.sum(axis=(1, 2)) / (buff.shape[1] * buff.shape[2]))
        distrib = np.asarray(distrib)
        bootstrap_percentiles = []
        perc_val = 95
        for _ in range(1000):
            resample = np.random.choice(distrib, size=len(distrib), replace=True)
            bootstrap_percentiles.append(np.percentile(resample, perc_val))
        print('ESTIMATION: ',np.mean(bootstrap_percentiles))
        print('NUM:',n,' percentili 95 e 99 ret: ',np.percentile(distrib,perc_val),np.percentile(distrib,99))
        alpha = np.percentile(distrib,perc_val)

        percentile_95[str(n)] = alpha
    return sel_ind_out,percentile_95

import matplotlib.pyplot as plt
import numpy as np

def plot_colored_points(vec1, vec2, indices):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot all points in blue
    ax.scatter(vec1, vec2, s=1, color='blue', label='All Points')
    
    # Highlight points corresponding to indices in red
    ax.scatter(vec1[indices], vec2[indices], s=1, color='red', label='Highlighted Points')
    
    # Set axis to log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add labels and legend
    ax.set_xlabel('retain')
    ax.set_ylabel('forget')
    ax.legend()
    
    # Save the plot
    plt.savefig('/home/jb/Documents/unlearning_sae/v2/SAEBench/feature_sparsity.png')

def get_top_features_threshold(
    forget_grad_norm: np.ndarray,
    retain_grad_norm: np.ndarray,
    activations: np.ndarray,
    ratio_threshold: float = 1e3,
    num_act_rem: int = None
) -> np.ndarray:


    forget_score = forget_grad_norm.copy()
    forget_score[activations < 0.05] = 0
    
    importance_ratio = forget_score / (retain_grad_norm + 1e-21)
    importance_ratio[retain_grad_norm == 0] = 0
    
    ratio_mask = (importance_ratio < ratio_threshold)
    print(f'Features available: {(ratio_mask == False).sum()} over {ratio_mask.shape[0]}')
    
    forget_score[ratio_mask] = 0
    sorted_indices = np.argsort(forget_score)
    
    available_features = (ratio_mask == False).sum()
    if num_act_rem is not None:
        num_to_return = min(num_act_rem, available_features)
        return sorted_indices[-num_to_return:]
    
    return sorted_indices[forget_score[sorted_indices] > 0]
    

def get_top_features_ratio(
    forget_activations: np.ndarray,
    retain_activations: np.ndarray,
    retain_threshold: float = 0.01
) -> np.ndarray:
    """
    A modified version of get_top_features_percentile that mirrors the logic of get_top_features
    but sorts by ratio instead of forget score.
    
    This method:
    1. Squares the activations (to match FIM-style importance)
    2. Filters features based on retain threshold
    3. Sorts remaining features by their forget/retain ratio
    
    Args:
        forget_activations: Raw feature activations on forget dataset
        retain_activations: Raw feature activations on retain dataset
        retain_threshold: Maximum allowed retain importance (on squared activations)
    """
    # Square activations to match FIM-style importance
    forget_score = forget_activations ** 2
    retain_score = retain_activations ** 2
    
    # Calculate importance ratio for all features
    importance_ratio = forget_score / (retain_score + 1e-21)
    
    # Filter features based on retain threshold (just like first method)
    eligible_features = np.where(retain_score <= retain_threshold)[0]
    
    # Sort eligible features by their ratio (instead of forget score)
    sorted_features = eligible_features[np.argsort(-importance_ratio[eligible_features])]
    
    return sorted_features
    

def check_existing_results(artifacts_folder: str, sae_name) -> bool:
    forget_path = os.path.join(artifacts_folder, sae_name, SPARSITIES_DIR, FORGET_FILENAME)
    retain_path = os.path.join(artifacts_folder, sae_name, SPARSITIES_DIR, RETAIN_FILENAME)
    return os.path.exists(forget_path) and os.path.exists(retain_path)


def calculate_sparsity(
    model: HookedTransformer, sae: SAE, forget_tokens, retain_tokens, batch_size: int,folder_name=None):

    feature_sparsity_forget,act_fgt = get_feature_activation_sparsity(
            forget_tokens,
            model,
            sae,
            batch_size=batch_size,
            layer=sae.cfg.hook_layer,
            hook_name=sae.cfg.hook_name,
            mask_bos_pad_eos_tokens=True,)

    feature_sparsity_retain,act_ret = get_feature_activation_sparsity(
            retain_tokens,
            model,
            sae,
            batch_size=batch_size,
            layer=sae.cfg.hook_layer,
            hook_name=sae.cfg.hook_name,
            mask_bos_pad_eos_tokens=True,
        )
    #save the activations which are lists of numpy vectors use pickle
    #import pdb; pdb.set_trace()
    import pickle
    with open(folder_name+'/act_fgt.pkl', 'wb') as f:
        pickle.dump(act_fgt, f)
    with open(folder_name+'/act_ret.pkl', 'wb') as f:
        pickle.dump(act_ret, f)
    
    return feature_sparsity_forget.cpu().numpy(), feature_sparsity_retain.cpu().numpy()


def calculate_sparsity_old(
    model: HookedTransformer, sae: SAE, forget_tokens, retain_tokens, batch_size: int
):
    feature_sparsity_forget = (
        get_feature_activation_sparsity_old(
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
        get_feature_activation_sparsity_old(
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
    dataset_fraction:int,
    fgt_set: str,
    retain_set: str
):
    if check_existing_results(artifacts_folder, sae_name):
       print(f"Sparsity calculation for {sae_name} is already done")
       return
    forget_tokens, retain_tokens = get_shuffled_forget_retain_tokens(
                                                                    model, 
                                                                    batch_size=dataset_size,
                                                                    seq_len=seq_len,
                                                                    dataset_fraction=dataset_fraction,
                                                                    forget_corpora=fgt_set,
                                                                    retain_corpora=retain_set)

    folder_name = os.path.join(artifacts_folder, sae_name, SPARSITIES_DIR)
    os.makedirs(folder_name, exist_ok=True)

    feature_sparsity_forget, feature_sparsity_retain = calculate_sparsity(
        model, sae, forget_tokens, retain_tokens, batch_size,folder_name=folder_name
    )

    save_results(artifacts_folder, sae_name, feature_sparsity_forget, feature_sparsity_retain)
