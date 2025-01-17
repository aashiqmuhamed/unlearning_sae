import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import glob

mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

def pad_sequences(sequences, padding_value=0):
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = [seq + [padding_value] * (max_len - len(seq)) for seq in sequences]
    return padded_sequences

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

def plot_importances(norm_fgt,norm_rtn,indexes,filename):
    #import pdb; pdb.set_trace()
    color = np.zeros_like(norm_fgt)
    color[indexes] = 1
    c_list = ['blue', 'red','green','black']
    c_plot = [c_list[int(color[i])] for i in range(len(color))]

    fig,ax = plt.subplots(figsize=(10, 6),ncols=1,nrows=1)
    ax.scatter(norm_rtn,norm_fgt,s=1,c=c_plot)

    for e in [ax]:
        min_val = min(e.get_xlim()[0], e.get_ylim()[0])
        max_val = max(e.get_xlim()[1], e.get_ylim()[1])
    # Plot the diagonal line
        delta = 5*1e2
        e.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Diagonal')
        e.plot([min_val, max_val], [min_val*delta, max_val*delta], color='red', linestyle='--', label='Diagonal')
        e.set_ylabel('Fgt.')
        e.set_xlabel('Ret.')
        e.set_yscale('log')
        e.set_xscale('log')
    plt.savefig(filename)
    plt.close()

def plot_results(input_folder):
    #collect df
    files = glob.glob(input_folder+'/*.csv')
    print('files csv collectd ',files)
    df = None
    for file in files:
        if df is None:
            df = pd.read_csv(file)
        else:
            df = pd.concat([df, pd.read_csv(file)], ignore_index=True)
    print(df.head())
    # filter for th and #of feat
    th_list = df.th_ratio.unique()
    num_feat_list = df.num_activations.unique()
    fig,ax = plt.subplots(figsize=(10, 6),ncols=1,nrows=1)
    for th in th_list:
        for num_feat in num_feat_list:
            df_th = df[df.th_ratio == th]
            df_th = df_th[df_th.num_activations == num_feat]

            #plot
            
            ax.plot(df_th['acc_wmdp'],df_th['acc_mmlu'],label=f'th={th},#_feat={num_feat}',marker='o')
            # Add values at the top of each plot point
            # for i, value in enumerate(df_th['clamp_val'].values):
            #    plt.text(df_th['acc_wmdp'].values[i],df_th['acc_mmlu'].values[i]-df_th['acc_mmlu'].values[i]*0.02, str(value), ha='center', va='bottom')
    #add legend
    ax.legend()
    plt.ylim(0.6,1)
    plt.xlim(0,1)
    #save
    print('plot_saved in: ',f'{input_folder}acc_wmdp_vs_acc_mmlu.png')
    plt.savefig(f'{input_folder}/acc_wmdp_vs_acc_mmlu.png')
    plt.close()

import torch 
import transformers
import random
import einops
from typing import Optional

def tokenize_and_concat_dataset(
    tokenizer: transformers.AutoTokenizer,
    dataset: list[str],
    seq_len: int,
    add_bos: bool = True,
    max_tokens: Optional[int] = None,
) -> torch.Tensor:
    full_text = tokenizer.eos_token.join(dataset)

    # divide into chunks to speed up tokenization
    num_chunks = 20
    chunk_length = (len(full_text) - 1) // num_chunks + 1
    chunks = [full_text[i * chunk_length : (i + 1) * chunk_length] for i in range(num_chunks)]
    tokens = tokenizer(chunks, return_tensors="pt", padding=True)["input_ids"].flatten()

    # remove pad token
    tokens = tokens[tokens != tokenizer.pad_token_id]

    if max_tokens is not None:
        tokens = tokens[: max_tokens + seq_len + 1]

    num_tokens = len(tokens)
    num_batches = num_tokens // seq_len

    # drop last batch if not full
    tokens = tokens[: num_batches * seq_len]
    tokens = einops.rearrange(tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len)

    if add_bos:
        tokens[:, 0] = tokenizer.bos_token_id
    return tokens

def get_shuffled_forget_retain_tokens(
    tokenizer: transformers.AutoTokenizer,
    forget_dataset: list,
    retain_dataset: list,
    batch_size: int = 2048,
    seq_len: int = 1024,
):
    """
    get shuffled forget tokens and retain tokens, with given batch size and sequence length
    note: wikitext has less than 2048 batches with seq_len=1024
    """
    #import pdb; pdb.set_trace()
    print(len(forget_dataset), len(forget_dataset[0]))
    print(len(retain_dataset), len(retain_dataset[0]))
    
    shuffled_forget_dataset = random.sample(forget_dataset, min(batch_size, len(forget_dataset)))

    forget_tokens = tokenize_and_concat_dataset(
        tokenizer, shuffled_forget_dataset, seq_len=seq_len
    ).to("cuda")
    retain_tokens = tokenize_and_concat_dataset(
        tokenizer, retain_dataset, seq_len=seq_len
    ).to("cuda")

    print(forget_tokens.shape, retain_tokens.shape)
    shuffled_forget_tokens = forget_tokens[torch.randperm(forget_tokens.shape[0])]
    shuffled_retain_tokens = retain_tokens[torch.randperm(retain_tokens.shape[0])]

    return shuffled_forget_tokens[:batch_size], shuffled_retain_tokens[:batch_size]


import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

def plot_sentence_length_distribution(
    dataset,
    dname:str,
    text_column: str='text',
    num_bins: int = 50,
    max_length: Optional[int] = None,
    title: Optional[str] = None
):
    """
    Plot the distribution of sentence lengths in a dataset.

    Args:
    dataset_name (str): Name of the dataset to load.
    text_column (str): Name of the column containing the text data.
    split (str): Which split of the dataset to use (e.g., "train", "test").
    num_bins (int): Number of bins for the histogram.
    max_length (int, optional): Maximum sentence length to include in the plot.
    title (str, optional): Title for the plot. If None, a default title is used.
    """
    

    # Calculate sentence lengths
    sentence_lengths = [len(sentence) for sentence in dataset[text_column]]

    # Filter lengths if max_length is specified
    if max_length:
        sentence_lengths = [length for length in sentence_lengths if length <= max_length]

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.hist(sentence_lengths, bins=num_bins, edgecolor='black')
    plt.xlabel('Sentence Length (words)')
    plt.ylabel('Frequency')
    
    if title:
        plt.title(title)
    else:
        plt.title(f'Distribution of Sentence Lengths of {dname}')

    plt.grid(True, linestyle='--', alpha=0.7)

    # Calculate and display statistics
    mean_length = np.mean(sentence_lengths)
    median_length = np.median(sentence_lengths)
    plt.axvline(mean_length, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_length:.2f}')
    plt.axvline(median_length, color='g', linestyle='dashed', linewidth=2, label=f'Median: {median_length:.2f}')

    plt.legend()
    plt.savefig(f'sentence_length_distribution_{dname}.png')

    # Print additional statistics
    print(f"Mean sentence length: {mean_length:.2f}")
    print(f"Median sentence length: {median_length:.2f}")
    print(f"Min sentence length: {min(sentence_lengths)}")
    print(f"Max sentence length: {max(sentence_lengths)}")

# Example usage:
# plot_sentence_length_distribution("wikitext", "text", split="train", max_length=200)

def clean_importances(importances):
    for key in ['W_enc', 'W_dec', 'b_enc', 'b_dec','threshold']:
        if key != 'W_dec':
            importances.pop(key)
        else:
            importances[key] = importances[key].detach().cpu()
    return importances
if __name__ == '__main__':
    plot_results(input_folder='/home/jb/Documents/unlearning_sae/results_FIM_SAE/SAEBench/')  