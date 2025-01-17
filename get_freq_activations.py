from forget_w_SAE import forget_w_SAE_CausalLM
from datasets import load_dataset,concatenate_datasets  
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
from opts import get_args
import os
import torch

import matplotlib as mpl
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

args = get_args()
#DATASETS
if args.fgt_dset == 'wmdp_forget_corpora':    
    fgt_set = load_dataset('json',data_files=args.root_folder+"data_wmdp_forget_corpora/bio_remove_dataset.jsonl")['train']
    fgt_set = fgt_set.filter(lambda x: len(x["text"])>50)
    fgt_set.shuffle(seed=42)
    #plot_sentence_length_distribution(fgt_set, "wmdp_forget_corpora")
else:
    raise NotImplementedError

if args.retain_dset == 'wikitext':
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    #filter wikitext for sentences <50
    retain_set = wikitext.filter(lambda x: len(x["text"])>50)
    retain_set.shuffle(seed=42)
    #plot_sentence_length_distribution(retain_set, "wikitext")
else:
    raise NotImplementedError

unlearn_method = forget_w_SAE_CausalLM(model_name = args.model_name,
                                    sae_name = args.sae_name,
                                    sae_id=args.sae_id,
                                    retain_dset=retain_set,
                                    fgt_dset=fgt_set.take(2000),#to do add in the parser the quantity to use 
                                    use_error_term=args.use_error_term,
                                    device=args.device,
                                    th_ratio=500,
                                    batch_size=args.batch_size,
                                    num_activations=[50])

dataloader_fgt = DataLoader(unlearn_method.fgt_dset, batch_size=unlearn_method.batch_size, shuffle=True,)# collate_fn=collate_fn)
dataloader_retain = DataLoader(unlearn_method.retain_dset, batch_size=unlearn_method.batch_size, shuffle=True,)
unlearn_method.select_activations_FIM()

freq_rtn = unlearn_method.compute_activation_freq(dataloader_retain,num_act_rem=50,flag_tokenizer=False)
freq_fgt = unlearn_method.compute_activation_freq(dataloader_fgt,num_act_rem=50,flag_tokenizer=False)
#save the freq activations
filename = os.path.join('importances', f"{unlearn_method.model_name.replace('/','_')}_{unlearn_method.sae_id.replace('/','_')}_frequences.pkl")
with open(filename, 'wb') as f:
    pkl.dump({'retain':freq_rtn,'forget':freq_fgt}, f)