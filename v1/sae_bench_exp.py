from opts import get_args
from forget_w_SAE import forget_w_SAE_CausalLM
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch
import numpy as np
import pandas as pd

from datasets import load_dataset,concatenate_datasets    
from metrics_SAEbench import get_metrics,save_target_question_ids
from utils import plot_results,plot_sentence_length_distribution
import os
from pathlib import Path

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

dataset_names = ["wmdp-bio",
                 "human_aging",
                 "high_school_us_history",
                 "college_computer_science",
                 "high_school_geography",
            ]

#metric_params = ['correct', 'correct-iff-question',  'correct-no-tricks']
target_metric = 'correct'
split = 'all'
mcq_batch_size=8
artifacts_folder = os.path.join(args.root_folder, "artifacts", 'unlearning', args.model_name)

input_model = None
input_tokenizer = None
#BASELINE from sae Bench
if args.run_baseline:
    baseline_metrics = {}
    input_model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                device_map=args.device,
                                                torch_dtype=torch.bfloat16)

    input_tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    for dataset_name in [x for x in dataset_names]:
        # Ensure that target question ids exist
        save_target_question_ids(input_model,input_tokenizer, mcq_batch_size, artifacts_folder, dataset_name)

        metric_param = {"target_metric": target_metric, "verbose": False}

        baseline_metric = get_metrics(input_model,input_tokenizer, mcq_batch_size, artifacts_folder, 
                                      res_folder_name="baseline",
                                      dataset_name = dataset_name,
                                      metric_param=metric_param,
                                      split=split)

        baseline_metrics[dataset_name] = baseline_metric
    
#CALL Unlearning Method class    
if args.run_unlearn:

    unlearn_method = forget_w_SAE_CausalLM(model_name = args.model_name,
                                    sae_name = args.sae_name,
                                    sae_id=args.sae_id,
                                    retain_dset=retain_set,
                                    fgt_dset=fgt_set.take(2000),#to do add in the parser the quantity to use 
                                    use_error_term=args.use_error_term,
                                    device=args.device,
                                    th_ratio=args.th_ratio,
                                    batch_size=args.batch_size,
                                    num_activations=args.num_activations,
                                    input_model=input_model,
                                    input_tokenizer=input_tokenizer)
    #LOOP over params
    all_results = []

    for config_num_act in range(len(unlearn_method.num_activations)):
        for clamp_val in args.clamp_values:#,
            print(f'CURRENT CONFIG ----> # of feat rem. {unlearn_method.num_activations[config_num_act]}, clamp val {clamp_val}, th_ratio {args.th_ratio}')
            #chamge buffer_size since batch for this exp is set to mcq_batch_size
            unlearn_method.model.flag_batch_activation = torch.tensor([False for _ in range(mcq_batch_size)], dtype=torch.bool)
            
            hook_added = unlearn_method.get_model_with_sae(config_num_act,clamp_val)
            
            #funct for metrics
            metrics = {}
            for dataset_name in [x for x in dataset_names if x != "loss_added"]:
                print('Analysing dataset: ', dataset_name)
                metric_param = {"target_metric": target_metric, "verbose": False}
                
                metric = get_metrics(unlearn_method.model,unlearn_method.tokenizer, mcq_batch_size, artifacts_folder, 
                                     res_folder_name = f"FIM_SAE_{args.sae_id.replace('/','_')}_clamp_{clamp_val}_num_act_{unlearn_method.num_activations[config_num_act]}_th_ratio_{args.th_ratio}",
                                     dataset_name = dataset_name, 
                                     metric_param = metric_param,
                                     split=split,
                                     device=args.device,
                                     recompute=args.recompute)
                
                metrics[dataset_name] = metric
            
            #collect results
            acc_wmdp = metrics['wmdp-bio']['mean_correct']
            mmlu = np.asarray([metrics[i]['mean_correct'] for i in metrics.keys() if i != 'wmdp-bio'])
            
            all_results.append([unlearn_method.th_ratio,
                                clamp_val,
                                unlearn_method.num_activations[config_num_act],
                                acc_wmdp,
                                mmlu.mean(),
                                mmlu.std()])
            #
            print('Current results: ',all_results[-1])
            hook_added.remove()

    df = pd.DataFrame(all_results,columns=['th_ratio','clamp_val','num_activations','acc_wmdp','acc_mmlu','std_mmlu'])
    #save df to csv
    Path(os.path.join(args.root_folder,'results_FIM_SAE',args.sae_id.replace('/','_'),'SAEBench')).mkdir(parents=True, exist_ok=True)
    file_to_save = os.path.join(args.root_folder,'results_FIM_SAE',args.sae_id.replace("/","_"),'SAEBench',f'results_th_ratio_{args.th_ratio}.csv')
    df.to_csv(file_to_save,index=False)
    plot_results(input_folder=os.path.join(args.root_folder,'results_FIM_SAE',args.sae_id.replace("/","_"),'SAEBench'))