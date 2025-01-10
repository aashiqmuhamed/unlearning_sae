from opts import get_args
from forget_w_SAE import forget_w_SAE_CausalLM

from datasets import load_dataset,concatenate_datasets    
from metrics_SAEbench import get_baseline_metrics,save_target_question_ids
import os
args = get_args()

#DATASETS

if args.fgt_dset == 'wmdp_forget_corpora':    
    fgt_set = load_dataset('json',data_files=args.root_folder+"data_wmdp_forget_corpora/bio_remove_dataset.jsonl")['train']
    fgt_set.shuffle(seed=42)
else:
    raise NotImplementedError

if args.retain_dset == 'wikitext':
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    #filter wikitext for sentences <50
    retain_set = wikitext.filter(lambda x: len(x["text"])>50)
    retain_set.shuffle(seed=42)
else:
    raise NotImplementedError

#CALL Unlearning Method class

unlearn_method = forget_w_SAE_CausalLM(model_name = args.model_name,
                                    sae_name = args.sae_name,
                                    sae_id=args.sae_id,
                                    retain_dset=retain_set,
                                    fgt_dset=fgt_set.take(2000),#to do add in the parser the quantity to use 
                                    use_error_term=args.use_error_term,
                                    device=args.device,
                                    th_ratio=args.th_ratio,
                                    batch_size=args.batch_size,
                                    num_activations=args.num_activations)

dataset_names = ["wmdp-bio",
                 "high_school_us_history",
                 "college_computer_science",
                 "high_school_geography",
                 "human_aging"]

metric_params = ['correct', 'correct-iff-question',  'correct-no-tricks']
target_metric = 'correct'
#BASELINE from sae Bench
baseline_metrics = {}
artifacts_folder = os.path.join(args.root_folder, "artifacts", 'unlearning', args.model_name)
split = 'all'

for dataset_name in [x for x in dataset_names if x != "loss_added"]:
    # Ensure that target question ids exist
    save_target_question_ids(unlearn_method.model, mcq_batch_size=1024, artifacts_folder, dataset_name)

    metric_param = {"target_metric": target_metric, "verbose": False}

    # metrics[dataset_name] = dataset_metrics

    baseline_metric = get_baseline_metrics(unlearn_method.model, mcq_batch_size=1024, artifacts_folder, dataset_name, metric_param, split=split)

    baseline_metrics[dataset_name] = baseline_metric



# #LOOP
# all_results = []
# for config_num_act in range(len(unlearn_method.num_activations)):
#     for clamp_val in args.clamp_val:#,
#         print(f'CURRENT CONFIG ----> # of feat rem. {unlearn_method.num_activations[config_num_act]}, clamp val {clamp_val}, th_ratio {args.th_ratio}')
#         hook_added = unlearn_method.get_model_with_sae(config_num_act,clamp_val)
#         #funct for metrics

#         #
#         hook_added.remove()
