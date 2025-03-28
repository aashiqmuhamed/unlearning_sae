import os
import numpy as np
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from evals.unlearning.utils.feature_activation_dynamic import (
    get_top_features,
    get_top_features_ratio,
    get_top_features_percentile,
    get_top_features_threshold,
    load_sparsity_data,
    save_feature_sparsity,
    get_shuffled_forget_retain_tokens,
    calculate_sparsity_old,

)
from evals.unlearning.utils.metrics_dynamic import calculate_metrics_list
from evals.unlearning.eval_config_dynamic import UnlearningEvalConfig

def run_metrics_calculation(
    model: HookedTransformer,
    sae: SAE,
    activation_store,
    forget_sparsity: np.ndarray,
    retain_sparsity: np.ndarray,
    artifacts_folder: str,
    sae_name: str,
    config: UnlearningEvalConfig,
    force_rerun: bool,
):
    dataset_names = config.dataset_names


    # Aashiq only for get_top_features_threshold
    
    # forget_tokens, retain_tokens = get_shuffled_forget_retain_tokens(
    # model, batch_size=config.dataset_size, seq_len=config.seq_len)

    # feature_sparsity_forget, _ = calculate_sparsity_old(
    #     model, sae, forget_tokens, retain_tokens, config.llm_batch_size
    # )

    ####################################################################
    folder_name = os.path.join(artifacts_folder, sae_name, "results","sparsities")
    for retain_threshold in config.retain_thresholds:
        # top_features_custom = get_top_features(
        #     forget_sparsity, retain_sparsity, retain_threshold=retain_threshold
        # )

        top_features_custom,threshold_inf = get_top_features_percentile(
            forget_sparsity, retain_sparsity, ratio_percentile=retain_threshold,folder_name=folder_name,n_features_lst=config.n_features_list
        )
        
        # #########################################################################################################

        # import sys
        # sys.path.append('/home/jb/Documents/unlearning_sae/v2/SAEBench/evals/unlearning/utils/')
        # from jb_exp import get_feature_activation,get_valid_forget_retain_dataloader

        # forget = 'bio-forget-corpus' if 'wmdp-bio' in dataset_names else 'cyber-forget-corpus'
        # dataloader_fgt, dataloader_retain =get_valid_forget_retain_dataloader(forget_corpora=forget, retain_corpora='wikitext')
        # # print('FGT')
        # # get_feature_activation(data_loader=dataloader_fgt,model=model,sae=sae,layer= sae.cfg.hook_layer,hook_name=sae.cfg.hook_name,features=top_features_custom[:config.n_features_list[0]])
        # import sae_bench_utils.dataset_utils as dataset_utils
        # dataloader_retain = dataset_utils.tokenize_and_concat_dataset(model.tokenizer, dataloader_retain, seq_len=512).to("cuda")
        # print('RETAIN')
        # get_feature_activation(data_loader=dataloader_retain,model=model,sae=sae,layer= sae.cfg.hook_layer,hook_name=sae.cfg.hook_name,features=top_features_custom[:config.n_features_list[0]])
        
        #bio 0-shot
        # top_features_custom = [12382, 9722, 343, 373, 11, 15969, 12117, 5877, 968, 622, 5231, 10546, 12037, 6150, 5704, 14747, 8786, 10933, 140, 13527]
        # threshold_inf={}
        # threshold_inf['20'] = 0.3
        #cyber 0-shot
        # top_features_custom = [15331, 2060, 15286, 11015, 364, 4836, 2905, 10931, 11716, 16160, 6309, 10543, 11513, 1803, 12681, 11520, 11323, 10415, 3943, 4686, 15519, 9669, 16341, 15390, 12716, 8825, 2162, 7359, 5525, 10222, 7885, 2005, 33, 4555, 4022, 6502, 11960, 4795, 2280, 10817, 14267, 7691, 9241, 5928, 5768, 11121]
        # threshold_inf={}
        # threshold_inf['30'] = 0.25
        #threshold_inf['20'] = 0.000
        ########################################################################################################
        # top_features_custom = get_top_features_ratio(
        #     forget_sparsity, retain_sparsity, retain_threshold=retain_threshold
        # )


        # top_features_custom = get_top_features_threshold(
        #     forget_sparsity, retain_sparsity, activations=feature_sparsity_forget, ratio_threshold=retain_threshold,
        # )

        main_ablate_params = {
            "intervention_method": config.intervention_method,
            
        }

        n_features_lst = config.n_features_list
        multipliers = config.multipliers

        sweep = {
            "features_to_ablate": [np.array(top_features_custom[:n]) for n in n_features_lst],
            "multiplier": multipliers,
            'activation_threshold': threshold_inf,
        }

        # import pdb; pdb.set_trace()

        save_metrics_dir = os.path.join(artifacts_folder, sae_name, "results/metrics")

        metrics_lst = calculate_metrics_list(
            model,
            (
                config.llm_batch_size #* 2
            ),  # multiple choice questions are shorter, so we can afford a larger batch size
            sae,
            main_ablate_params,
            sweep,
            artifacts_folder,
            force_rerun,
            dataset_names,
            n_batch_loss_added=config.n_batch_loss_added,
            activation_store=activation_store,
            target_metric=config.target_metric,
            save_metrics=config.save_metrics,
            save_metrics_dir=save_metrics_dir,
            retain_threshold=retain_threshold,
            seed=config.random_seed,
        )

    return metrics_lst

def compute_params_SAE(
    model: HookedTransformer,
    sae: SAE,
    activation_store,
    forget_sparsity: np.ndarray,
    retain_sparsity: np.ndarray,
    artifacts_folder: str,
    sae_name: str,
    config: UnlearningEvalConfig,
    force_rerun: bool,
):
    dataset_names = config.dataset_names

    ####################################################################
    folder_name = os.path.join(artifacts_folder, sae_name, "results","sparsities")
    for retain_threshold in config.retain_thresholds:

        top_features_custom,threshold_inf = get_top_features_percentile(
            forget_sparsity, retain_sparsity, ratio_percentile=retain_threshold,folder_name=folder_name,n_features_lst=config.n_features_list
        )
        
        main_ablate_params = {
            "intervention_method": config.intervention_method,
            
        }

        n_features_lst = config.n_features_list
        multipliers = config.multipliers

        sweep = {
            'threshold':retain_threshold,
            "features_to_ablate": [np.array(top_features_custom[:n]) for n in n_features_lst],
            "multiplier": multipliers,
            'activation_threshold': threshold_inf,
        }
        print(sweep)



def run_eval_single_sae(
    model: HookedTransformer,
    sae: SAE,
    config: UnlearningEvalConfig,
    artifacts_folder: str,
    sae_release_and_id: str,
    force_rerun: bool,
):
    """sae_release_and_id: str is the name used when saving data for this SAE. This data will be reused at various points in the evaluation."""

    os.makedirs(artifacts_folder, exist_ok=True)

    torch.set_grad_enabled(False)

    #calculate feature sparsity
    save_feature_sparsity(
        model,
        sae,
        artifacts_folder,
        sae_release_and_id,
        config.dataset_size,
        config.seq_len,
        config.llm_batch_size,
        config.dataset_fraction,
        fgt_set=config.fgt_set,
        retain_set=config.retain_set
    )
    forget_sparsity, retain_sparsity = load_sparsity_data(artifacts_folder, sae_release_and_id)
    #forget_sparsity, retain_sparsity = np.asarray([0]),np.asarray([0])
    # do intervention and calculate eval metrics
    # activation_store = setup_activation_store(sae, model)
    activation_store = None
    if config.fgt_set=='books' or config.fgt_set=='news':
        compute_params_SAE(  
            model,
            sae,
            activation_store,
            forget_sparsity,
            retain_sparsity,
            artifacts_folder,
            sae_release_and_id,
            config,
            force_rerun,
        )
    else:
        results = run_metrics_calculation(
            model,
            sae,
            activation_store,
            forget_sparsity,
            retain_sparsity,
            artifacts_folder,
            sae_release_and_id,
            config,
            force_rerun,
        )

        return results
