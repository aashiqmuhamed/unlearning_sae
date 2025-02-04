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


    for retain_threshold in config.retain_thresholds:
        # top_features_custom = get_top_features(
        #     forget_sparsity, retain_sparsity, retain_threshold=retain_threshold
        # )

        top_features_custom = get_top_features_percentile(
            forget_sparsity, retain_sparsity, ratio_percentile=retain_threshold,
        )

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
        }

        # import pdb; pdb.set_trace()

        save_metrics_dir = os.path.join(artifacts_folder, sae_name, "results/metrics")

        metrics_lst = calculate_metrics_list(
            model,
            (
                config.llm_batch_size * 2
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
        )

    return metrics_lst


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

    # calculate feature sparsity
    save_feature_sparsity(
        model,
        sae,
        artifacts_folder,
        sae_release_and_id,
        config.dataset_size,
        config.seq_len,
        config.llm_batch_size,
    )
    forget_sparsity, retain_sparsity = load_sparsity_data(artifacts_folder, sae_release_and_id)

    # do intervention and calculate eval metrics
    # activation_store = setup_activation_store(sae, model)
    activation_store = None
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
