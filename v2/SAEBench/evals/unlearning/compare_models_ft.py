"""
Script to evaluate feature activations and MCQ performance for base and finetuned models.
Handles models sequentially due to memory constraints.
"""

import os
import time
import torch
import numpy as np
import argparse
from tqdm import tqdm
import gc
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM
from sae_lens import SAE

from evals.unlearning.eval_config import UnlearningEvalConfig
from sae_bench_utils.sae_selection_utils import get_saes_from_regex
from sae_bench_utils.general_utils import setup_environment, load_and_format_sae
import sae_bench_utils.activation_collection as activation_collection
from evals.unlearning.utils.feature_activation import get_shuffled_forget_retain_tokens
from evals.unlearning.utils.metrics import calculate_MCQ_metrics, save_target_question_ids

ACTIVATIONS_DIR = "results/activations"
MCQ_RESULTS_DIR = "results/mcq_performance"
DATASETS = ["wmdp-bio",
            "high_school_us_history",
            "college_computer_science",
            "high_school_geography",
            "human_aging"]


def calculate_average_activations(
        model: HookedTransformer,
        sae: SAE,
        tokens: torch.Tensor,
        batch_size: int
) -> np.ndarray:
    """Calculate average activations using the same pattern as get_feature_activation_sparsity."""
    device = sae.device
    running_sum = torch.zeros(sae.W_dec.shape[0], dtype=torch.float32, device=device)
    total_tokens = 0

    for i in tqdm(range(0, tokens.shape[0], batch_size)):
        tokens_batch = tokens[i:i + batch_size]
        _, cache = model.run_with_cache(
            tokens_batch,
            stop_at_layer=sae.cfg.hook_layer + 1,
            names_filter=sae.cfg.hook_name
        )
        resid = cache[sae.cfg.hook_name]
        acts = sae.encode(resid)

        # Apply masking like in get_feature_activation_sparsity
        attn_mask = (
                (tokens_batch != model.tokenizer.pad_token_id) &
                (tokens_batch != model.tokenizer.bos_token_id) &
                (tokens_batch != model.tokenizer.eos_token_id)
        ).to(device=acts.device)

        acts = acts * attn_mask[:, :, None]
        total_tokens += attn_mask.sum().item()
        running_sum += acts.sum(dim=(0, 1))

    return (running_sum / total_tokens).cpu().numpy()


def evaluate_and_save_model(
        model_name,
        model,
        sae: SAE,
        forget_tokens: torch.Tensor,
        retain_tokens: torch.Tensor,
        sae_dir: str,
        artifacts_folder: str,
        device: str,
        filtered_questions: dict,
        batch_size: int = 32
):
    """Evaluate a single model and save results."""

    # Calculate feature activations
    print(f"\nCalculating activations for {model_name}")

    with torch.no_grad():
        avg_acts_forget = calculate_average_activations(
            model, sae, forget_tokens, batch_size=batch_size
        )
        avg_acts_retain = calculate_average_activations(
            model, sae, retain_tokens, batch_size=batch_size
        )

    # Save activations
    save_dir = os.path.join(sae_dir, model_name, ACTIVATIONS_DIR)
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "feature_acts_forget.npy"), avg_acts_forget)
    np.save(os.path.join(save_dir, "feature_acts_retain.npy"), avg_acts_retain)

    # Evaluate on each dataset
    all_results = {}
    for dataset_name in DATASETS:
        question_subset = filtered_questions.get(dataset_name)
        results = calculate_MCQ_metrics(
            model=model,
            mcq_batch_size=batch_size,
            artifacts_folder=artifacts_folder,
            dataset_name=dataset_name,
            question_subset=question_subset,
            permutations=[[0, 1, 2, 3]],
            target_metric="correct",
            verbose=True
        )
        all_results[dataset_name] = results

    # Save MCQ results
    mcq_dir = os.path.join(sae_dir, model_name, MCQ_RESULTS_DIR)
    os.makedirs(mcq_dir, exist_ok=True)
    np.save(os.path.join(mcq_dir, "mcq_results.npy"), all_results)

    # Clear model from memory
    del model
    torch.cuda.empty_cache()
    gc.collect()


def load_filtered_questions(artifacts_folder: str) -> dict:
    """Load filtered question indices for all datasets."""
    filtered_questions = {}
    for dataset_name in DATASETS:
        # Convert dataset name format for MMLU datasets
        file_dataset_name = (
            f'mmlu-{dataset_name.replace("_", "-")}'
            if dataset_name != "wmdp-bio"
            else dataset_name
        )
        file_path = os.path.join(
            artifacts_folder,
            "data/question_ids/all",
            f"{file_dataset_name}_correct.csv"
        )
        if os.path.exists(file_path):
            filtered_questions[dataset_name] = np.genfromtxt(file_path, dtype=int)
    return filtered_questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--finetuned_model", type=str, required=True)
    parser.add_argument("--sae_regex_pattern", type=str, required=True)
    parser.add_argument("--sae_block_pattern", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="eval_results/activation_mcq")
    parser.add_argument("--model_path", type=str, required=True, help="Path to directory containing finetuned model weights")
    args = parser.parse_args()

    device = setup_environment()
    start_time = time.time()

    # Get SAEs
    selected_saes = get_saes_from_regex(args.sae_regex_pattern, args.sae_block_pattern)
    assert len(selected_saes) > 0, "No SAEs selected"

    # First load base model to get tokens and filter questions
    print("\nGetting tokens and filtering questions with base model...")
    # base_model = HookedTransformer.from_pretrained_no_processing(
    #     args.base_model,
    #     device=device,
    #     dtype=activation_collection.LLM_NAME_TO_DTYPE[args.base_model]
    # )

    # forget_tokens, retain_tokens = get_shuffled_forget_retain_tokens(
    #     base_model,
    #     batch_size=2048,
    #     seq_len=1024,
    # )

    # Filter questions using base model for all datasets
    # for dataset_name in DATASETS:
    #     save_target_question_ids(
    #         base_model,
    #         mcq_batch_size=32,
    #         artifacts_folder=args.output_dir,
    #         dataset_name=dataset_name
    #     )

    # Clear base model
    # del base_model
    # torch.cuda.empty_cache()
    # gc.collect()

    # Load filtered questions


    
    filtered_questions = load_filtered_questions(args.output_dir)
    # filtered_questions = load_filtered_questions("/data/aashiq_muhamed/unlearning/SAEBench/eval_results/activation_mcq")
     

    # Process each SAE
    for sae_release, sae_id in tqdm(selected_saes, desc="Processing SAEs"):
        _, sae, _ = load_and_format_sae(sae_release, sae_id, device)
        sae = sae.to(device=device)

        print(f"\nProcessing SAE: {sae_release}_{sae_id}")
        sae_dir = os.path.join(args.output_dir, f"{sae_release}_{sae_id}")

        # Evaluate both models on filtered questions
        for model_name in [args.finetuned_model]: #

            # First load finetuned weights using AutoModelForCausalLM
            finetuned = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map='cpu')

            # import pdb; pdb.set_trace()
        

            # model = HookedTransformer.from_pretrained_no_processing(
            #                             model_name,
            #                             device=device,
            #                             dtype= activation_collection.LLM_NAME_TO_DTYPE[model_name],
            #                             cache_dir="/data/datasets/wmdp_test/model_dir"
            #                         )

            model = HookedTransformer.from_pretrained_no_processing(model_name=model_name, device=device,  dtype= activation_collection.LLM_NAME_TO_DTYPE[model_name], hf_model=finetuned)



            # # Copy and verify parameters
            # finetuned_params = dict(finetuned.named_parameters())
            # model_params = dict(model.named_parameters())

            # if set(finetuned_params.keys()) != set(model_params.keys()):
            #     import pdb; pdb.set_trace()
            #     raise ValueError("Parameter names don't match between models")

            # Copy parameters
            # for name, param in finetuned.named_parameters():
            #     model.state_dict()[name].copy_(param)

            # del finetuned
            # torch.cuda.empty_cache()
                
            forget_tokens, retain_tokens = get_shuffled_forget_retain_tokens(
                model,
                batch_size=2048,
                seq_len=1024,
            )
            
            evaluate_and_save_model(
                model_name=model_name,
                model=model,
                sae=sae,
                forget_tokens=forget_tokens,
                retain_tokens=retain_tokens,
                sae_dir=sae_dir,
                artifacts_folder=args.output_dir,
                device=device,
                filtered_questions=filtered_questions
            )

    end_time = time.time()
    print(f"\nEvaluation completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()

