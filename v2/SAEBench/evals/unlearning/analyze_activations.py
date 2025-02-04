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
import torch.nn.functional as F

from sae_bench_utils.sae_selection_utils import get_saes_from_regex
from sae_bench_utils.general_utils import setup_environment, load_and_format_sae
import sae_bench_utils.activation_collection as activation_collection
from evals.unlearning.utils.feature_activation import get_shuffled_forget_retain_tokens

import matplotlib.pyplot as plt
# import seaborn as sns

ACTIVATIONS_DIR = "results/activations"

def calculate_activations(
        model: HookedTransformer,
        tokens: torch.Tensor,
        hook_layer: int,
        hook_name: str,
        batch_size: int
) -> torch.Tensor:
    """Calculate residual activations, excluding special tokens."""
    device = next(model.parameters()).device
    all_activations = []

    for i in tqdm(range(0, tokens.shape[0], batch_size)):
        tokens_batch = tokens[i:i + batch_size]
        _, cache = model.run_with_cache(
            tokens_batch,
            stop_at_layer=hook_layer + 1,
            names_filter=hook_name
        )
        resid = cache[hook_name]  # [batch, seq_len, hidden_dim]

        # Create mask for non-special tokens
        valid_tokens = (
            (tokens_batch != model.tokenizer.pad_token_id) &
            (tokens_batch != model.tokenizer.bos_token_id) &
            (tokens_batch != model.tokenizer.eos_token_id)
        )

        # Filter and collect only valid token activations
        valid_activations = resid[valid_tokens]
        all_activations.append(valid_activations)

    return torch.cat(all_activations, dim=0)  # [n_valid_tokens, hidden_dim]

# def compare_and_visualize(base_acts: torch.Tensor, ft_acts: torch.Tensor, save_dir: str):
#     """Compare activations and create visualizations."""
#     # Calculate metrics
#     cosine_similarities = F.cosine_similarity(base_acts, ft_acts, dim=1)
#     magnitude_ratios = torch.norm(ft_acts, dim=1) / torch.norm(base_acts, dim=1)

#     # Create visualizations
#     plt.style.use('seaborn')
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

#     # Plot cosine similarity distribution
#     sns.histplot(cosine_similarities.cpu(), ax=ax1, stat='density')
#     ax1.set_title('Distribution of Cosine Similarities')
#     ax1.set_xlabel('Cosine Similarity')
#     ax1.set_ylabel('Density')

#     # Plot magnitude ratio distribution
#     sns.histplot(magnitude_ratios.cpu(), ax=ax2, stat='density')
#     ax2.set_title('Distribution of Magnitude Ratios')
#     ax2.set_xlabel('Magnitude Ratio (FT/Base)')
#     ax2.set_ylabel('Density')

#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, 'activation_distributions.png'), dpi=300, bbox_inches='tight')
#     plt.close()

#     # Save numerical data
#     np.savez(
#         os.path.join(save_dir, "activation_metrics.npz"),
#         cosine_similarities=cosine_similarities.cpu().numpy(),
#         magnitude_ratios=magnitude_ratios.cpu().numpy(),
#         mean_cosine=cosine_similarities.mean().item(),
#         std_cosine=cosine_similarities.std().item(),
#         mean_magnitude_ratio=magnitude_ratios.mean().item(),
#         std_magnitude_ratio=magnitude_ratios.std().item()
#     )


def compare_and_visualize(base_acts: torch.Tensor, ft_acts: torch.Tensor, save_dir: str):
    """Compare activations and create visualizations using matplotlib."""
    # Calculate metrics and convert to float32
    cosine_similarities = F.cosine_similarity(base_acts, ft_acts, dim=1).to(torch.float32)
    magnitude_ratios = (torch.norm(ft_acts, dim=1) / torch.norm(base_acts, dim=1)).to(torch.float32)

    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot cosine similarity distribution
    n_bins = 50
    ax1.hist(cosine_similarities.cpu().numpy(), bins=n_bins, density=True, 
             color='cornflowerblue', edgecolor='black', alpha=0.7)
    ax1.set_title('Distribution of Cosine Similarities', fontsize=12, pad=10)
    ax1.set_xlabel('Cosine Similarity', fontsize=10)
    ax1.set_ylabel('Density', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot magnitude ratio distribution
    ax2.hist(magnitude_ratios.cpu().numpy(), bins=n_bins, density=True,
             color='cornflowerblue', edgecolor='black', alpha=0.7)
    ax2.set_title('Distribution of Magnitude Ratios', fontsize=12, pad=10)
    ax2.set_xlabel('Magnitude Ratio (FT/Base)', fontsize=10)
    ax2.set_ylabel('Density', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'activation_distributions.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

    # Save numerical data (converting to float32 before saving)
    np.savez(
        os.path.join(save_dir, "activation_metrics.npz"),
        cosine_similarities=cosine_similarities.cpu().numpy(),
        magnitude_ratios=magnitude_ratios.cpu().numpy(),
        mean_cosine=cosine_similarities.mean().item(),
        std_cosine=cosine_similarities.std().item(),
        mean_magnitude_ratio=magnitude_ratios.mean().item(),
        std_magnitude_ratio=magnitude_ratios.std().item()
    )

def evaluate_models(
        base_model: HookedTransformer,
        finetuned_model: HookedTransformer,
        sae: SAE,
        tokens: torch.Tensor,
        save_dir: str,
        batch_size: int = 32
):
    """Compare base and finetuned model activations."""
    print("\nCalculating activation metrics...")
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        base_acts = calculate_activations(
            base_model, 
            tokens, 
            sae.cfg.hook_layer,
            sae.cfg.hook_name,
            batch_size
        )
        ft_acts = calculate_activations(
            finetuned_model,
            tokens,
            sae.cfg.hook_layer,
            sae.cfg.hook_name,
            batch_size
        )
        
        compare_and_visualize(base_acts, ft_acts, save_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--finetuned_model", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--sae_regex_pattern", type=str, required=True)
    parser.add_argument("--sae_block_pattern", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="eval_results/activation_analysis")
    args = parser.parse_args()

    device = setup_environment()
    start_time = time.time()

    # Get SAEs
    selected_saes = get_saes_from_regex(args.sae_regex_pattern, args.sae_block_pattern)
    assert len(selected_saes) > 0, "No SAEs selected"

    # Load base model
    base_model = HookedTransformer.from_pretrained_no_processing(
        args.base_model,
        device=device,
        dtype=activation_collection.LLM_NAME_TO_DTYPE[args.base_model]
    )
    
    # Load finetuned model
    finetuned = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16,
        device_map='cpu'
    )
    finetuned_model = HookedTransformer.from_pretrained_no_processing(
        model_name=args.finetuned_model,
        device=device,
        dtype=activation_collection.LLM_NAME_TO_DTYPE[args.finetuned_model],
        hf_model=finetuned
    )

    # Generate tokens once to use for both models
    tokens = get_shuffled_forget_retain_tokens(
        base_model,
        batch_size=2048,
        seq_len=1024,
    )[0]  # Using just forget_tokens for simplicity

    # Process each SAE
    for sae_release, sae_id in tqdm(selected_saes, desc="Processing SAEs"):
        _, sae, _ = load_and_format_sae(sae_release, sae_id, device)
        sae = sae.to(device=device)

        print(f"\nProcessing SAE: {sae_release}_{sae_id}")
        sae_dir = os.path.join(args.output_dir, f"{sae_release}_{sae_id}")
        
        evaluate_models(
            base_model=base_model,
            finetuned_model=finetuned_model,
            sae=sae,
            tokens=tokens,
            save_dir=sae_dir
        )

    end_time = time.time()
    print(f"\nEvaluation completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()