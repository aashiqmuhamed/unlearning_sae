#!/bin/bash

# Script to analyze activation changes between base and finetuned models
# Usage: ./analyze_activations.sh

# Set path variables
BASE_MODEL="gemma-2-2b-it"
FINETUNED_MODEL="gemma-2-2b-it"
MODEL_PATH="/data/datasets/wmdp_test/baselines/gradient_ascent_20250105/alpha_1/gradient_ascent/"
SAE_PATTERN="sae_bench_gemma-2-2b_topk_width-2pow14_date-1109"
BLOCK_PATTERN="blocks.5.hook_resid_post__trainer_2"
OUTPUT_DIR="/data/aashiq_muhamed/unlearning/SAEBench/evals/unlearning/eval_results/activation_analysis_ft_test/"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Run the activation analysis
python ~/unlearning/SAEBench/evals/unlearning/analyze_activations.py \
    --base_model ${BASE_MODEL} \
    --finetuned_model ${FINETUNED_MODEL} \
    --model_path ${MODEL_PATH} \
    --sae_regex_pattern ${SAE_PATTERN} \
    --sae_block_pattern ${BLOCK_PATTERN} \
    --output_dir ${OUTPUT_DIR}

echo "Analysis completed. Results saved to ${OUTPUT_DIR}"