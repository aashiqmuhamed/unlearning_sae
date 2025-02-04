#!/bin/bash

# Base configuration
MODEL="google/gemma-2-2b-it"
OUTPUT_DIR="/data/datasets/wmdp_test/baselines/relearning_$(date +%Y%m%d)"
METHOD="gradient_ascent"

# Create output directory
mkdir -p $OUTPUT_DIR

# Alpha values to test (from conservative to aggressive unlearning)
alpha_values=(
    "1"
)

# Log file for experiment tracking
log_file="$OUTPUT_DIR/experiment_log.txt"
echo "Starting gradient ascent experiments at $(date)" > $log_file
echo "Model: $MODEL" >> $log_file
echo "Method: $METHOD" >> $log_file
echo "-------------------" >> $log_file

# Run experiments for each alpha value
for alpha in "${alpha_values[@]}"; do
    echo "Running experiment with alpha=$alpha"
    echo "Starting experiment with alpha=$alpha at $(date)" >> $log_file

    experiment_dir="$OUTPUT_DIR/alpha_${alpha}"
    mkdir -p $experiment_dir

    python relearning.py \
        --method $METHOD \
        --model_name_or_path $MODEL \
        --retain_corpora "wikitext" \
        --forget_corpora "bio-forget-corpus" \
        --alpha "$alpha" \
        --output_dir $experiment_dir \
        --batch_size 16 \
        --max_num_batches 1000 \
        --layer_id 7 \
        --layer_ids "all" \
        --param_ids "all" \
        --gradient_accumulation_steps 8 \
        --lr 5e-5 \
        2>&1 | tee -a "$experiment_dir/run.log"

    echo "Completed experiment with alpha=$alpha at $(date)" >> $log_file
    echo "-------------------" >> $log_file
done

echo "All experiments completed at $(date)" >> $log_file

# Create a summary of experiments
echo "Generating experiment summary..."
{
    echo "Gradient Ascent Experiments Summary"
    echo "=================================="
    echo "Model: $MODEL"
    echo "Date: $(date)"
    echo "Number of experiments: ${#alpha_values[@]}"
    echo ""
    echo "Alpha values tested:"
    for alpha in "${alpha_values[@]}"; do
        echo "- $alpha"
    done
} > "$OUTPUT_DIR/summary.txt"

echo "Experiments completed. Results are in $OUTPUT_DIR"
