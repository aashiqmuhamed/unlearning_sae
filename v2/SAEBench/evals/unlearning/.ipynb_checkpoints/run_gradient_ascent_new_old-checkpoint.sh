#!/bin/bash
# Base configuration
MODEL="google/gemma-2-2b-it"
OUTPUT_DIR="/data/datasets/wmdp_test/baselines/gradient_ascent_new_"
METHOD="gradient_ascent"

# Create output directory
mkdir -p $OUTPUT_DIR

# Parameters to test
alpha_values=(
    # "0"
    # "1"
    "10"
    "100"
    "300"
    "500"
    
    
)

max_steps=(
    "10"
    "50"
    "100"
    "150"
)

# Log file for experiment tracking
log_file="$OUTPUT_DIR/experiment_log.txt"

# Initialize log file
{
    echo "Starting gradient ascent experiments at $(date)"
    echo "Model: $MODEL"
    echo "Method: $METHOD"
    echo "-------------------"
} > $log_file

# Run experiments for each combination
for alpha in "${alpha_values[@]}"; do
    for steps in "${max_steps[@]}"; do
        echo "Running experiment with alpha=$alpha, max_steps=$steps"
        {
            echo "Starting experiment at $(date)"
            echo "Alpha: $alpha"
            echo "Max Steps: $steps"
        } >> $log_file
        
        experiment_dir="$OUTPUT_DIR/alpha${alpha}_steps${steps}"
        mkdir -p $experiment_dir
        
        python train_baselines.py \
            --method $METHOD \
            --model_name_or_path $MODEL \
            --retain_corpora "wikitext" \
            --forget_corpora "bio-forget-corpus" \
            --alpha "$alpha" \
            --output_dir $experiment_dir \
            --batch_size 8 \
            --max_num_batches "$steps" \
            --layer_id 7 \
            --layer_ids "all" \
            --param_ids "all" \
            --gradient_accumulation_steps 4 \
            --lr 2e-5 \
            2>&1 | tee -a "$experiment_dir/run.log"
            
        echo "Completed experiment at $(date)" >> $log_file
        echo "-------------------" >> $log_file
    done
done

echo "All experiments completed at $(date)" >> $log_file

# Create a summary of experiments
{
    echo "Gradient Ascent Experiments Summary"
    echo "=================================="
    echo "Model: $MODEL"
    echo "Date: $(date)"
    echo "Total experiments: $((${#alpha_values[@]} * ${#max_steps[@]}))"
    echo ""
    echo "Alpha values tested:"
    printf '%s\n' "${alpha_values[@]/#/- }"
    echo ""
    echo "Max steps tested:"
    printf '%s\n' "${max_steps[@]/#/- }"
    echo ""
    echo "Fixed parameters:"
    echo "- Batch size: 16"
    echo "- Learning rate: 1e-4"
    echo "- Gradient accumulation steps: 8"
    echo "- Layer IDs: all"
} > "$OUTPUT_DIR/summary.txt"

echo "Experiments completed. Results are in $OUTPUT_DIR"