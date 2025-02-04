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
    "1"
    "5"
    "10"
)

max_steps=(
    "10"
    "50"
    "100"
    "500"
)

layer_configs=(
    "all"
    "3,7,11"
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
        for layer_ids in "${layer_configs[@]}"; do
            # Keep original directory name for "all", add suffix for specific layers
            experiment_dir="$OUTPUT_DIR/alpha${alpha}_steps${steps}"
            if [[ "$layer_ids" != "all" ]]; then
                experiment_dir="${experiment_dir}_layers${layer_ids//,/-}"
            fi
            
            echo "Running experiment with alpha=$alpha, max_steps=$steps, layer_ids=$layer_ids"
            {
                echo "Starting experiment at $(date)"
                echo "Alpha: $alpha"
                echo "Max Steps: $steps"
                echo "Layer IDs: $layer_ids"
            } >> $log_file
            
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
                --layer_ids "$layer_ids" \
                --param_ids "all" \
                --gradient_accumulation_steps 4 \
                --lr 2e-5 \
                2>&1 | tee -a "$experiment_dir/run.log"
                
            echo "Completed experiment at $(date)" >> $log_file
            echo "-------------------" >> $log_file
        done
    done
done

echo "All experiments completed at $(date)" >> $log_file

# Create a summary of experiments
{
    echo "Gradient Ascent Experiments Summary"
    echo "=================================="
    echo "Model: $MODEL"
    echo "Date: $(date)"
    echo "Total experiments: $((${#alpha_values[@]} * ${#max_steps[@]} * ${#layer_configs[@]}))"
    echo ""
    echo "Alpha values tested:"
    printf '%s\n' "${alpha_values[@]/#/- }"
    echo ""
    echo "Max steps tested:"
    printf '%s\n' "${max_steps[@]/#/- }"
    echo ""
    echo "Layer configurations tested:"
    printf '%s\n' "${layer_configs[@]/#/- }"
    echo ""
    echo "Fixed parameters:"
    echo "- Batch size: 8"
    echo "- Learning rate: 2e-5"
    echo "- Gradient accumulation steps: 4"
    echo "- Layer ID: 7"
} > "$OUTPUT_DIR/summary.txt"

echo "Experiments completed. Results are in $OUTPUT_DIR"