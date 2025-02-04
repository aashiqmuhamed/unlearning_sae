#!/bin/bash

# Base configuration
MODEL="google/gemma-2-2b-it"
OUTPUT_DIR="/data/datasets/wmdp_test/baselines/rmu_$(date +%Y%m%d)"
METHOD="rmu"

# Create output directory
mkdir -p $OUTPUT_DIR

# Hyperparameter combinations to test
steering_coeffs=(
    "100"
    "200"
    "400"
)

alpha_values=(
    "100"
    "300"
    "500"
)

# Monitoring layer IDs to test (currently just 3, but extensible)
monitoring_layers=(
    3
)

# Log file for experiment tracking
log_file="$OUTPUT_DIR/experiment_log.txt"

# Initialize log file
{
    echo "Starting RMU experiments at $(date)"
    echo "Model: $MODEL"
    echo "Method: $METHOD"
    echo "-------------------"
} > $log_file

# Run experiments for each combination
for steering in "${steering_coeffs[@]}"; do
    for alpha in "${alpha_values[@]}"; do
        for monitor_layer in "${monitoring_layers[@]}"; do
            echo "Running experiment with steering=$steering, alpha=$alpha, monitoring_layer=$monitor_layer"
            {
                echo "Starting experiment at $(date)"
                echo "Steering coefficients: $steering"
                echo "Alpha: $alpha"
                echo "Monitoring Layer: $monitor_layer"
            } >> $log_file

            experiment_dir="$OUTPUT_DIR/steer${steering}_alpha${alpha}_monitorLayer${monitor_layer}"
            mkdir -p $experiment_dir

            python train_baselines.py \
                --method $METHOD \
                --model_name_or_path $MODEL \
                --retain_corpora "wikitext" \
                --forget_corpora "bio-forget-corpus" \
                --steering_coeffs "$steering" \
                --alpha "$alpha" \
                --layer_id "$monitor_layer" \
                --output_dir $experiment_dir \
                --batch_size 8 \
                --max_num_batches 400 \
                --layer_ids "3,7,11" \
                --param_ids "all" \
                --gradient_accumulation_steps 1 \
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
    echo "RMU Experiments Summary"
    echo "======================"
    echo "Model: $MODEL"
    echo "Date: $(date)"
    echo "Total experiments: $((${#steering_coeffs[@]} * ${#alpha_values[@]} * ${#monitoring_layers[@]}))"
    echo ""
    echo "Steering coefficients tested:"
    printf '%s\n' "${steering_coeffs[@]/#/- }"
    echo ""
    echo "Alpha values tested:"
    printf '%s\n' "${alpha_values[@]/#/- }"
    echo ""
    echo "Monitoring layers tested:"
    printf '%s\n' "${monitoring_layers[@]/#/- }"
    echo ""
    echo "Fixed parameters:"
    echo "- Layer IDs for updating: 3,7,11"
} > "$OUTPUT_DIR/summary.txt"

echo "Experiments completed. Results are in $OUTPUT_DIR"