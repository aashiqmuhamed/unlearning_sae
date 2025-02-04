#!/bin/bash

# Base directories
RMU_BASE_DIR="/data/datasets/wmdp_test/baselines/rmu_20250109"
OUTPUT_BASE_DIR="/data/aashiq_muhamed/unlearning/SAEBench/eval_results/rmu"
TEMP_DIR="${OUTPUT_BASE_DIR}/temp"

# Function to process a single model directory
process_model() {
    local steer_dir=$1
    local rmu_dir="${steer_dir}/rmu"
    local model_name=$(basename "$steer_dir")
    local output_dir="${OUTPUT_BASE_DIR}/${model_name}"
    
    echo "Processing model: $model_name"
    echo "RMU directory: $rmu_dir"
    
    # Verify RMU directory exists
    if [ ! -d "$rmu_dir" ]; then
        echo "Warning: RMU directory not found in ${steer_dir}"
        return
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Copy required base files
    cp -r "${TEMP_DIR}/base_model" "$output_dir/"
    cp -r "${TEMP_DIR}/data" "$output_dir/"
    
    # Run the Python script
    python ~/unlearning/SAEBench/evals/unlearning/compare_models_ft.py \
        --base_model gemma-2-2b-it \
        --finetuned_model gemma-2-2b-it \
        --model_path "$rmu_dir" \
        --sae_regex_pattern "sae_bench_gemma-2-2b_topk_width-2pow14_date-1109" \
        --sae_block_pattern "blocks.5.hook_resid_post__trainer_2" \
        --output_dir "$output_dir"
        
    echo "Completed processing $model_name"
    echo "----------------------------------------"
}

# Main execution
echo "Starting RMU results processing"
echo "----------------------------------------"

# Find all steer* directories and process them
for steer_dir in "$RMU_BASE_DIR"/steer*; do
    if [ -d "$steer_dir" ]; then
        process_model "$steer_dir"
    fi
done

echo "All processing complete!"