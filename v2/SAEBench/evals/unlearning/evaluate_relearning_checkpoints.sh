#!/bin/bash

# Base paths
BASE_MODEL_PATH="/data/datasets/wmdp_test/baselines/relearning_20250204/alpha_1"
BASE_ARTIFACTS_PATH="/data/aashiq_muhamed/unlearning/SAEBench"
DUMMY_PATH="${BASE_ARTIFACTS_PATH}/dummy"
BASE_OUTPUT_PATH="/data/aashiq_muhamed/unlearning/SAEBench/eval_results"

# Number of epochs to process (0-9)
NUM_EPOCHS=10

for ((epoch=0; epoch<NUM_EPOCHS; epoch++)); do
    echo "Processing epoch ${epoch}"
    
    # Create paths for this epoch
    EPOCH_ARTIFACTS_PATH="${BASE_ARTIFACTS_PATH}/artifacts_relearning_epoch_${epoch}"
    EPOCH_OUTPUT_PATH="${BASE_OUTPUT_PATH}/unlearning_dynamic_bs1_epoch_${epoch}"
    EPOCH_MODEL_PATH="${BASE_MODEL_PATH}/epoch_${epoch}"
    
    # Create output directory if it doesn't exist
    mkdir -p "${EPOCH_OUTPUT_PATH}"
    
    # Copy dummy folder with new name
    # echo "Copying dummy folder for epoch ${epoch}"
    # cp -r "${DUMMY_PATH}" "${EPOCH_ARTIFACTS_PATH}"
    
    # Run the evaluation script
    echo "Running evaluation for epoch ${epoch}"
    python evals/unlearning/main_dynamic_relearning.py \
        --sae_regex_pattern "gemma-scope-2b-pt-res" \
        --sae_block_pattern "layer_3/width_16k/average_l0_142" \
        --model_name gemma-2-2b-it \
        --llm_batch_size 1 \
        --llm_dtype float32 \
        --model_path "${EPOCH_MODEL_PATH}" \
        --artifacts_path "${EPOCH_ARTIFACTS_PATH}" \
        --output_folder "${EPOCH_OUTPUT_PATH}"
    
    # Check if the command was successful
    if [ $? -ne 0 ]; then
        echo "Error occurred while processing epoch ${epoch}"
        exit 1
    fi
    
    echo "Completed processing epoch ${epoch}"
    echo "----------------------------------------"
done

echo "All epochs processed successfully"