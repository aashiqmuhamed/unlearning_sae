python ~/unlearning/SAEBench/evals/unlearning/compare_models_base.py \
    --base_model gemma-2-2b-it \
    --finetuned_model gemma-2-2b-it \
    --sae_regex_pattern "sae_bench_gemma-2-2b_topk_width-2pow14_date-1109" \
    --sae_block_pattern "blocks.5.hook_resid_post__trainer_2" \
    --output_dir eval_results/activation_mcq_revised/alpha_1


# /data/datasets/wmdp_test/baselines/gradient_ascent_20241218/alpha_1/gradient_ascent/