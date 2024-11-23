# SAELens Feature Ablation Tutorials

This repository contains tutorials for ablating SAE features and evaluating their impact on various benchmark datasets. Each script focuses on specific datasets for evaluation, with **Wikitext** as the **retain set** and **WMDP-Bio** as the **forget set**.

---

## Contents

### **Feature Ablation Scripts**

1. **`feature_ablate_1.py`**  
   - Focus: Evaluation on **MMLU**.  
   - Ablates SAE features and assesses their influence on multi-task language understanding.

2. **`feature_ablate_2.py`**  
   - Focus: Evaluation on **WMDP**.  
   - Explores the impact of SAE feature removal on medical domain processing tasks.

3. **`feature_ablate_3.py`**  
   - Focus: Evaluation on **OpenWebText (OWT)**.  
   - Assesses how ablation affects general language modeling performance.

---

## Evaluation Setup

- **Retain Set**: **Wikitext**  
  The features retained are evaluated on this dataset to ensure core language modeling capabilities are preserved.

- **Forget Set**: **WMDP-Bio**  
  The features ablated are evaluated on this dataset to understand the forgetting dynamics in medical domain processing.

---

## TODO Tasks
### To Be Implemented:
- [ ] **Efficient Evaluation with Patched-in SAEs**  
  - Use `lm-eval` for faster and more scalable evaluations.  
  - Support multi-GPU setups to improve computational efficiency.

- [ ] **Pareto Curves**  
  - Sweep over the number of ablated features and plot Pareto curves to visualize the trade-offs between feature ablation and task performance.

- [ ] **RMU Baseline**  
  - Implement the **Random Masking Unit (RMU)** baseline to compare ablation effectiveness.

- [ ] **Custom Model Evaluation via `lm-eval`**  
  - Configure `lm-eval` to:  
    - Evaluate on specific dataset subsets.  
    - Create flexible configurations for custom evaluations.

- [ ] **Additional Baselines**  
  - Implement more baselines to broaden comparative insights.

---

## Contributing
We welcome contributions! If you would like to implement any of the TODO tasks, propose improvements, or address issues, feel free to open a pull request or submit an issue.

---

Explore the repository and experiment with the scripts to uncover new insights into SAE feature importance!
