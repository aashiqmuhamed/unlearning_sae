import numpy as np

def analyze_activation_metrics(npz_path):
    """
    Reads and displays activation comparison metrics from a saved NPZ file.
    
    Args:
        npz_path: Path to the NPZ file containing activation metrics
    """
    metrics = np.load(npz_path)
    
    print("Activation Distribution Analysis")
    print("-" * 50)
    
    print("\nCosine Similarity Statistics:")
    print(f"Mean: {metrics['mean_cosine']:.4f}")
    print(f"Standard Deviation: {metrics['std_cosine']:.4f}")
    
    print("\nMagnitude Ratio Statistics:")
    print(f"Mean: {metrics['mean_magnitude_ratio']:.4f}")
    print(f"Standard Deviation: {metrics['std_magnitude_ratio']:.4f}")
    
    # Additional percentile analysis for both metrics
    cos_sims = metrics['cosine_similarities']
    mag_ratios = metrics['magnitude_ratios']
    
    print("\nCosine Similarity Percentiles:")
    print(f"25th percentile: {np.percentile(cos_sims, 25):.4f}")
    print(f"50th percentile: {np.percentile(cos_sims, 50):.4f}")
    print(f"75th percentile: {np.percentile(cos_sims, 75):.4f}")
    
    print("\nMagnitude Ratio Percentiles:")
    print(f"25th percentile: {np.percentile(mag_ratios, 25):.4f}")
    print(f"50th percentile: {np.percentile(mag_ratios, 50):.4f}")
    print(f"75th percentile: {np.percentile(mag_ratios, 75):.4f}")

if __name__ == "__main__":
    # Example usage
    npz_path = "activation_metrics.npz"  # Replace with your actual path
    analyze_activation_metrics(npz_path)