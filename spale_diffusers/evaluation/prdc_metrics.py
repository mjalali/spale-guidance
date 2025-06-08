import numpy as np
import pandas as pd
from prdc import compute_prdc # Ensure 'pip install prdc'

def calculate_f1(precision, recall):
    """Calculates F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def extract_and_sample_features(real_df, fake_df, n_samples=None, seed=1234):
    """
    Extracts features and samples an equal number from real and fake DataFrames.
    
    Args:
        real_df (pd.DataFrame): DataFrame with real features (must have 'features' column).
        fake_df (pd.DataFrame): DataFrame with fake features (must have 'features' column).
        n_samples (int, optional): Number of samples to draw from each. 
                                   If None, uses min(len(real_df), len(fake_df)).
        seed (int): Random seed for sampling.

    Returns:
        tuple: (real_features_array, fake_features_array)
    """
    if "features" not in real_df.columns or "features" not in fake_df.columns:
        raise ValueError("Both DataFrames must contain a 'features' column.")

    if n_samples is None:
        n_to_sample = min(len(real_df), len(fake_df))
    else:
        n_to_sample = min(len(real_df), len(fake_df), n_samples)

    if n_to_sample == 0:
        print("Warning: One or both feature sets are empty, or n_samples is 0. Cannot sample features.")
        return np.array([]), np.array([])

    print(f"Number of real samples available: {len(real_df)}")
    print(f"Number of fake samples available: {len(fake_df)}")
    print(f"Will use {n_to_sample} samples from each for this PRDC run.")
    
    np.random.seed(seed)
    real_df_sampled = real_df.sample(n=n_to_sample, random_state=seed)
    # Use a different seed for fake sampling to ensure independence if datasets overlap in index
    fake_df_sampled = fake_df.sample(n=n_to_sample, random_state=seed + 1) 
    
    try:
        real_features = np.array(real_df_sampled["features"].tolist())
        fake_features = np.array(fake_df_sampled["features"].tolist())
    except Exception as e:
        raise ValueError(f"Error converting 'features' column to numpy array. Ensure features are lists/arrays of numbers. Original error: {e}")

    # Handle cases where features might be nested (e.g., list of lists if not processed correctly before)
    if real_features.ndim > 2 or (real_features.ndim == 1 and isinstance(real_features[0], (list, np.ndarray))):
        real_features = np.stack(real_features.tolist())
    if fake_features.ndim > 2 or (fake_features.ndim == 1 and isinstance(fake_features[0], (list, np.ndarray))):
        fake_features = np.stack(fake_features.tolist())

    print(f"Real features shape after sampling: {real_features.shape}")
    print(f"Fake features shape after sampling: {fake_features.shape}")
    
    if real_features.shape[0] == 0 or fake_features.shape[0] == 0:
        print("Warning: Resulting feature arrays are empty after sampling.")
        return np.array([]), np.array([])
    if real_features.shape[1] != fake_features.shape[1]:
        raise ValueError(
            f"Feature dimensions mismatch: Real features have {real_features.shape[1]} dims, "
            f"Fake features have {fake_features.shape[1]} dims."
        )
            
    return real_features, fake_features

def compute_metrics_with_f1(real_features, fake_features, nearest_k=5):
    """Computes PRDC metrics and adds F1 score."""
    if real_features.shape[0] < nearest_k or fake_features.shape[0] < nearest_k:
        print(f"Warning: Not enough samples for nearest_k={nearest_k}. "
              f"Real samples: {real_features.shape[0]}, Fake samples: {fake_features.shape[0]}. "
              "PRDC metrics might be unreliable or fail.")
        # Return NaNs or default error values if computation is not possible
        return {'precision': np.nan, 'recall': np.nan, 'density': np.nan, 'coverage': np.nan, 'f1': np.nan}

    metrics = compute_prdc(
        real_features=real_features,
        fake_features=fake_features,
        nearest_k=nearest_k
    )
    metrics['f1'] = calculate_f1(metrics['precision'], metrics['recall'])
    return metrics

def compute_prdc_multiple_runs(real_df, fake_df, n_runs=5, nearest_k=5, n_samples_per_run=None):
    """
    Computes PRDC metrics over multiple runs with different random samples
    and returns the mean and standard deviation of these metrics.
    """
    all_metrics_runs = []
    
    for run_idx in range(n_runs):
        print(f"\nPRDC Run {run_idx + 1}/{n_runs}")
        current_real_features, current_fake_features = extract_and_sample_features(
            real_df, 
            fake_df,
            n_samples=n_samples_per_run,
            seed=42 + run_idx # Consistent seeding across runs
        )
        
        if current_real_features.size == 0 or current_fake_features.size == 0:
            print(f"Skipping PRDC computation for run {run_idx + 1} due to empty feature arrays.")
            metrics = {metric: np.nan for metric in ['precision', 'recall', 'density', 'coverage', 'f1']}
        else:
            metrics = compute_metrics_with_f1(current_real_features, current_fake_features, nearest_k)
        
        all_metrics_runs.append(metrics)
        
        print(f"Run {run_idx + 1} PRDC metrics:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name.capitalize()}: {value:.4f}")
    
    # Aggregate metrics
    avg_metrics = {}
    std_metrics = {}
    metric_keys = ['precision', 'recall', 'density', 'coverage', 'f1']
    
    for key in metric_keys:
        valid_values = [m[key] for m in all_metrics_runs if m and not np.isnan(m[key])]
        if valid_values:
            avg_metrics[key] = np.mean(valid_values)
            std_metrics[key] = np.std(valid_values)
        else:
            avg_metrics[key] = np.nan
            std_metrics[key] = np.nan
            
    return avg_metrics, std_metrics