import pandas as pd
import os
import numpy as np
import torch
import argparse
from sklearn.metrics.pairwise import cosine_similarity # For CLIPScore calculation

try:
    from prdc_metrics import compute_prdc_multiple_runs
except ImportError:
    print("Error: prdc_metrics.py not found.")
    exit()

# Basic get_df_deduped for DINOv2 features
def get_df_deduped(df_to_dedup, feature_column_name='features', id_column=None, keep_one=True):
    if df_to_dedup is None or df_to_dedup.empty:
        return pd.DataFrame()
    if feature_column_name not in df_to_dedup.columns:
        print(f"Warning: Feature column '{feature_column_name}' not found for deduplication. Returning original DataFrame.")
        return df_to_dedup

    def to_hashable(x):
        if x is None:
            return None
        try:
            return tuple(x)
        except TypeError:
            return x

    features_for_dedup = df_to_dedup[feature_column_name].apply(to_hashable)

    initial_count = len(df_to_dedup)
    keep_strategy = 'first' if keep_one else False

    # Drop duplicates based on the hashable representation of features
    df_deduped = df_to_dedup.loc[features_for_dedup.drop_duplicates(keep=keep_strategy).index]

    num_duplicates_removed = initial_count - len(df_deduped)

    if num_duplicates_removed > 0:
        print(f"Deduplication: Removed {num_duplicates_removed} duplicate entries based on '{feature_column_name}'. Kept one instance: {keep_one}.")
    else:
        print(f"Deduplication: No duplicate entries found based on '{feature_column_name}'.")

    return df_deduped

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

def load_data_pkl(path, df_name="", required_columns=None, data_type_checks=None, dim_checks=None):
    """
    Loads a DataFrame from a pickle file.
    Checks for required columns.
    Optionally checks data types and dimensions for specified columns.
    """
    print(f"Loading {df_name} from: {path}")
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return None
    try:
        df = pd.read_pickle(path)

        current_required_columns = []
        if isinstance(required_columns, str):
            current_required_columns = [required_columns]
        elif isinstance(required_columns, list):
            current_required_columns = required_columns

        missing_cols = [col for col in current_required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Required columns {missing_cols} not found in {df_name} DataFrame at {path}.")
            return None

        cols_to_validate = current_required_columns

        for col_name in cols_to_validate:
            if col_name not in df.columns: continue

            if df[col_name].dropna().empty and not df.empty :
                 print(f"Warning: Column '{col_name}' in {df_name} exists but contains only None values.")
                 continue

            first_valid_entry = next((entry for entry in df[col_name] if entry is not None), None)

            if first_valid_entry is None and not df[col_name].empty:
                 print(f"Warning: All entries in '{col_name}' column of {df_name} are None, despite column not being empty by dropna().")
            elif first_valid_entry is not None:
                if data_type_checks and col_name in data_type_checks:
                    expected_types = data_type_checks[col_name]
                    if not isinstance(first_valid_entry, expected_types):
                        print(f"Warning: Entries in '{col_name}' of {df_name} (type {type(first_valid_entry)}) are not of expected types {expected_types}.")
                        if expected_types == (list, np.ndarray) and not isinstance(first_valid_entry, (list, np.ndarray)):
                            print(f"Attempting conversion of '{col_name}' to np.ndarray.")
                            try:
                                df[col_name] = df[col_name].apply(lambda x: np.array(x) if x is not None and hasattr(x, '__iter__') else x)
                                first_valid_entry = next((entry for entry in df[col_name] if entry is not None), None)
                                if first_valid_entry is not None and not isinstance(first_valid_entry, np.ndarray):
                                    raise ValueError("Conversion to np.ndarray failed.")
                            except Exception as e_conv:
                                print(f"Could not convert '{col_name}'. Error: {e_conv}")

                if dim_checks and col_name in dim_checks:
                    if isinstance(first_valid_entry, (list, np.ndarray)):
                        expected_dim = dim_checks[col_name]
                        try:
                            actual_dim = len(first_valid_entry)
                            if actual_dim != expected_dim:
                                print(f"Warning: Data in '{col_name}' of {df_name} have dimension {actual_dim}, but expected {expected_dim}.")
                        except TypeError:
                             print(f"Warning: Could not determine dimension for data in '{col_name}' of {df_name} (type: {type(first_valid_entry)}). Expected list/array.")
                    else:
                        print(f"Warning: Dimension check for '{col_name}' skipped as data is not list/array (type: {type(first_valid_entry)}).")

        print(f"Successfully loaded {df_name} DataFrame with shape {df.shape}.")
        return df
    except Exception as e:
        print(f"Error loading or processing {df_name} pickle file '{path}': {e}")
        return None

def compute_clip_scores_from_features(image_features_list, text_features_list):
    if not image_features_list or not text_features_list:
        print("Warning: Empty image or text feature list for CLIPScore calculation.")
        return []
    if len(image_features_list) != len(text_features_list):
        print(f"Error: Image ({len(image_features_list)}) and text ({len(text_features_list)}) feature lists have different lengths. Cannot compute CLIP scores accurately.")
        return []

    scores = []
    for img_feat, txt_feat in zip(image_features_list, text_features_list):
        if img_feat is None or txt_feat is None:
            scores.append(np.nan)
            continue
        try:
            img_feat_np = np.asarray(img_feat, dtype=np.float32).reshape(1, -1)
            txt_feat_np = np.asarray(txt_feat, dtype=np.float32).reshape(1, -1)
        except Exception as e:
            print(f"Warning: Could not convert features to NumPy array for CLIPScore: {e}. Skipping this pair.")
            scores.append(np.nan)
            continue

        norm_img = np.linalg.norm(img_feat_np, axis=1, keepdims=True)
        norm_txt = np.linalg.norm(txt_feat_np, axis=1, keepdims=True)

        if norm_img == 0 or norm_txt == 0: 
            scores.append(0.0) 
            continue

        img_feat_norm = img_feat_np / norm_img
        txt_feat_norm = txt_feat_np / norm_txt

        similarity = cosine_similarity(img_feat_norm, txt_feat_norm)[0,0]
        scores.append(similarity * 100.0)
    return scores

def main_evaluation_pipeline(args):
    os.makedirs(args.output_base_dir, exist_ok=True)
    all_summary_metrics = {}

    # --- 1. CLIP Score Evaluation ---
    if args.run_clip_score:
        print("\n--- Calculating CLIP Score (from pre-computed CLIP features) ---")
        if args.fake_clip_image_features_pkl and args.fake_clip_text_features_pkl:
            image_clip_df = load_data_pkl(
                args.fake_clip_image_features_pkl, "Fake Image CLIP Features", 'features',
                data_type_checks={'features': (list, np.ndarray)},
                dim_checks={'features': args.clip_feature_dim} if args.clip_feature_dim else None
            )
            text_clip_df = load_data_pkl(
                args.fake_clip_text_features_pkl, "Fake Text CLIP Features", 'features',
                data_type_checks={'features': (list, np.ndarray)},
                dim_checks={'features': args.clip_feature_dim} if args.clip_feature_dim else None
            )
            if image_clip_df is not None and not image_clip_df.empty and \
               text_clip_df is not None and not text_clip_df.empty:

                img_features = image_clip_df['features'].dropna().tolist() if 'features' in image_clip_df else []
                txt_features = text_clip_df['features'].dropna().tolist() if 'features' in text_clip_df else []

                if not img_features or not txt_features :
                    all_summary_metrics['Average CLIP Score'] = "N/A (No valid features found in one or both PKLs)"
                    print("Warning: No valid features found for CLIPScore after loading and dropna().")
                elif len(img_features) != len(txt_features):
                    print(f"Error: Mismatch in number of valid image features ({len(img_features)}) and text features ({len(txt_features)}) after dropna().")
                    all_summary_metrics['Average CLIP Score'] = "N/A (Feature count mismatch after dropna)"
                else:
                    clip_scores = compute_clip_scores_from_features(img_features, txt_features)
                    if clip_scores: 
                        valid_scores = [s for s in clip_scores if not np.isnan(s)]
                        if valid_scores:
                            avg_clip_score_val = np.mean(valid_scores)
                            all_summary_metrics['Average CLIP Score'] = f"{avg_clip_score_val:.4f}"
                            print(f"Average CLIP Score: {avg_clip_score_val:.4f} (from {len(valid_scores)} valid pairs)")
                        else:
                            all_summary_metrics['Average CLIP Score'] = "N/A (All scores were NaN)"
                            print("Warning: All calculated CLIP scores were NaN.")

                        # Save calculated scores
                        scores_df = pd.DataFrame({'clip_score': clip_scores})
                        path = args.clip_calculated_scores_save_pkl or os.path.join(args.output_base_dir, "calculated_clip_scores.pkl")
                        scores_df.to_pickle(path)
                        print(f"Calculated CLIP scores (incl. NaNs) saved to: {path}")
                    else:
                        all_summary_metrics['Average CLIP Score'] = "N/A (Calculation produced no scores)"
            else:
                all_summary_metrics['Average CLIP Score'] = "N/A (Feature loading failed for one or both PKLs)"
        else:
            all_summary_metrics['Average CLIP Score'] = "N/A (Required PKL paths not provided)"
        print("="*70)

    # --- 2. DINOv2 Based PRDC Evaluation ---
    real_dinov2_features_prdc_df = None
    fake_dinov2_features_df = None

    if args.run_prdc:
        if args.real_dinov2_features_prdc_pkl:
            real_dinov2_features_prdc_df = load_data_pkl(
                args.real_dinov2_features_prdc_pkl, "Real DINOv2 Features (for PRDC)", 'features',
                data_type_checks={'features': (list, np.ndarray)},
                dim_checks={'features': args.dinov2_feature_dim} if args.dinov2_feature_dim else None
            )
        else:
            print("Skipping PRDC: Real DINOv2 features PKL not provided.")
            args.run_prdc = False 

        if args.run_prdc and args.fake_dinov2_features_pkl: 
            fake_dinov2_features_df = load_data_pkl(
                args.fake_dinov2_features_pkl, "Fake DINOv2 Features (for PRDC)", 'features',
                data_type_checks={'features': (list, np.ndarray)},
                dim_checks={'features': args.dinov2_feature_dim} if args.dinov2_feature_dim else None
            )
            if fake_dinov2_features_df is None: 
                print("Disabling PRDC due to failed fake DINOv2 features load.")
                args.run_prdc = False
        elif args.run_prdc: 
            print("Skipping PRDC: Fake DINOv2 features PKL not provided.")
            args.run_prdc = False


    if args.run_prdc: 
        print("\n--- Running PRDC Evaluation (using DINOv2 features) ---")
        if real_dinov2_features_prdc_df is not None and not real_dinov2_features_prdc_df.empty and \
           fake_dinov2_features_df is not None and not fake_dinov2_features_df.empty:

            if 'features' not in real_dinov2_features_prdc_df.columns or \
               'features' not in fake_dinov2_features_df.columns:
                print("Error: 'features' column missing in DINOv2 DataFrames for PRDC. Skipping.")
            else:
                
                real_features_list = [np.array(f, dtype=np.float32) for f in real_dinov2_features_prdc_df['features'].dropna() if f is not None]
                fake_features_list = [np.array(f, dtype=np.float32) for f in fake_dinov2_features_df['features'].dropna() if f is not None]

                if not real_features_list or not fake_features_list:
                    print("Error: No valid DINOv2 features found for PRDC after filtering Nones.")
                else:
                    
                    first_real_dim = len(real_features_list[0])
                    first_fake_dim = len(fake_features_list[0])

                    if not all(len(f) == first_real_dim for f in real_features_list) or \
                       not all(len(f) == first_fake_dim for f in fake_features_list):
                        print("Error: Inconsistent feature dimensions within real or fake DINOv2 sets. Skipping PRDC.")
                    elif args.dinov2_feature_dim and (first_real_dim != args.dinov2_feature_dim or first_fake_dim != args.dinov2_feature_dim):
                         print(f"Error: DINOv2 feature dimension mismatch against expected {args.dinov2_feature_dim}. Real: {first_real_dim}, Fake: {first_fake_dim}. Skipping PRDC.")
                    elif first_real_dim != first_fake_dim:
                        print(f"Error: DINOv2 feature dimension mismatch between Real ({first_real_dim}) and Fake ({first_fake_dim}) sets. Skipping PRDC.")
                    else:
                        print(f"DINOv2 features for PRDC have dimension: {first_real_dim}")

                        df_r_prdc = pd.DataFrame({'features': real_features_list})
                        df_f_prdc = pd.DataFrame({'features': fake_features_list})

                        # Deduplication
                        if args.prdc_deduplicate_real_features: df_r_prdc = get_df_deduped(df_r_prdc, 'features', keep_one=args.prdc_dedup_real_keep_one)
                        if args.prdc_deduplicate_fake_features: df_f_prdc = get_df_deduped(df_f_prdc, 'features', keep_one=args.prdc_dedup_fake_keep_one)

                        if df_r_prdc.empty or df_f_prdc.empty:
                            print("Skipping PRDC: One or both feature sets are empty after deduplication/filtering.")
                        else:
                            avg_m, std_m = compute_prdc_multiple_runs(
                                df_r_prdc, df_f_prdc,
                                n_runs=args.prdc_n_runs,
                                nearest_k=args.prdc_nearest_k,
                                n_samples_per_run=args.prdc_n_samples_per_run
                            )
                            for name in avg_m: # Precision, Recall, Density, Coverage
                                metric_key = f'DINOv2 PRDC {name.capitalize()}'
                                if not np.isnan(avg_m[name]):
                                    avg_val_str = f"{avg_m[name]:.4f}"
                                    if not np.isnan(std_m[name]):
                                        std_val_str = f"{std_m[name]:.4f}"
                                        all_summary_metrics[metric_key] = f"{avg_val_str} ± {std_val_str}"
                                    else: # std is NaN
                                        all_summary_metrics[metric_key] = f"{avg_val_str} ± N/A"
                                else: # avg is NaN
                                    all_summary_metrics[metric_key] = "N/A"
        else:
            print("Skipping PRDC: DINOv2 features missing or load failed for real or fake sets.")
        print("="*70)

    # --- Final Summary ---
    summary_log_path = args.summary_log_path or os.path.join(args.output_base_dir, "evaluation_summary_clip_prdc.txt")
    print("\n--- Final Evaluation Summary ---")
    with open(summary_log_path, "w") as f:
        f.write("Evaluation Summary\n")
        try:
            f.write(f"Command: {' '.join(os.sys.argv)}\n")
        except Exception:
            f.write("Command: Could not retrieve command line arguments.\n")
        f.write("="*30 + "\n")
        if not all_summary_metrics:
            f.write("No metrics were computed.\n")
            print("No metrics were computed.")
        else:
            for key, value in all_summary_metrics.items():
                log_line = f"{key}: {value}"
                print(log_line)
                f.write(log_line + "\n")
    print(f"\nSummary written to: {summary_log_path}")
    print("="*70)
    print("\nEvaluation Pipeline Finished.")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation pipeline (CLIPScore, PRDC) using pre-computed features.")

    # General Args
    parser.add_argument('--output_base_dir', type=str, default="outputs_evaluation_clip_prdc", help="Base directory for all outputs.")
    parser.add_argument('--device', type=str, default=None, help="Device to use. Not directly used as features are pre-computed.")

    # Evaluation Toggles
    eval_toggles = parser.add_argument_group('Evaluation Toggles')
    eval_toggles.add_argument('--run_clip_score', type=str2bool, nargs='?', const=True, default=True, help="Run CLIP Score from pre-computed CLIP features.")
    eval_toggles.add_argument('--run_prdc', type=str2bool, nargs='?', const=True, default=True, help="Run PRDC from pre-computed DINOv2 features.")


    # CLIP Score Args
    clip_group = parser.add_argument_group('CLIP Score Parameters (from pre-computed features)')
    clip_group.add_argument('--fake_clip_image_features_pkl', type=str, help="Path to PKL with FAKE image CLIP features (col 'features').")
    clip_group.add_argument('--fake_clip_text_features_pkl', type=str, help="Path to PKL with TEXT CLIP features (col 'features'), aligned with image features.")
    clip_group.add_argument('--clip_feature_dim', type=int, default=None, help="Expected dimension of CLIP features (e.g., 512 for ViT-B/32, 768 for ViT-L/14). For validation.")
    clip_group.add_argument('--clip_calculated_scores_save_pkl', type=str, default=None, help="Path to save calculated CLIP scores PKL.")

    # DINOv2 Data Paths (for PRDC)
    dinov2_group = parser.add_argument_group('DINOv2 Pre-computed Data Paths (for PRDC)')
    dinov2_group.add_argument('--real_dinov2_features_prdc_pkl', type=str, help="Path to PKL with REAL DINOv2 features for PRDC (col 'features').")
    dinov2_group.add_argument('--fake_dinov2_features_pkl', type=str, help="Path to PKL file with FAKE DINOv2 features (col 'features') for PRDC.")
    dinov2_group.add_argument('--dinov2_feature_dim', type=int, default=None, help="Expected dimension of DINOv2 features (e.g., 384 for ViT-S/14, 768 for ViT-B/14). For validation.")

    # PRDC Args
    prdc_group = parser.add_argument_group('PRDC Parameters (with DINOv2 features)')
    prdc_group.add_argument('--prdc_n_runs', type=int, default=5, help="Number of runs for PRDC for stable estimates.")
    prdc_group.add_argument('--prdc_nearest_k', type=int, default=3, help="Nearest K for PRDC's precision and recall.")
    prdc_group.add_argument('--prdc_n_samples_per_run', type=int, default=None, help="Number of samples per run for PRDC (None for min(real, fake) features).")
    prdc_group.add_argument('--prdc_deduplicate_real_features', type=str2bool, nargs='?', const=True, default=True, help="Deduplicate real DINOv2 features for PRDC.")
    prdc_group.add_argument('--prdc_dedup_real_keep_one', type=str2bool, nargs='?', const=True, default=True, help="If deduplicating real DINOv2 features for PRDC, keep one instance.")
    prdc_group.add_argument('--prdc_deduplicate_fake_features', type=str2bool, nargs='?', const=True, default=True, help="Deduplicate fake DINOv2 features for PRDC.")
    prdc_group.add_argument('--prdc_dedup_fake_keep_one', type=str2bool, nargs='?', const=True, default=False, help="If deduplicating fake DINOv2 features for PRDC, keep one instance (False to remove all duplicates).")

    # Summary Log Path
    parser.add_argument('--summary_log_path', type=str, default=None, help="Path for the final summary TXT file.")

    args = parser.parse_args()

    # Validation
    if args.run_clip_score and (not args.fake_clip_image_features_pkl or not args.fake_clip_text_features_pkl):
        parser.error("--fake_clip_image_features_pkl and --fake_clip_text_features_pkl are required when --run_clip_score is enabled.")
    if args.run_prdc and (not args.real_dinov2_features_prdc_pkl or not args.fake_dinov2_features_pkl):
        parser.error("--real_dinov2_features_prdc_pkl and --fake_dinov2_features_pkl are required when --run_prdc is enabled.")

    main_evaluation_pipeline(args)