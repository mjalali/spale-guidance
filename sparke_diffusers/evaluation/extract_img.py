import os
import torch
import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np
from glob import glob
from PIL import Image # For PickScore

# --- ImageReward Imports and Setup ---
try:
    import ImageReward as RM
except ImportError:
    print("ImageReward library not found. Please install it: pip install image-reward")
    RM = None

# --- PickScore Imports and Setup ---
try:
    from transformers import AutoProcessor, AutoModel
except ImportError:
    print("Transformers library not found. Please install it for PickScore: pip install transformers")
    AutoProcessor = None
    AutoModel = None

# Global models and processors (initialized by load functions)
REWARD_MODEL = None
PICKSCORE_MODEL = None
PICKSCORE_PROCESSOR = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_image_reward_model(model_name, cache_dir, device):
    global REWARD_MODEL
    if RM is None:
        print("ImageReward library is not available. Skipping ImageReward model loading.")
        return False

    print(f"Loading ImageReward model: {model_name} to device: {device}")
    if cache_dir:
        print(f"ImageReward model download/cache directory set to: {cache_dir}")
    else:
        default_rm_cache = os.path.join(os.path.expanduser("~"), ".cache", "ImageReward")
        print(f"ImageReward model_cache_dir not specified, using ImageReward default: {default_rm_cache}")

    try:
        REWARD_MODEL = RM.load(model_name, download_root=cache_dir)
        if hasattr(REWARD_MODEL, 'to'):
            REWARD_MODEL = REWARD_MODEL.to(device)
        elif hasattr(REWARD_MODEL, 'module') and hasattr(REWARD_MODEL.module, 'to'):
            REWARD_MODEL.module.to(device)
        print("ImageReward model loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading ImageReward model: {e}")
        if "Disk quota exceeded" in str(e) or "Errno 122" in str(e):
            print("Hint: This looks like a disk quota or space issue for ImageReward.")
        REWARD_MODEL = None
        return False

def load_pickscore_model(processor_name, model_name, cache_dir, device):
    global PICKSCORE_MODEL, PICKSCORE_PROCESSOR
    if AutoProcessor is None or AutoModel is None:
        print("Transformers library not available. Skipping PickScore model loading.")
        return False

    print(f"Loading PickScore processor: {processor_name} and model: {model_name} to device: {device}")
    if cache_dir:
        print(f"PickScore Hugging Face cache directory set to: {cache_dir}")
        os.environ['HF_HOME'] = cache_dir

    try:
        PICKSCORE_PROCESSOR = AutoProcessor.from_pretrained(processor_name, cache_dir=cache_dir)
        PICKSCORE_MODEL = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).eval().to(device)
        print("PickScore model and processor loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading PickScore model/processor: {e}")
        PICKSCORE_MODEL = None
        PICKSCORE_PROCESSOR = None
        return False

def calc_pickscore_probs(prompt: str, pil_images: list, device: str):
    if PICKSCORE_MODEL is None or PICKSCORE_PROCESSOR is None:
        return [np.nan] * len(pil_images)
    if not pil_images:
        return []

    try:
        image_inputs = PICKSCORE_PROCESSOR(
            images=pil_images, padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(device)
        text_inputs = PICKSCORE_PROCESSOR(
            text=prompt, padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            image_embs = PICKSCORE_MODEL.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            text_embs = PICKSCORE_MODEL.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
            scores = PICKSCORE_MODEL.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            probs = torch.softmax(scores, dim=-1)
        return probs.cpu().tolist()
    except Exception as e:
        print(f"Error during PickScore calculation for prompt '{prompt[:30]}...': {e}")
        return [np.nan] * len(pil_images)


def main():
    global DEVICE

    parser = argparse.ArgumentParser(description="Calculate ImageReward and/or PickScore for images against prompts and save to TXT.")
    parser.add_argument('--image_dir', type=str, required=True, help="Path to the directory containing images.")
    parser.add_argument('--prompt_csv', type=str, required=True, help="Path to a CSV file containing prompts.")
    parser.add_argument('--prompt_column_name', type=str, default='prompts', help="Name of the column containing prompts in the CSV file.")
    parser.add_argument('--image_extensions', type=str, nargs='+', default=['.png', '.jpg', '.jpeg', '.webp', '.bmp'], help="List of image extensions to process.")
    parser.add_argument('--output_txt', type=str, required=True, help="Path to save the output TXT file with scores and means.") # Changed from output_pkl
    parser.add_argument('--device', type=str, default=DEVICE, help=f"Device to use ('cuda' or 'cpu', default: {DEVICE}).")

    ir_group = parser.add_argument_group('ImageReward Parameters')
    ir_group.add_argument('--run_image_reward', action='store_true', help="Enable ImageReward calculation.")
    ir_group.add_argument('--image_reward_model_name', type=str, default="ImageReward-v1.0", help="Name of the ImageReward model.")
    ir_group.add_argument('--image_reward_cache_dir', type=str, default=None, help="Directory for ImageReward model (download_root).")

    ps_group = parser.add_argument_group('PickScore Parameters')
    ps_group.add_argument('--run_pickscore', action='store_true', help="Enable PickScore calculation.")
    ps_group.add_argument('--pickscore_processor_name', type=str, default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", help="Name of PickScore processor.")
    ps_group.add_argument('--pickscore_model_name', type=str, default="yuvalkirstain/PickScore_v1", help="Name of PickScore model.")
    ps_group.add_argument('--pickscore_hf_cache_dir', type=str, default=None, help="Hugging Face cache directory for PickScore models.")

    args = parser.parse_args()
    DEVICE = args.device

    if not args.run_image_reward and not args.run_pickscore:
        print("Neither ImageReward nor PickScore calculation is enabled. Use --run_image_reward and/or --run_pickscore. Exiting.")
        return

    image_reward_loaded = False
    if args.run_image_reward:
        if RM is None: print("ImageReward library not installed, but --run_image_reward specified.")
        else: image_reward_loaded = load_image_reward_model(args.image_reward_model_name, args.image_reward_cache_dir, DEVICE)
        if not image_reward_loaded: print("ImageReward scores will not be calculated.")

    pickscore_loaded = False
    if args.run_pickscore:
        if AutoProcessor is None or AutoModel is None: print("Transformers library not installed, but --run_pickscore specified.")
        else: pickscore_loaded = load_pickscore_model(args.pickscore_processor_name, args.pickscore_model_name, args.pickscore_hf_cache_dir, DEVICE)
        if not pickscore_loaded: print("PickScore probabilities will not be calculated.")

    if (args.run_image_reward and not image_reward_loaded) and (args.run_pickscore and not pickscore_loaded):
        print("Failed to load any requested models. Exiting.")
        return
    if not image_reward_loaded and not pickscore_loaded:
        print("No models were successfully loaded (or requested). Exiting.")
        return

    if not os.path.exists(args.prompt_csv):
        print(f"Error: Prompt CSV file not found at {args.prompt_csv}"); return
    try:
        prompt_df = pd.read_csv(args.prompt_csv)
        if args.prompt_column_name not in prompt_df.columns:
            print(f"Error: Prompt column '{args.prompt_column_name}' not in {args.prompt_csv}. Columns: {prompt_df.columns.tolist()}"); return
        prompts_list = prompt_df[args.prompt_column_name].astype(str).unique().tolist()
        if not prompts_list: print(f"No prompts in '{args.prompt_column_name}' of {args.prompt_csv}."); return
        print(f"Loaded {len(prompts_list)} unique prompts from {args.prompt_csv}.")
    except Exception as e: print(f"Error reading prompt CSV '{args.prompt_csv}': {e}"); return

    if not os.path.isdir(args.image_dir): print(f"Error: Image directory not found at {args.image_dir}"); return
    image_paths = []
    for ext in args.image_extensions:
        image_paths.extend(sorted(glob(os.path.join(args.image_dir, f"*{ext}" if ext.startswith('.') else f"*.{ext}"))))
    if not image_paths: print(f"No images found in '{args.image_dir}' with extensions {args.image_extensions}"); return
    print(f"Found {len(image_paths)} images to process against each prompt.")

    pil_images_for_pickscore = []
    if args.run_pickscore and pickscore_loaded:
        print("Pre-loading PIL images for PickScore...")
        for img_path in tqdm(image_paths, desc="Loading PIL Images"):
            try: pil_images_for_pickscore.append(Image.open(img_path).convert("RGB"))
            except Exception as e:
                print(f"Warning: Could not load image {img_path} for PickScore: {e}. It will be skipped for PickScore.")
                pil_images_for_pickscore.append(None)

    results_list = []
    print(f"Starting scoring for {len(prompts_list) * len(image_paths)} (prompt, image) potential pairs...")

    with torch.no_grad():
        for prompt_text in tqdm(prompts_list, desc="Processing Prompts"):
            current_prompt_pickscore_probs = [np.nan] * len(image_paths)
            if args.run_pickscore and pickscore_loaded and pil_images_for_pickscore:
                valid_pil_images_for_prompt = [img for img in pil_images_for_pickscore if img is not None]
                if valid_pil_images_for_prompt:
                    probs_for_valid_images = calc_pickscore_probs(prompt_text, valid_pil_images_for_prompt, DEVICE)
                    prob_idx = 0
                    for original_idx, pil_img in enumerate(pil_images_for_pickscore):
                        if pil_img is not None:
                            if prob_idx < len(probs_for_valid_images):
                                current_prompt_pickscore_probs[original_idx] = probs_for_valid_images[prob_idx]; prob_idx += 1
                            else: current_prompt_pickscore_probs[original_idx] = np.nan # Should not happen
                        else: current_prompt_pickscore_probs[original_idx] = np.nan
                else: print(f"No valid PIL images for PickScore for prompt: {prompt_text[:30]}...")

            for img_idx, img_path in enumerate(tqdm(image_paths, desc=f"Images for prompt: {prompt_text[:30]}...", leave=False)):
                result_entry = {'image_path': img_path, 'prompt_text': prompt_text,
                                'image_reward_score': np.nan, 'pickscore_probability': np.nan}
                if not os.path.exists(img_path):
                    print(f"Warning: Image {img_path} not found. Skipping."); results_list.append(result_entry); continue
                
                if args.run_image_reward and image_reward_loaded:
                    try: result_entry['image_reward_score'] = REWARD_MODEL.score(prompt_text, img_path)
                    except Exception as e: print(f"Error (ImageReward) on {img_path} with '{prompt_text}': {e}")
                
                if args.run_pickscore and pickscore_loaded:
                    result_entry['pickscore_probability'] = current_prompt_pickscore_probs[img_idx]
                results_list.append(result_entry)

    # --- Output to TXT file and Calculate Means ---
    output_df = pd.DataFrame(results_list) # Use DataFrame for easy NaN handling and mean calculation
    
    mean_image_reward_score = np.nan
    if args.run_image_reward and 'image_reward_score' in output_df.columns:
        # Ensure column is numeric, coercing errors, then calculate mean ignoring NaNs
        numeric_ir_scores = pd.to_numeric(output_df['image_reward_score'], errors='coerce')
        if not numeric_ir_scores.isnull().all(): # Check if there are any valid scores
             mean_image_reward_score = numeric_ir_scores.mean()

    mean_pickscore_prob = np.nan
    if args.run_pickscore and 'pickscore_probability' in output_df.columns:
        numeric_ps_probs = pd.to_numeric(output_df['pickscore_probability'], errors='coerce')
        if not numeric_ps_probs.isnull().all():
            mean_pickscore_prob = numeric_ps_probs.mean()

    try:
        with open(args.output_txt, 'w', encoding='utf-8') as f:
            f.write("--- Individual Scores ---\n")
            if not output_df.empty:
                for index, row in output_df.iterrows():
                    line = f"Prompt: \"{row['prompt_text']}\", Image: \"{os.path.basename(row['image_path'])}\""
                    if args.run_image_reward:
                        line += f", ImageReward: {row['image_reward_score']:.4f}" if pd.notna(row['image_reward_score']) else ", ImageReward: NaN"
                    if args.run_pickscore:
                        line += f", PickScore: {row['pickscore_probability']:.4f}" if pd.notna(row['pickscore_probability']) else ", PickScore: NaN"
                    f.write(line + "\n")
            else:
                f.write("No results to display.\n")

            f.write("\n--- Summary Statistics ---\n")
            if args.run_image_reward:
                f.write(f"Mean ImageReward Score: {mean_image_reward_score:.4f}\n" if pd.notna(mean_image_reward_score) else "Mean ImageReward Score: NaN (No valid scores or not run)\n")
            if args.run_pickscore:
                f.write(f"Mean PickScore Probability: {mean_pickscore_prob:.4f}\n" if pd.notna(mean_pickscore_prob) else "Mean PickScore Probability: NaN (No valid scores or not run)\n")
            f.write(f"Total image-prompt pairs processed: {len(output_df)}\n")

        print(f"\nResults and means saved to {args.output_txt}")
        if args.run_image_reward:
            print(f"Mean ImageReward Score: {mean_image_reward_score:.4f}" if pd.notna(mean_image_reward_score) else "Mean ImageReward Score: NaN")
        if args.run_pickscore:
            print(f"Mean PickScore Probability: {mean_pickscore_prob:.4f}" if pd.notna(mean_pickscore_prob) else "Mean PickScore Probability: NaN")

    except Exception as e:
        print(f"Error saving output TXT file: {e}")

if __name__ == "__main__":
    main()