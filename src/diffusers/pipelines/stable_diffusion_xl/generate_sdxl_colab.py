from pipeline_stable_diffusion_xl_sparke import SPARKEGuidedStableDiffusionXLPipeline
STABLE_DIFFUSION_XL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
CLIP_Feature_Extractor_PATH = "openai/clip-vit-base-patch32"
REGIONS = {
    'geode': ['Africa', 'EastAsia', 'America', 'Europe'],
    'dollarstreet': ['Europe', 'Africa', 'the Americas', 'Asia'],
}


import os
import argparse
import pandas as pd
import pickle
import torch
from transformers import CLIPFeatureExtractor, CLIPModel
import clip
from torch import distributed as dist

vendi_dir = 'drive/MyDrive/generated_datasets'
import sys
sys.path.insert(0, vendi_dir)

import numpy as np


from utils import image_to_pil, get_F_M
from log import make_logger
from conditional_evaluation import ConditionalEvaluation
# from diffusers.pipelines.stable_diffusion_xl import settings



logger = None


def get_image_path(csv_path, img_id, prompt, algorithm, vscore_scale, guidance_freq, cfg, object_id, seed):
    """Generate the image path."""
    folder_name = csv_path.split('.csv')[0] + '_sdxl_' + f'_{algorithm}_{vscore_scale}_{cfg}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    global logger
    if logger is None:
        logger = make_logger(folder_name, 'logs')

    return f'{folder_name}/{object_id}_{img_id}_{seed}_{prompt.replace("/", "_")}_{algorithm}_clip:{vscore_scale}:{guidance_freq}_cfg:{cfg}.png'


def setup():
    """Set up the distributed environment."""
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())  # Assign the appropriate GPU for each process.


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def generate(dfs_n_csv_path, clip_for_guidance, pre_process_clip, clip_model, feature_extractor, num_real_samples,
             dataset, args):
    """Main image generation function."""
    torch_dtype = torch.float16

    # Load the diffusion pipeline
    criteria_pipe = SPARKEGuidedStableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    criteria_pipe.enable_vae_slicing()

    regions_list = REGIONS

    # if False or not os.path.exists(f'F_M_real_all_{dataset}_{num_real_samples}.pkl'):  # TODO edit it if needed
        # precompute_F_M_F_T_real(pre_process_clip, clip_for_guidance, dataset, num_real_samples)

    if args.contextual_weight != 0:
        # with open(f'F_M_real_all_{dataset}_{num_real_samples}.pkl', 'rb') as f:
        with open('/content/diffusers/src/diffusers/pipelines/stable_diffusion_xl/F_M_real_all_dogs.pkl', 'rb') as f:
            F_M_real_all = pickle.load(f)
    else:
        F_M_real_all = None

    df, csv_path = dfs_n_csv_path

    obj_ref, F_M, F_T, F_M_real, F_T_real = '', None, None, None, None

    # Iterate over the rows of the dataframe
    for index, row in df.iterrows():
        # Extract row data
        prompt = row['prompt']
        obj = row['object']
        guidance_type = row.get('guidance_type', args.guidance_type)
        if guidance_type != 'vscore_clip':
            args.algorithm = guidance_type
        region = row.get(['region'], 'no_region')
        img_id = row.get('img_id', index)
        guidance_freq = row.get('guidance_freq', args.guidance_freq)
        beta = row.get('beta', args.contextual_weight)

        initial_bank_per_object = args.initial_bank_per_object
        if guidance_type == 'vscore_clip' and initial_bank_per_object is True and obj_ref != obj:
            F_M, F_T, F_M_real, F_T_real = None, None, None, None
            obj_ref = obj
            if F_M_real_all is not None:
                F_M_real = F_M_real_all[0] # F_M_real_all[obj.lower()][0]
                F_T_real = F_M_real_all[1] #F_M_real_all[obj.lower()][1]
            print('---------------vscore bank is initiated!-----------')

        # Other parameters
        seed = row.get('seed', index)
        vscore_scale = row.get('vscore_scale', args.vscore_scale)
        cfg = row.get('cfg', args.guidance_scale)
        num_inference_steps = row.get('num_inference_steps', args.num_inference_steps)

        # Generate image path
        image_path = get_image_path(csv_path, img_id, prompt, args.algorithm, vscore_scale, guidance_freq, cfg, object_id=obj, seed=seed)

        # Check if the image already exists
        if not os.path.exists(image_path):
            logger.info(f'Generating img: {image_path}')
            generator = torch.Generator(device='cuda').manual_seed(0)  # TODO set to zero for figure one
            import time
            t1 = time.time()
            torch.cuda.empty_cache()
            # Generate image using the pipeline
            if guidance_type in ['clip_entropy', 'clip_loss', 'vscore_clip']:
                out, F_M, F_T = criteria_pipe(
                    prompt=prompt,
                    guidance_scale=cfg,
                    criteria=guidance_type,
                    algorithm=args.algorithm,
                    height=1024,
                    width=1024,
                    criteria_guidance_scale=vscore_scale,  # check it
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                    clip_for_guidance=clip_for_guidance,  # does not exist
                    use_latents_for_guidance=args.use_latents_for_guidance,
                    guidance_freq=guidance_freq,
                    region=region,
                    F_M=F_M,
                    F_T=F_T,
                    F_M_real=F_M_real,
                    F_T_real=F_T_real,
                    beta=beta,
                    regions_list=regions_list,
                    logger_=logger,
                    return_kernels=True
                )
                image = out[0]
                if F_M is not None:
                    print(F_M[0].shape)
            else:
                raise NotImplementedError
            image.save(image_path)
            del out
            print(image_path)
        else:
            print(f'image id {img_id} is already generated!')
            if args.add_prev_to_bank is True and (F_M is None or F_M[0].shape[0] < args.max_bank_size):
                print(f'loading image id {img_id} to bank!')
                image = image_to_pil(image_path)
                if args.use_latents_for_guidance is True:
                    raise Exception("Use latents is True and add_prev_to_bank is True!")
                features_ = clip_for_guidance.encode_image(image.cuda().unsqueeze(0)).to(torch.float32)
                features_ = features_ / features_.norm(2, dim=1, keepdim=True)
                if F_M is None:
                    F_M = [features_.detach(), torch.mm(features_, features_.T)]
                else:
                    F_M = get_F_M(M=F_M[1], F=F_M[0], f=features_.detach())

                if 'cond' in args.algorithm:
                    features_text = clip_for_guidance.encode_text(
                        clip.tokenize([prompt] * features_.shape[0]).to("cuda")).to(torch.float32)
                    features_text = features_text / features_text.norm(2, dim=1, keepdim=True)
                    if F_T is None:
                        F_T = [features_text.detach(), torch.mm(features_text, features_text.T)]
                    else:
                        F_T = get_F_M(M=F_T[1], F=F_T[0], f=features_text.detach())

        torch.cuda.empty_cache()
        if index > 2 and index % 5 == 0:
            with torch.no_grad():
                eval_model = ConditionalEvaluation(sigma=(0, 0))
                k_x = eval_model.cosine_kernel(F_M[0])
                if F_T is None:  # TODO fix it (buggy when it is not conditional)
                    F_T = F_M
                k_t = eval_model.cosine_kernel(F_T[0])
                cond_vendi, _, joint_vendi, vendi, _ = eval_model.conditional_entropy(k_x, k_t, order=1, compute_kernel=False)
                cond_rke, _, joint_rke, rke, _ = eval_model.conditional_entropy(k_x, k_t, order=2, compute_kernel=False)
                logger.info(f'cond_vendi: {torch.exp(cond_vendi).item() if "cond" in args.algorithm else "not defined"}, joint_vendi: {torch.exp(joint_vendi).item()}, vendi: {torch.exp(vendi).item()}')
                logger.info(f'cond_rke: {torch.exp(cond_rke).item() if "cond" in args.algorithm else "not defined"}, joint_rke: {torch.exp(joint_rke).item()}, rke: {torch.exp(rke).item()}')
        torch.cuda.empty_cache()
    if F_T is not None:
        np.savez(f"{csv_path.split('.csv')[0]}_{args.algorithm}_{args.vscore_scale}_{args.guidance_freq}_txt_feats.npz", txt_feats=F_T[0].detach().cpu().numpy())
    if F_M is not None:
        np.savez(f"{csv_path.split('.csv')[0]}_{args.algorithm}_{args.vscore_scale}_{args.guidance_freq}_img_feats.npz", img_feats=F_M[0].detach().cpu().numpy())


def main():
    """Parse arguments and start processing."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default=f'{vendi_dir}/toy_dataset_same_seed',
                        help='root dir that contains all the csv files of samples to be generated.')
    parser.add_argument('--chunk_size', type=int, default=1_000_000, help='the number of images passed to each gpu.')
    parser.add_argument('--num_real_samples', type=int, default=2)
    parser.add_argument('--dataset', type=str, default='geode')
    parser.add_argument('--guidance_type', type=str, choices=['clip_entropy', 'clip_loss', 'vscore_clip'], default='vscore_clip')
    parser.add_argument('--algorithm', type=str, default='cond-rke', choices=['vendi', 'rke', 'cond-rke', 'cond-vendi'],
                        help="Choose the algorithm: 'vendi', 'rke', or 'cond-rke'. Default is 'vendi'.")
    parser.add_argument('--guidance_freq', type=int, default=10)
    parser.add_argument('--contextual_weight', type=float, default=0)
    parser.add_argument('--vscore_scale', type=float, default=0.05)
    parser.add_argument('--guidance_scale', type=float, default=5.0)
    parser.add_argument('--num_inference_steps', type=int, default=51)

    parser.add_argument('--initial_bank_per_object', type=bool, default=False)
    parser.add_argument('--load_bank_dir', type=str, help='Path to load previously generated images in history bank.', default=None)
    parser.add_argument('--max_bank_size', type=int, help='Maximum number of images to store in the bank.', default=100)
    parser.add_argument('--add_prev_to_bank', type=bool, help='Load previously generated images to bank', default=True)
    parser.add_argument('--use_latents_for_guidance', type=bool, help='Use latents for guidance or CLIP features', default=False)

    args = parser.parse_args()

    torch_dtype = torch.float16
    feature_extractor = CLIPFeatureExtractor.from_pretrained(CLIP_Feature_Extractor_PATH,
                                                             )
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch_dtype,
                                           )
    clip_for_guidance, pre_process_clip = clip.load("ViT-B/32", device='cuda')

    # if args.chunk_size % 180 != 0 and args.dataset == 'geode':
    #     print('chunk_size has to be divisible by 180 (count_per_obj)!')
    #     raise NotImplementedError

    # Setup distributed environment only if running with multiple nodes
    if dist.is_available() and dist.is_initialized():
        setup()

    root_dir = args.root_dir
    list_dfs_n_csv_path = []
    for file in os.listdir(root_dir):
        if '.csv' in file:
            csv_path = os.path.join(root_dir, file)
            df = pd.read_csv(csv_path)
            for chunk in range(0, len(df), args.chunk_size):
                list_dfs_n_csv_path.append((df.iloc[chunk:chunk + args.chunk_size], csv_path))

    # Run on the local machine or use distributed training
    for dfs_n_csv_path in list_dfs_n_csv_path:
        generate(dfs_n_csv_path, clip_for_guidance, pre_process_clip, clip_model, feature_extractor,
                 args.num_real_samples, args.dataset, args)

    # Cleanup distributed environment
    if dist.is_available() and dist.is_initialized():
        cleanup()


if __name__ == "__main__":
    main()
