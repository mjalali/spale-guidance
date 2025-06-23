from diffusers import ControlNetModel

# RKE Guidance Pipelines
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_sparke import SPARKEGuidedStableDiffusionPipeline
from diffusers.pipelines.controlnet.pipeline_controlnet_rke import RKEGuidedStableDiffusionControlNetPipeline
from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl_sparke import SPARKEGuidedStableDiffusionXLControlNetPipeline
from diffusers.pipelines.stable_diffusion_xl import SPARKEGuidedStableDiffusionXLPipeline
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma_sparke import SPARKEGuidedPixArtSigmaPipeline
from diffusers.pipelines.kandinsky3.pipeline_kandinsky3_sparke import SPARKEGuidedKandinsky3Pipeline
from diffusers.pipelines.dit.pipeline_dit_sparke import SPARKEGuidedDiTPipeline


from diffusers.utils import load_image
from controlnet_aux import OpenposeDetector
from diffusers import DPMSolverMultistepScheduler
from functools import partial, wraps

STABLE_DIFFUSION_MODEL_2_1 = "/media/student/data/models/SD2.1/stable-diffusion-2-1/"
STABLE_DIFFUSION_MODEL_1_5 = "/media/student/data/models/SD1.5/stable-diffusion-v1-5/"
PIXART_SIGMA_MODEL = "/media/student/data/models/Pixart/PixArt-Sigma-XL-2-1024-MS/"
KANDINSKY_3_MODEL = "/media/student/data/models/Kandinsky/kandinsky-3/"
PLAYGROUND_2_5 = "/media/student/data/models/playground/playground-v2.5-1024px-aesthetic/"
DiT_XL_2_MODEL = "/media/student/data/models/DiT/DiT-XL-2-256/"

ControlNet_PATH = "/media/student/data/models/ControlNet/ControlNet/"
ControlNet_OPENPOSE_PATH = "/media/student/data/models/ControlNet/sd-controlnet-openpose/"


from diffusers.pipelines.rke_guidance_utils import RKEGuidedSampling

import os
import argparse
import pandas as pd
import pickle
import torch
from transformers import CLIPFeatureExtractor, CLIPModel
import clip
from torch import distributed as dist
from datetime import datetime

vendi_dir_ = '/Contextualized-Vendi-Score-Guidance'
vendi_dir = '/media/student/data/rke_guidance'


import sys
sys.path.insert(0, vendi_dir_)

import numpy as np


from diffusers.pipelines.stable_diffusion_xl.utils import image_to_pil, get_F_M
from log import make_logger
import settings
from diffusers.pipelines.stable_diffusion_xl.conditional_evaluation import ConditionalEvaluation

def get_diffusion_pipeline(name='sdv2.1'):
    if name == 'sdv2.1':
        pipe = SPARKEGuidedStableDiffusionPipeline.from_pretrained(STABLE_DIFFUSION_MODEL_2_1, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")
        return pipe
    if name == 'sdv1.5':
        pipe = SPARKEGuidedStableDiffusionPipeline.from_pretrained(STABLE_DIFFUSION_MODEL_1_5, torch_dtype=torch.float16,
                                                                   safety_checker=None)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")
        return partial(pipe, height=512, width=512)
    if name == 'sdxl':
        pipe = SPARKEGuidedStableDiffusionXLPipeline.from_pretrained(
            settings.STABLE_DIFFUSION_XL_MODEL,
            torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")
        pipe.enable_vae_slicing()
        return pipe
    if name == 'controlnet_pose_sdv1.5':
        controlnet = ControlNetModel.from_pretrained(ControlNet_OPENPOSE_PATH,torch_dtype=torch.float16)
        pipe = RKEGuidedStableDiffusionControlNetPipeline.from_pretrained(
            STABLE_DIFFUSION_MODEL_1_5,
            controlnet=controlnet,
            torch_dtype=torch.float16
        )
        # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")
        return pipe
    if name == 'controlnet_pose_sdxl':  # TODO add sdxl pose to image
        pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to("cuda")
        # Load SDXL + Pose ControlNet
        controlnet = ControlNetModel.from_pretrained(
            "thibaud/controlnet-openpose-sdxl-1.0",  # SDXL-compatible OpenPose ControlNet
            torch_dtype=torch.float16
        ).to("cuda")
        pipe = SPARKEGuidedStableDiffusionXLControlNetPipeline.from_pretrained(
            settings.STABLE_DIFFUSION_XL_MODEL,
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to("cuda")
        return pipe
    if name == 'pixart-sigma':
        pipe = SPARKEGuidedPixArtSigmaPipeline.from_pretrained(
            PIXART_SIGMA_MODEL,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")
        return pipe
    if name == 'kandinsky-3':
        pipe = SPARKEGuidedKandinsky3Pipeline.from_pretrained(
            KANDINSKY_3_MODEL,
            variant="fp16",
            torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()
        return pipe
    if name == 'playgroundv2.5':
        pass
    if name == 'dit-xl-2':
        def class_name_to_id(original_func):
            """Decorator to convert class names to ImageNet IDs for DiT models"""

            @wraps(original_func)
            def wrapper(*args, **kwargs):
                class_names = kwargs.pop('prompt', [])
                if ',' in class_names and type(class_names) == str:  # TODO edit this for cases when you have additional information in prompt
                    class_names = [class_names.split(',')[0]]
                # Get label IDs conversion logic (implement this separately)
                label_ids = original_func.get_label_ids(class_names)
                pipe = original_func(*args, **kwargs, class_labels=label_ids)
                return pipe
            return wrapper
        logger.warning("You are using DiT-XL-2 model which is a class to image model, "
                       "prompt should be the imagenet classes, such as `golden retriever`")
        pipe = SPARKEGuidedDiTPipeline.from_pretrained(DiT_XL_2_MODEL)
        pipe = pipe.to("cuda")
        return class_name_to_id(pipe)


logger = None
folder_name = None
MODEL_NAME = None


def get_image_path(csv_path, img_id, prompt, algorithm, vscore_scale, guidance_freq, cfg, object_id, seed, bank_dir=None):
    """Generate the image path."""
    global folder_name
    if folder_name is None:
        folder_name = csv_path.split('.csv')[0] + f'_{MODEL_NAME}_' + f'_{algorithm}_{vscore_scale}_{cfg}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    global logger
    if logger is None:
        logger = make_logger(folder_name, 'logs')

    return f'{bank_dir or folder_name}/{object_id}_{img_id}_{seed}_{prompt.replace("/", "_")}_{algorithm}_clip:{vscore_scale}:{guidance_freq}_cfg:{cfg}.png'


def setup():
    """Set up the distributed environment."""
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())  # Assign the appropriate GPU for each process.


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def generate(dfs_n_csv_path, clip_for_guidance, pre_process_clip, clip_model, feature_extractor,
             dataset, args):
    """Main image generation function."""
    torch_dtype = torch.float16

    # Load the diffusion pipeline
    criteria_pipe = get_diffusion_pipeline(name=MODEL_NAME)
    regions_list = settings.REGIONS

    openpose = None
    if 'controlnet' in MODEL_NAME:
        openpose = OpenposeDetector.from_pretrained(ControlNet_PATH)

    if args.contextual_weight != 0:
        with open('./F_M_real_all_dogs.pkl', 'rb') as f:
            F_M_real_all = pickle.load(f)
    else:
        F_M_real_all = None

    df, csv_path = dfs_n_csv_path

    obj_ref, F_M, F_T, F_M_real, F_T_real = '', None, None, None, None

    eval_model = ConditionalEvaluation(sigma=(0, 0))
    if args.add_prev_to_bank is True and args.load_bank_npz is not None:
        img_feats = torch.tensor(np.load(args.load_bank_npz)['img_feats'], device='cuda', dtype=torch_dtype)
        if args.kernel == 'gaussian':
            img_kernel = eval_model.gaussian_kernel(img_feats, sigma=args.sigma_image, normalize=False)
        F_M = [img_feats.detach(), img_kernel]
        if os.path.exists(args.load_bank_npz.replace('img_feats', 'txt_feats')):
            txt_feats = torch.tensor(np.load(args.load_bank_npz.replace('img_feats', 'txt_feats'))['txt_feats'],
                                     device='cuda', dtype=torch_dtype)
            if args.kernel == 'gaussian':
                txt_kernel = eval_model.gaussian_kernel(txt_feats, sigma=args.sigma_text, normalize=False)
            F_T = [txt_feats.detach(), txt_kernel]

    rke_guided_sampler = RKEGuidedSampling(
        algorithm=args.algorithm,
        kernel=args.kernel,
        sigma_image=args.sigma_image,
        sigma_text=args.sigma_text,
        max_bank_size=args.max_bank_size,
        use_latents_for_guidance=args.use_latents_for_guidance,
        model_name=MODEL_NAME,
        model=criteria_pipe
    )
    # Iterate over the rows of the dataframe
    for index, row in df.iterrows():
        if index >= args.num_samples:
            break
        # if index < 60:
        #     continue
        # Extract row data
        # args.sigma_text = 0.2 / (index + 1) ** (1 / 3)
        # print(args.sigma_text)
        prompt = row['prompt']
        obj = row.get('object', 'no_object')
        guidance_type = row.get('guidance_type', args.guidance_type)
        if guidance_type != 'vscore_clip':
            args.algorithm = guidance_type
        region = row.get('region', 'no_region')
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
            print('---------------SPARKE bank is initiated!-----------')

        # Other parameters
        seed = row.get('seed', 0)
        vscore_scale = row.get('vscore_scale', args.vscore_scale)
        cfg = row.get('cfg', args.guidance_scale)
        num_inference_steps = row.get('num_inference_steps', args.num_inference_steps)
        kwargs = {}
        if 'controlnet' in MODEL_NAME:
            # negative_prompt = "poorly drawn, ugly, deformed, blurry, cartoon, anime, low-resolution, complex background"
            prompt = prompt + ", photorealistic, 8K, Canon EOS R5, ultra-detailed skin texture. No background, portrait"
            # kwargs['negative_prompt'] = negative_prompt
            input_image = load_image(row.get('pose_path'))
            pose_image = openpose(input_image)
            kwargs['image'] = pose_image

        # Generate image path
        image_path = get_image_path(csv_path, img_id, prompt, args.algorithm, vscore_scale, guidance_freq, cfg, object_id=obj, seed=row.get('seed', index))
        bank_path = None
        if args.load_bank_dir is not None:
            bank_path = get_image_path(csv_path, img_id, prompt, args.algorithm, vscore_scale, guidance_freq, cfg, object_id=obj, seed=row.get('seed', index), bank_dir=args.load_bank_dir)

        # Check if the image already exists
        if not os.path.exists(image_path) and (not args.load_bank_dir or not os.path.exists(bank_path)):
            logger.info(f'Generating img: {image_path.split("/")[-1]}')
            generator = torch.Generator(device='cuda').manual_seed(seed)  # TODO set to zero for figure one
            import time
            t1 = time.time()
            torch.cuda.empty_cache()
            negative_prompt = "poorly drawn, ugly, deformed, blurry, cartoon, anime, low-resolution, black and white"
            prompt = prompt + ", photorealistic, ultra-detailed skin texture." # , 8K, Canon EOS R5
            # Generate image using the pipeline
            if guidance_type in ['clip_entropy', 'clip_loss', 'vscore_clip']:
                out = criteria_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=cfg,
                    criteria=guidance_type,
                    algorithm=args.algorithm,
                    # height=1024,
                    # width=1024,
                    criteria_guidance_scale=vscore_scale,  # check it
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                    clip_for_guidance=clip_for_guidance,  # does not exist
                    use_latents_for_guidance=args.use_latents_for_guidance,
                    regularize=args.regularize,
                    regularize_weight=args.regularize_weight,
                    kernel=args.kernel,
                    sigma_image=args.sigma_image,  # (sigma_image, sigma_text)
                    sigma_text=args.sigma_text,
                    guidance_freq=guidance_freq,
                    region=region,
                    F_M=F_M,
                    F_T=F_T,
                    F_M_real=F_M_real,
                    F_T_real=F_T_real,
                    beta=beta,
                    regions_list=regions_list,
                    logger_=logger,
                    return_kernels=True,
                    rke_guided_sampler=rke_guided_sampler,
                    **kwargs
                )
                image = out[0][0]
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
                image = image_to_pil(image_path) if bank_path is None else image_to_pil(bank_path)
                if args.use_latents_for_guidance is True:
                    raise Exception("Use latents is True and add_prev_to_bank is True! (Not implemented)")
                features_ = clip_for_guidance.encode_image(image.cuda().unsqueeze(0)).to(torch.float32)
                features_ = features_ / features_.norm(2, dim=1, keepdim=True)
                if F_M is None:
                    F_M = [features_.detach(), torch.mm(features_, features_.T)]
                else:
                    F_M = get_F_M(M=F_M[1], F=F_M[0], f=features_.detach(), kernel=args.kernel, sigma=args.sigma_image)

                if 'cond' in args.algorithm:
                    features_text = clip_for_guidance.encode_text(
                        clip.tokenize([prompt] * features_.shape[0]).to("cuda")).to(torch.float32)
                    features_text = features_text / features_text.norm(2, dim=1, keepdim=True)
                    if F_T is None:
                        F_T = [features_text.detach(), torch.mm(features_text, features_text.T)]
                    else:
                        F_T = get_F_M(M=F_T[1], F=F_T[0], f=features_text.detach(), kernel=args.kernel, sigma=args.sigma_text)

        torch.cuda.empty_cache()
        print(index)
        if index > 2 and (index+1) % 100 == 0 and args.vscore_scale != 0:
            with torch.no_grad():
                for kernel_function in ['cosine', 'gaussian']:
                    k_x = eval_model.cosine_kernel(F_M[0], batchsize=32) if kernel_function == 'cosine' else eval_model.gaussian_kernel(F_M[0], sigma=args.sigma_image)
                    if not 'cond' in args.algorithm:  # TODO fix it (buggy when it is not conditional)
                        F_T = F_M
                    k_t = eval_model.cosine_kernel(F_T[0], batchsize=32) if kernel_function == 'cosine' else eval_model.gaussian_kernel(F_T[0], sigma=args.sigma_text)
                    print(k_t.shape)
                    print(k_x.shape)
                    cond_vendi, _, joint_vendi, vendi, vendi_text = eval_model.conditional_entropy(k_x, k_t, order=1, compute_kernel=False)
                    cond_rke, _, joint_rke, rke, rke_text = eval_model.conditional_entropy(k_x, k_t, order=2, compute_kernel=False)
                    logger.info(f'{kernel_function}: cond_vendi: {torch.exp(cond_vendi).item() if "cond" in args.algorithm else "not defined"}, joint_vendi: {torch.exp(joint_vendi).item()}, vendi: {torch.exp(vendi).item()}, vendi-text: {torch.exp(vendi_text).item()}')
                    logger.info(f'{kernel_function}: cond_rke: {torch.exp(cond_rke).item() if "cond" in args.algorithm else "not defined"}, joint_rke: {torch.exp(joint_rke).item()}, rke: {torch.exp(rke).item()}, rke-text: {torch.exp(rke_text).item()}')
        torch.cuda.empty_cache()
    if F_T is not None:
        np.savez(f"{folder_name}/{args.algorithm}_{args.vscore_scale}_{args.guidance_freq}_txt_feats.npz", txt_feats=F_T[0].detach().cpu().numpy())
    if F_M is not None:
        np.savez(f"{folder_name}/{args.algorithm}_{args.vscore_scale}_{args.guidance_freq}_img_feats.npz", img_feats=F_M[0].detach().cpu().numpy())


def main():
    """Parse arguments and start processing."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default=f'/home/student/Documents/evaluation/rke_guidance/prompts/txt_prompts.txt',
                        help='root dir that contains all the csv files of samples to be generated.')
    parser.add_argument('--num_samples', type=int, default=100_000, help='number of samples to generate.')
    parser.add_argument('--chunk_size', type=int, default=1_000_000, help='the number of images passed to each gpu.')
    parser.add_argument('--device', type=str, default=None, help='device to run the pipeline on.')
    parser.add_argument('--model_name', type=str, default='sdxl',
                        choices=['sdv1.5', 'sdv2.1', 'sdxl', 'controlnet_pose_sdv1.5', 'controlnet_pose_sdxl',
                                 'pixart-sigma', 'kandinsky-3', 'playgroundv2.5', 'dit-xl-2'])
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--guidance_type', type=str, choices=['vscore_clip'], default='vscore_clip')
    parser.add_argument('--algorithm', type=str, default='cond-rke', choices=['vendi', 'rke', 'cond-rke', 'cond-vendi'],
                        help="Choose the algorithm: 'vendi', 'rke', or 'cond-rke'. Default is 'cond-rke'.")
    parser.add_argument('--guidance_freq', type=int, default=10)
    parser.add_argument('--contextual_weight', type=float, default=0)
    parser.add_argument('--vscore_scale', type=float, default=0.03)
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='CFG scale for the guidance.')
    parser.add_argument('--num_inference_steps', type=int, default=51)

    parser.add_argument('--initial_bank_per_object', type=bool, default=False)
    parser.add_argument('--load_bank_dir', type=str, help='Path to load previously generated images in history bank.', default=None)
    parser.add_argument('--load_bank_npz', type=str, help='Path to load previously generated images and text (NPZ).', default=None)
    parser.add_argument('--max_bank_size', type=int, help='Maximum number of images to store in the bank.', default=4000)
    parser.add_argument('--add_prev_to_bank', type=bool, help='Load previously generated images to bank', default=False)
    parser.add_argument('--use_latents_for_guidance', type=bool, help='Use latents for guidance or CLIP features', default=True)
    parser.add_argument('--regularize', type=bool, help='Regularize the image generation process', default=False)
    parser.add_argument('--regularize_weight', type=float, help='Regularization weight', default=0.0001)
    parser.add_argument('--kernel', type=str, help='Kernel type for the guidance', default='gaussian', choices=['cosine', 'gaussian'])
    parser.add_argument('--sigma_image', type=float, help='Sigma for the gaussian kernel of image', default=0.6)
    parser.add_argument('--sigma_text', type=float, help='Sigma for the gaussian kernel of text', default=0.3)

    args = parser.parse_args()

    if args.device is not None:
        torch.cuda.set_device(args.device)
    
    global MODEL_NAME
    MODEL_NAME = args.model_name

    torch_dtype = torch.float16
    feature_extractor = CLIPFeatureExtractor.from_pretrained(settings.CLIP_Feature_Extractor_PATH,
                                                             proxies=settings.proxies)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch_dtype,
                                           proxies=settings.proxies)
    clip_for_guidance, pre_process_clip = clip.load("ViT-B/32", device='cuda')

    # if args.chunk_size % 180 != 0 and args.dataset == 'geode':
    #     print('chunk_size has to be divisible by 180 (count_per_obj)!')
    #     raise NotImplementedError

    # Setup distributed environment only if running with multiple nodes
    if dist.is_available() and dist.is_initialized():
        setup()

    root_dir = args.root_dir
    list_dfs_n_csv_path = []
    if root_dir.endswith('.csv'):
        df = pd.read_csv(root_dir)
        df.reset_index(drop=False, inplace=True)
        for chunk in range(0, len(df), args.chunk_size):
            list_dfs_n_csv_path.append((df.iloc[chunk:chunk + args.chunk_size], root_dir))
    elif root_dir.endswith('.txt'):
        with open(root_dir, 'r') as file:
            lines = file.readlines()
        df = pd.DataFrame(lines, columns=['prompt'])
        df['prompt'] = df['prompt'].str.strip()
        list_dfs_n_csv_path.append((df, root_dir))
    else:
        for file in os.listdir(root_dir):
            if '.csv' in file:
                csv_path = os.path.join(root_dir, file)
                df = pd.read_csv(csv_path)
                for chunk in range(0, len(df), args.chunk_size):
                    list_dfs_n_csv_path.append((df.iloc[chunk:chunk + args.chunk_size], csv_path))

    # Run on the local machine or use distributed training
    for dfs_n_csv_path in list_dfs_n_csv_path:
        global folder_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        folder_name = dfs_n_csv_path[1].split('.csv')[0] + f'_{MODEL_NAME}_' + f'{args.algorithm}_{args.vscore_scale}_{args.guidance_scale}_{timestamp}_simga={(args.sigma_image, args.sigma_text)}'
        folder_name = vendi_dir + '/' + dfs_n_csv_path[1].split('.csv')[0].split('/')[-1] + '/' + f'{MODEL_NAME}_' + f'{args.algorithm}_{args.vscore_scale}_{args.guidance_scale}_{timestamp}_simga={(args.sigma_image, args.sigma_text)}'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        global logger
        if logger is None:
            logger = make_logger(folder_name, 'logs')
            logger.info("Parsed arguments:")
            for arg in vars(args):
                logger.info(f"{arg}: {getattr(args, arg)}")

        generate(dfs_n_csv_path, clip_for_guidance, pre_process_clip, clip_model, feature_extractor,
                 args.dataset, args)

    # Cleanup distributed environment
    if dist.is_available() and dist.is_initialized():
        cleanup()


if __name__ == "__main__":
    main()
