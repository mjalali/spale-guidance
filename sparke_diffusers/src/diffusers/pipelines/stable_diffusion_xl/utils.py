import os

from PIL import Image
import random
import csv
import pickle
import torch
import pandas as pd
from torchvision.transforms import transforms

# import settings
import clip
import numpy as np


def get_F_M(self, M, F, f, kernel='cosine', sigma=None, max_bank_size=100_000):
    """
        Vendi / RKE guidance
        Add the new feature to the feature matrix and calculate the new covariance matrix
    """
    if kernel == 'gaussian' and sigma is None:
        raise ValueError("Gaussian Kernel requires kernel bandwidth `sigma`.")
    if isinstance(sigma, tuple):
        print("Sigma should be float but passed tuple. You should specify it is for image or for text.")
    F_ = torch.cat((F, f))

    if kernel == 'cosine':
        m = torch.mm(F_, f.T)
    elif kernel == 'gaussian':
        squared_diffs = (F_ - f) ** 2
        dist_sq = torch.sum(squared_diffs, dim=1, keepdim=True)
        m = torch.exp(-dist_sq / (2 * sigma ** 2))

    M_ = torch.cat((M, m[:-1].T))
    M_ = torch.cat((M_, m), dim=1)

    if M_.size(0) > max_bank_size:
        M_ = M_[-max_bank_size:, -max_bank_size:]

    return F_, M_


DATASET_DIR = ''
REAL_HOLD_NOT_IN_EVAL_PATH = ''

def precompute_F_M_F_T_real(pre_process_clip, clip_for_guidance, dataset, num_real_samples):
    # precomputations of the VS with respect to examplar images.
    F_M_real_all = {}
    if dataset == 'dollarstreet':
        with open('dollar_street_train.pkl', 'rb') as f:
            dollar_street_train = pickle.load(f)
        all_objects_ds = list(set(dollar_street_train['object_reformatted'].tolist()))
        for obj in all_objects_ds:
            selected_samples_per_obj_list = dollar_street_train[dollar_street_train['object_reformatted'] == obj]['file_path'].tolist()
            root_dir = 'PATH_TO_DOLLARSTREET'
            F_M_real = None
            for obj_path in selected_samples_per_obj_list:
                img_path = os.path.join(root_dir,obj_path)
                image = pre_process_clip(Image.open(img_path)).unsqueeze(0).to('cuda')
                features_ = clip_for_guidance.encode_image(image).to(torch.float32)
                features_ = features_ / features_.norm(2, dim=1, keepdim=True)
                if F_M_real is None:
                    F_M_real = [features_.detach(), torch.mm(features_, features_.T)]
                else:
                    F_M_real = get_F_M(M=F_M_real[1], F=F_M_real[0], f=features_.detach())
            assert F_M_real[1].shape[0] == 4
            assert F_M_real[1].shape[1] == 4
            assert F_M_real[0].shape[0] == 4
            assert F_M_real[0].shape[1] == 512
            F_M_real_all[obj] = (F_M_real, selected_samples_per_obj_list)
            print(f'obj {obj} done!')

    if dataset == 'geode':
        root_dir = DATASET_DIR
        real_hold_not_in_eval_sub = pd.read_csv(REAL_HOLD_NOT_IN_EVAL_PATH)
        # real_hold_not_in_eval_sub.csv is a csv of the images that are not used in evaluation
        # for quick test of the code you can use the whole dataset but for general use of the code
        # you will need to define an evaluation set.


        all_objects = list(set(real_hold_not_in_eval_sub['object'].tolist()))
        for obj in all_objects:
            grouped = real_hold_not_in_eval_sub.groupby('object')
            # change random_state to get a new set of samples
            # change n to get a different sample size
            selected_samples = grouped.apply(lambda x: x.sample(n=num_real_samples, random_state=10))
            selected_samples_per_obj_list = selected_samples[selected_samples['object'] == obj][['file_path', 'region']].apply(tuple, axis=1).tolist()
            F_M_real, F_T_real = None, None
            for obj_path, region in selected_samples_per_obj_list:
                img_path = os.path.join(root_dir, 'images',obj_path)
                image = pre_process_clip(Image.open(img_path)).unsqueeze(0).to('cuda')
                features_ = clip_for_guidance.encode_image(image).to(torch.float32)
                features_text = clip_for_guidance.encode_text(
                    clip.tokenize([f'{obj} in {region}'] * features_.shape[0]).cuda()
                ).to(torch.float32)
                features_ = features_ / features_.norm(2, dim=1, keepdim=True)
                features_text = features_text / features_text.norm(2, dim=1, keepdim=True)
                if F_M_real is None or F_T_real is None:
                    F_M_real = [features_.detach(), torch.mm(features_, features_.T)]
                    F_T_real = [features_text.detach(), torch.mm(features_text, features_text.T)]
                else:
                    F_M_real = get_F_M(M=F_M_real[1], F=F_M_real[0], f=features_.detach())
                    F_T_real = get_F_M(M=F_T_real[1], F=F_T_real[0], f=features_text.detach())

            assert F_M_real[1].shape[0] == num_real_samples
            assert F_M_real[1].shape[1] == num_real_samples
            assert F_M_real[0].shape[0] == num_real_samples
            assert F_M_real[0].shape[1] == 512
            F_M_real_all[obj] = (F_M_real, F_T_real, selected_samples_per_obj_list)
            print(f'obj {obj} done!')

    with open(f'F_M_real_all_{dataset}_{num_real_samples}.pkl', 'wb') as f:
        pickle.dump(F_M_real_all, f)
    print(f'F_M_real_all_{dataset}_{num_real_samples}.pkl saved!')


def cosine_kernel(x, y=None, batchsize=64, normalize=True, device="cuda"):
    '''
    Calculate the cosine similarity kernel matrix. The shape of x and y should be equal except for the batch dimension.

    x:
        Input tensor, dim: [batch, dims]
    y:
        Input tensor, dim: [batch, dims]. If y is `None`, then y = x and it will compute cosine similarity k(x, x).
    batchsize:
        Batchify the formation of the kernel matrix, trade time for memory.
        batchsize should be smaller than the length of data.

    return:
        Scalar: Mean of cosine similarity values
    '''
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
        y = x if y is None else torch.from_numpy(y).to(device)
    else:
        x = x.to(device)
        y = x if y is None else y.to(device)

    batch_num = (y.shape[0] // batchsize) + 1
    assert (x.shape[1:] == y.shape[1:])

    total_res = torch.zeros((x.shape[0], 0), device=x.device)
    for batchidx in range(batch_num):
        y_slice = y[batchidx * batchsize:min((batchidx + 1) * batchsize, y.shape[0])]

        # Normalize x and y_slice
        x_norm = x / x.norm(dim=1, keepdim=True)
        y_norm = y_slice / y_slice.norm(dim=1, keepdim=True)

        # Calculate cosine similarity
        res = torch.mm(x_norm, y_norm.T)

        total_res = torch.hstack([total_res, res])

        del res, y_slice

    if normalize is True:
        total_res = total_res / (x.shape[0] * y.shape[0])

    return total_res

def image_to_pil(image_path):
    preprocess = transforms.Compose(
        [
            transforms.Resize(
                224, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    with Image.open(image_path).convert("RGB") as img:
        img = preprocess(img)
        return img

def get_F_M(M, F, f):
    """
        Add the new feature to the feature matrix and calculate the new covariance matrix
    """
    F_ = torch.cat((F, f))
    m = torch.mm(F_, f.T)
    M_ = torch.cat((M, m[:-1].T))
    M_ = torch.cat((M_, m), dim=1)
    return F_, M_
