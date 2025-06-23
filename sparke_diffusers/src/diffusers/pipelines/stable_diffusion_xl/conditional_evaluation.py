import numpy as np
import functools
import random

from sympy.abc import omega
from tqdm import tqdm
import torch
import inspect
from log import make_logger

logger = make_logger('logs','conditional-eval')


def gaussian_kernel_decorator(function):
    def wrap_kernel(self, *args, **kwargs):
        # Get the function's signature
        sig = inspect.signature(function)
        params = list(sig.parameters.keys())
        
        # Determine if `compute_kernel`, `algorithm`, and `sigma` parameter is in args or kwargs
        bound_args = sig.bind_partial(*args, **kwargs).arguments
        compute_kernel = bound_args.get('compute_kernel', True)
        algorithm = bound_args.get('algorithm', 'kernel')

        sigma = bound_args.get('sigma', None)
        sigma_x, sigma_y = None, None
        if type(sigma) == tuple:
            sigma_x, sigma_y == sigma[0], sigma[1]
        elif isinstance(sigma, (int, float)):
            sigma_x = sigma_y = sigma  # TODO check other types in the future
        else:
            sigma_x, sigma_y = self.sigma

        if compute_kernel is True and algorithm == 'kernel':
            args = list(args)  # To be able to edit args
            if 'X' in params:
                index = params.index('X') - 1
                args[index] = self.gaussian_kernel(args[index], sigma=sigma_x)  # TODO: this is buggy

            if 'Y' in params:
                index = params.index('Y') - 1
                if args[index] is not None:
                    args[index] = self.gaussian_kernel(args[index], sigma=sigma_y)

        return function(self, *args, **kwargs)

    return wrap_kernel


def entropy_q(p, q=1, log_base='e'):
    # TODO check if it should be log2 or log
    log = torch.log if log_base == 'e' else torch.log2
    p_ = p[p > 0]
    if q == 1:
        return -(p_ * log(p_)).sum()
    if q == "inf":
        return -log(torch.max(p))
    return log((p_ ** q).sum()) / (1 - q)


def cov_rff2(x, feature_dim, std, batchsize=16, presign_omeaga=None, normalise = True):
    assert len(x.shape) == 2  # [B, dim]

    x_dim = x.shape[-1]

    if presign_omeaga is None:
        omegas = torch.randn((x_dim, feature_dim), device=x.device) * (1 / std)
    else:
        omegas = presign_omeaga
    product = torch.matmul(x, omegas)
    batched_rff_cos = torch.cos(product)  # [B, feature_dim]
    batched_rff_sin = torch.sin(product)  # [B, feature_dim]

    batched_rff = torch.cat([batched_rff_cos, batched_rff_sin], dim=1) / np.sqrt(feature_dim)  # [B, 2 * feature_dim]

    batched_rff = batched_rff.unsqueeze(2)  # [B, 2 * feature_dim, 1]

    cov = torch.zeros((2 * feature_dim, 2 * feature_dim), device=x.device)
    batch_num = (x.shape[0] // batchsize) + 1
    i = 0
    for batchidx in tqdm(range(batch_num)):
        batched_rff_slice = batched_rff[batchidx * batchsize:min((batchidx + 1) * batchsize,
                                                                 batched_rff.shape[0])]  # [mini_B, 2 * feature_dim, 1]
        cov += torch.bmm(batched_rff_slice, batched_rff_slice.transpose(1, 2)).sum(dim=0)
        i += batched_rff_slice.shape[0]
    cov /= x.shape[0]
    assert i == x.shape[0]

    assert cov.shape[0] == cov.shape[1] == feature_dim * 2

    return cov, batched_rff.squeeze()


def cov_rff2_joint(x, feature_dim, std, batchsize=16, presign_omeaga=None, normalise = True):
    assert len(x.shape) == 2 # [B, dim]

    x_dim = x.shape[-1]

    if presign_omeaga is None:
        omegas = torch.randn((x_dim, feature_dim), device=x.device) * (1 / std)
    else:
        omegas = presign_omeaga
    product = torch.matmul(x, omegas)
    batched_rff_cos = torch.cos(product) # [B, feature_dim]
    batched_rff_sin = torch.sin(product) # [B, feature_dim]
    y = x.clone()
    y[:, 780:] = -y[:, 780:].clone()
    product = torch.matmul(y, omegas)
    batched_rff_cos_negative = torch.cos(product) # [B, feature_dim]
    batched_rff_sin_negative = torch.sin(product) # [B, feature_dim]


    batched_rff = torch.cat([batched_rff_cos, batched_rff_sin, batched_rff_cos_negative, batched_rff_sin_negative], dim=1) / (np.sqrt(2) * np.sqrt(feature_dim)) # [B, 2 * feature_dim]

    batched_rff = batched_rff.unsqueeze(2) # [B, 2 * feature_dim, 1]

    cov = torch.zeros((4 * feature_dim, 4 * feature_dim), device=x.device)
    batch_num = (x.shape[0] // batchsize) + 1
    i = 0
    for batchidx in tqdm(range(batch_num)):
        batched_rff_slice = batched_rff[batchidx*batchsize:min((batchidx+1)*batchsize, batched_rff.shape[0])] # [mini_B, 2 * feature_dim, 1]
        cov += torch.bmm(batched_rff_slice, batched_rff_slice.transpose(1, 2)).sum(dim=0)
        i += batched_rff_slice.shape[0]
    cov /= x.shape[0]
    assert i == x.shape[0]

    assert cov.shape[0] == cov.shape[1] == feature_dim * 4

    return cov, batched_rff.squeeze()


def cov_rff2_joint_v2(x, feature_dim, std, batchsize=16, presign_omeaga=None, normalise = True):
    assert len(x.shape) == 2 # [B, dim]

    x_dim = x.shape[-1]

    if presign_omeaga is None:
        omegas = torch.randn((x_dim, feature_dim), device=x.device) * (1 / std)
    else:
        omegas = presign_omeaga
    omegas_img, omegas_txt = omegas[:780], omegas[780:]
    img, txt = x[:, :780], x[:, 780:]
    product_img = img @ omegas_img
    product_txt = txt @ omegas_txt
    batched_rff_cos = torch.cos(img @ omegas_img + txt @ omegas_txt) # [B, feature_dim]
    batched_rff_sin = torch.sin(img @ omegas_img + txt @ omegas_txt) # [B, feature_dim]
    batched_rff_cos_negative = torch.cos(img @ omegas_img - txt @ omegas_txt) # [B, feature_dim]
    batched_rff_sin_negative = torch.sin(img @ omegas_img - txt @ omegas_txt) # [B, feature_dim]

    batched_rff = torch.cat([torch.cos(product_img) * torch.cos(product_txt),
                            torch.cos(product_img) * torch.sin(product_txt),
                            torch.sin(product_img) * torch.cos(product_txt),
                            torch.sin(product_img) * torch.sin(product_txt)], dim=1) / np.sqrt(feature_dim) # [B, 4 * feature_dim]

    # batched_rff = torch.cat([batched_rff_cos, batched_rff_sin, batched_rff_cos_negative, batched_rff_sin_negative], dim=1) / (np.sqrt(2) * np.sqrt(feature_dim)) # [B, 4 * feature_dim]

    batched_rff = batched_rff.unsqueeze(2) # [B, 2 * feature_dim, 1]

    cov = torch.zeros((4 * feature_dim, 4 * feature_dim), device=x.device)
    batch_num = (x.shape[0] // batchsize) + 1
    i = 0
    for batchidx in tqdm(range(batch_num)):
        batched_rff_slice = batched_rff[batchidx*batchsize:min((batchidx+1)*batchsize, batched_rff.shape[0])] # [mini_B, 2 * feature_dim, 1]
        cov += torch.bmm(batched_rff_slice, batched_rff_slice.transpose(1, 2)).sum(dim=0)
        i += batched_rff_slice.shape[0]
    cov /= x.shape[0]
    assert i == x.shape[0]

    assert cov.shape[0] == cov.shape[1] == feature_dim * 4

    return cov, batched_rff.squeeze()


def cov_diff_rff(x, y, feature_dim, std, batchsize=16):
    assert len(x.shape) == len(y.shape) == 2 # [B, dim]

    B, D = x.shape
    x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
    y = y.to('cuda' if torch.cuda.is_available() else 'cpu')

    omegas = torch.randn((D, feature_dim), device=x.device) * (1 / std)

    x_cov, x_feature = cov_rff2(x, feature_dim, std, batchsize=batchsize, presign_omeaga=omegas)
    y_cov, y_feature = cov_rff2(y, feature_dim, std, batchsize=batchsize, presign_omeaga=omegas)

    return x_cov, y_cov, omegas, x_feature, y_feature # [2 * feature_dim, 2 * feature_dim], [D, feature_dim], [B, 2 * feature_dim], [B, 2 * feature_dim]

def cov_rff(x, feature_dim, std, batchsize=16, normalise=True):
    assert len(x.shape) == 2 # [B, dim]

    x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
    B, D = x.shape
    omegas = torch.randn((D, feature_dim), device=x.device) * (1 / std)

    x_cov, x_feature = cov_rff2(x, feature_dim, std, batchsize=batchsize, presign_omeaga=omegas, normalise=normalise)

    return x_cov, omegas, x_feature # [2 * feature_dim, 2 * feature_dim], [D, feature_dim], [B, 2 * feature_dim]


def joint_cov_rff(x, y, feature_dim, std_x, std_y, batchsize=16, normalise=True, omegas_x=None, omegas_y=None):
    assert len(x.shape) == 2 # [B, dim]

    x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
    y = y.to('cuda' if torch.cuda.is_available() else 'cpu')
    B, D_x = x.shape
    _, D_y = y.shape

    if omegas_x is None or omegas_y is None:
        omegas = torch.randn((D_x + D_y, feature_dim), device=x.device)
        omegas_x = omegas[:D_x] * (1 / std_x)
        omegas_y = omegas[D_x:] * (1 / std_y)
    else:
        omegas = torch.cat([omegas_x, omegas_y])

    x_cov, x_feature = cov_rff2(
        torch.cat([x, y], dim=1),
        feature_dim,
        std=None,
        batchsize=batchsize,
        presign_omeaga=torch.cat([omegas_x, omegas_y]),
        normalise=normalise
    )

    return x_cov, omegas_x, omegas_y, x_feature # [2 * feature_dim, 2 * feature_dim], [D, feature_dim], [B, 2 * feature_dim]


class ConditionalEvaluation:
    def __init__(self, similarity_function=None, sigma=None):
        if similarity_function is None and sigma is None:
            raise ValueError("Both similarity_function and sigma can not be None")
        self.similarity = similarity_function
        self.sigma = sigma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # RFF Conditional-Vendi
        self.omegas_x = None
        self.omegas_y = None

    def gaussian_kernel(self, x, y=None, sigma=None, batchsize=32, normalize=True):
        '''
        calculate the kernel matrix, the shape of x and y should be equal except for the batch dimension

        x:
            input, dim: [batch, dims]
        y:
            input, dim: [batch, dims], If y is `None` then y = x and it will compute k(x, x).
        sigma:
            bandwidth parameter
        batchsize:
            Batchify the formation of kernel matrix, trade time for memory
            batchsize should be smaller than length of data

        return:
            scalar : mean of kernel values
        '''
        if sigma is None and type(self.sigma) == tuple:
            raise ValueError("`sigma` is None, while `self.sigma` is a tuple, causing ambiguity.")

        if sigma is None:
            sigma = self.sigma

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)
            y = x if y is None else torch.from_numpy(y).to(self.device)
        else:
            x = x.to(self.device)
            y = x if y is None else y.to(self.device)

        batch_num = (y.shape[0] // batchsize) + 1
        assert (x.shape[1:] == y.shape[1:])

        total_res = torch.zeros((x.shape[0], 0), device=x.device)
        for batchidx in range(batch_num):
            y_slice = y[batchidx*batchsize:min((batchidx+1)*batchsize, y.shape[0])]
            res = torch.norm(x.unsqueeze(1)-y_slice, dim=2, p=2).pow(2)
            res = torch.exp((- 1 / (2*sigma*sigma)) * res)
            total_res = torch.hstack([total_res, res])

            del res, y_slice

        if normalize is True:
            total_res = total_res / np.sqrt(x.shape[0] * y.shape[0])

        return total_res

    def cosine_kernel(self, x, y=None, batchsize=256, normalize=True):
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
            x = torch.from_numpy(x).to(self.device)
            y = x if y is None else torch.from_numpy(y).to(self.device)
        else:
            x = x.to(self.device)
            y = x if y is None else y.to(self.device)

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
            total_res = total_res / np.sqrt(x.shape[0] * y.shape[0])

        return total_res

    @gaussian_kernel_decorator
    def compute_entropy(self, X, order, compute_kernel=True, algorithm='kernel', omegas=None, sigma=None,
                        feature_dim=3000):

        if algorithm == 'kernel':
            assert X.shape[0] == X.shape[1]
            if order == 2:
                frobenius_norm_squared = torch.linalg.norm(X, 'fro') ** 2
                trace_X_squared = frobenius_norm_squared
                # Calculate S_2
                return -torch.log(trace_X_squared)  # TODO check if it should be log2 or log
            elif order in [1, 1.5]:  # TODO check if we want exp or not. also check log or log2!
                vals = torch.linalg.eigvalsh(X)
                return entropy_q(vals, q=order)
            else:
                raise NotImplementedError()
        if algorithm == 'rff':
            assert X.shape[0] != X.shape[1]
            if omegas is None and sigma is None:
                raise ValueError("When using RFF one of omegas or sigma should be passed.")

            cov_x, _ = cov_rff2(X, feature_dim, std=sigma, presign_omeaga=omegas, normalise=True)
            return self.compute_entropy(cov_x, order, compute_kernel=False)

    def compute_joint_entropy(self, X, Y, order, compute_kernel=False, algorithm='kernel', feature_dim=3000):
        if algorithm == 'kernel':
            # assert X.shape[0] == X.shape[1]
            # K_x = self.gaussian_kernel(X, sigma=self.sigma[0], normalize=False)
            # K_y = self.gaussian_kernel(Y, sigma=self.sigma[1], normalize=False)
            if X.shape[0] == X.shape[1]:
                K_x = X
                K_y = Y
            else:
                K_x = self.gaussian_kernel(X, sigma=self.sigma[0], normalize=False)
                K_y = self.gaussian_kernel(Y, sigma=self.sigma[1], normalize=False)
            XoY = K_x * K_y  # Hadamard product
            S_AB = XoY / torch.trace(XoY)
            return self.compute_entropy(X=S_AB, order=order, compute_kernel=False)
        if algorithm == 'rff':
            assert X.shape[0] != X.shape[1]
            sigma_x, sigma_y = self.sigma[0], self.sigma[1] if type(self.sigma) == tuple else (self.sigma, self.sigma)
            cov_joint, self.omegas_x, self.omegas_y, _ = joint_cov_rff(x=X, y=Y, feature_dim=feature_dim, std_x=sigma_x,
                                                                       std_y=sigma_y, batchsize=32, normalise=True,
                                                                       omegas_x=self.omegas_x, omegas_y=self.omegas_x)
            return self.compute_entropy(cov_joint, order, compute_kernel=False)

    def conditional_entropy(self, X, Y,  order, n_samples=10_000, compute_kernel=False, algorithm='kernel', feature_dim=3000):  # H_a(X|Y)
        entropy_joint = self.compute_joint_entropy(X, Y, order=order, compute_kernel=False, algorithm=algorithm, feature_dim=feature_dim)
        entropy_Y = self.compute_entropy(Y, order=order, compute_kernel=compute_kernel, algorithm=algorithm, omegas=self.omegas_y, feature_dim=feature_dim, sigma=self.sigma[1])
        entropy_X = self.compute_entropy(X, order=order, compute_kernel=compute_kernel, algorithm=algorithm, omegas=self.omegas_x, feature_dim=feature_dim, sigma=self.sigma[0])

        if True:
            logger.info(f'joint: {entropy_joint}, X: {entropy_X}, Y: {entropy_Y}')
            logger.info(f'H(X|Y) = {entropy_joint - entropy_Y}, I(X; Y) = {entropy_X + entropy_Y - entropy_joint}')
        return entropy_joint - entropy_Y, entropy_X + entropy_Y - entropy_joint, entropy_joint, entropy_X, entropy_Y  # H(X|Y), I(X, Y), H(X, Y), H(X), H(Y)
        #return entropy_joint - entropy_Y TODO: You should only return this

    @gaussian_kernel_decorator
    def mutual_information_order_2(self, X, Y, order, compute_kernel=True, entropy='shannon'):
        i_xy = self.compute_entropy(X, order=order, compute_kernel=False) + self.compute_entropy(Y, order=order, compute_kernel=False) - self.compute_joint_entropy(X, Y, order=order, compute_kernel=False)
        logger.info(f'I(X, Y) = {i_xy}')
        return i_xy

    # def visulize_modes()
