import math

import clip
import torch
from torch.nn import functional as F
from torchvision import transforms

logger = None
try:
    from .stable_diffusion_xl.log import make_logger
    logger = make_logger('.', __name__)
except:
    import logging
    logger = logging.getLogger(__name__)


class RKEGuidedSampling:
    """
    RKE-guided sampling class for Stable Diffusion.

    Example:
    # Initialize the RKE-guided sampler
    rke_guided_sampler = RKEGuidedSampling(
        algorithm='vendi',
        kernel='cosine',
        sigma_image=0.8,
        sigma_text=0.35,
        use_latents_for_guidance=True
    )

    # Define required inputs
    latents = torch.randn(1, 4, 64, 64)  # Example latent tensor
    timestep = 50
    index = 0
    noise_pred = torch.randn_like(latents)
    extra_step_kwargs = {}
    criteria_guidance_scale = 5.0
    prompt = "A beautiful landscape"
    clip_for_guidance = clip.load("ViT-B/32", device='cuda')[0]
    regularize = False # It is recommended to not regularize
    regularize_weight = 0.0001
    F_M, F_T, F_M_real, F_T_real = None, None, None, None
    beta = 0.5

    # Perform guidance
    grads, F_M, F_T = rke_guided_sampler.cond_fn(
        latents=latents,
        timestep=timestep,
        index=index,
        noise_pred=noise_pred,
        extra_step_kwargs=extra_step_kwargs,
        criteria_guidance_scale=criteria_guidance_scale,
        prompt=prompt,
        clip_for_guidance=clip_for_guidance,
        regularize=regularize,
        regularize_weight=regularize_weight,
        F_M=F_M,
        F_T=F_T,
        F_M_real=F_M_real,
        F_T_real=F_T_real,
        beta=beta
    )

    # Update latents with the computed guidance gradient
    latents = latents + grads

    """

    def __init__(self, algorithm, kernel, sigma_image, sigma_text=None, use_latents_for_guidance=True, max_bank_size=10_000, model_name=None, model=None, **kwargs):
        self.algorithm = algorithm
        self.kernel = kernel
        self.sigma = (sigma_image, sigma_text)
        self.use_latents_for_guidance = use_latents_for_guidance
        if use_latents_for_guidance is False and model_name is None:
            raise ValueError("Model name with model object should be provided if `use_latents_for_guidance` is False.")
        self.model_name = model_name
        self.model = model
        self.F_M = None
        self.F_T = None
        self.max_bank_size = max_bank_size

    @staticmethod
    def clip_process_image(image):
        # apply clip pre-processing
        torch_preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias='warn'),
            transforms.CenterCrop(size=(224, 224)),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)), ])
        image = torch_preprocess(image)
        return image

    def get_image_from_latent(self, latents, noise_pred=None, t=None, update_latents_from_noise=False,
                              postprocess=False, **extra_step_kwargs):
        # TODO this raise memmory error. check it later
        if self.model_name in ['sdv2.1', 'sdv1.5']:
            alpha_prod_t = self.model.scheduler.alphas_cumprod[t]
            beta_prod_t = 1 - alpha_prod_t
            pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

            fac = torch.sqrt(beta_prod_t)
            sample = pred_original_sample * (fac) + latents * (1 - fac)
            sample = 1 / self.model.vae.config.scaling_factor * sample
            image = self.model.vae.decode(sample).sample
            return image
        else:
            raise NotImplementedError()
        """TODO: This function is for SD-XL only! so it should be removed from this class and put in the pipeline class"""
        if update_latents_from_noise is True:
            if noise_pred is None or t is None:
                raise ValueError("If `update_latents_from_noise` is True, `noise_pred` and `t` must be provided.")
            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            self.scheduler._step_index -= 1  # Substract step_index by 1 because of calling twice
            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

        # The same code used in generating image from latents part of the __call__ code
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        if needs_upcasting:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
        elif latents.dtype != self.vae.dtype:
            if torch.backends.mps.is_available():
                # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                self.vae = self.vae.to(latents.dtype)

        # unscale/denormalize the latents
        # denormalize with the mean and std if available and not None
        has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
        has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
        if has_latents_mean and has_latents_std:
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            )
            latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
        else:
            latents = latents / self.vae.config.scaling_factor

        image = self.vae.decode(latents, return_dict=False)[0]
        torch.cuda.empty_cache()
        if postprocess:
            image = self.image_processor.postprocess(image.to("cuda:0"))
        return image

    def get_F_M(self, M, F, f, kernel='cosine', sigma=None, max_bank_size=None):
        """
            Vendi / RKE guidance
            Add the new feature to the feature matrix and calculate the new covariance matrix
        """
        if max_bank_size is None:
            max_bank_size = self.max_bank_size
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
            F_ = F_[-max_bank_size:]

        return F_, M_

    def get_rank(self, M, M_text=None, feature_m=None, feature_t=None, kernel='cosine', sigma_image=None,
                 sigma_text=None, n_samples=50_000, batch_size=256):
        """
        Calculate the diversity term for Diffusion Model guidance.

        :param M: Memory bank matrix for image features.
        :param M_text: Memory bank matrix for text features (optional, used for conditional guidance).
        :param feature_m: Current image feature vector.
        :param feature_t: Current text feature vector (optional, used for conditional guidance).
        :param kernel: Kernel type to use for similarity calculation ('cosine' or 'gaussian').
        :param sigma_image: Bandwidth parameter for the Gaussian kernel (image features).
        :param sigma_text: Bandwidth parameter for the Gaussian kernel (text features, optional).
        :param n_samples: Number of samples to consider for diversity calculation.
        :param batch_size: Batch size for processing samples.
        :return: Diversity term as a scalar value.
        """
        if M.shape[0] % 50 == 0:
            logger.info(f'shape of memory bank: {M.shape}')
        if torch.isnan(M).any() or torch.isinf(M).any():
            logger.warning("M contains NaN or Inf. Fixing values...")
            M = torch.where(torch.isnan(M) | torch.isinf(M), torch.tensor(1e-6, device=M.device, dtype=M.dtype), M)

        if self.algorithm == 'rke':  # RKE
            if M.shape[0] > 20 and feature_m is not None:  # TODO 200 should be a variable
                if kernel == 'cosine':
                    similarities = F.cosine_similarity(feature_m[-1].unsqueeze(0), feature_m) ** 2
                elif kernel == 'gaussian':
                    squared_diffs = (feature_m - feature_m[-1]) ** 2
                    dist_sq = torch.sum(squared_diffs, dim=1)
                    similarities = torch.exp(-dist_sq / (2 * sigma_image ** 2))

                return 1 / similarities.mean()  # exp(Renyi Entropy_2)

            else:
                # Avoid zero-matrix issues
                if torch.norm(M) < 1e-6:
                    M = M + 1e-6  # Small perturbation

                frobenius_norm_squared = torch.linalg.norm(M, 'fro') ** 2
                if frobenius_norm_squared < 1e-6:
                    logger.warning("Frobenius norm is too small. Adjusting to stable value.")
                    frobenius_norm_squared = 1e-6  # Use small value instead of clamping

                return 1 / (frobenius_norm_squared + 1e-8)

        elif self.algorithm == 'cond-rke':
            if M.shape[0] > 10 and feature_m is not None:
                if kernel == 'cosine':
                    similarities = (
                                        F.cosine_similarity(feature_m[-1].unsqueeze(0), feature_m) *
                                        F.cosine_similarity(feature_t[-1].unsqueeze(0), feature_t)
                                   ) ** 2
                elif kernel == 'gaussian':
                    squared_diffs = (feature_m - feature_m[-1]) ** 2
                    dist_sq = torch.sum(squared_diffs, dim=1)
                    similarities = torch.exp(-dist_sq / (2 * sigma_image ** 2))

                    squared_diffs = (feature_t - feature_t[-1]) ** 2
                    dist_sq = torch.sum(squared_diffs, dim=1)
                    similarities = similarities * torch.exp(-dist_sq / (2 * sigma_text ** 2))

                # threshold_index = int(0.25 * len(all_similarities))  # 25 percentile
                # if threshold_index > 0:
                #     threshold = torch.kthvalue(all_similarities, threshold_index).values
                #     all_similarities = torch.where(all_similarities < threshold, torch.tensor(0.0, device='cuda'), all_similarities)
                return 1 / similarities.mean()

            else:
                XoY = M * M_text  # Hadamard product
                S_AB = XoY / torch.trace(XoY)
                frobenius_norm_squared = torch.linalg.norm(S_AB, 'fro') ** 2
                return 1 / frobenius_norm_squared

        elif self.algorithm == 'cond-vendi':
            XoY = M * M_text  # Hadamard product
            S_AB = XoY / torch.trace(XoY)
            U, _, VT = torch.linalg.svd(S_AB)
            S = torch.diag(torch.mm(U.T, torch.mm(S_AB, VT.T)))
            S = S / S.sum()
            entropy = -torch.sum(S * torch.log(S))
            rank = torch.exp(entropy)
            return rank

        elif self.algorithm == 'vendi':
            U, _, VT = torch.linalg.svd(M.to(torch.float32))
            S = torch.diag(torch.mm(U.T, torch.mm(M.to(torch.float32), VT.T)))
            S = S / S.sum()
            entropy = -torch.sum(S * torch.log(S))
            rank = torch.exp(entropy)
            return rank

    def update_feature_matrices(self, feature_m, features_text=None):
        if self.F_M is None:
            self.F_M = [feature_m.detach(), torch.mm(feature_m, feature_m.T)]
            if self.F_T is None and 'cond' in self.algorithm:
                self.F_T = [features_text.detach(), torch.mm(features_text, features_text.T)]
        else:
            # update F if reached clean sample
            self.F_M = self.get_F_M(M=self.F_M[1], F=self.F_M[0], f=feature_m.detach(), kernel=self.kernel, sigma=self.sigma[0])
            if 'cond' in self.algorithm:
                self.F_T = self.get_F_M(M=self.F_T[1], F=self.F_T[0], f=features_text.detach(), kernel=self.kernel,
                                   sigma=self.sigma[1])
        return self.F_M, self.F_T

    @torch.enable_grad()
    def cond_fn(
            self,
            latents,
            timestep,
            index,
            noise_pred,
            extra_step_kwargs,
            criteria_guidance_scale,
            prompt,
            clip_for_guidance,
            regularize,
            regularize_weight,
            F_M,
            F_T,
            F_M_real,
            F_T_real,
            beta,
    ):
        """
        Main function that should be called in `__call__` method of the Diffusion Model pipeline.
        :param latents: The latent representations of the input data.
        :param timestep: The current timestep in the diffusion process.
        :param index: The index of the current step in the sampling process.
        :param noise_pred: The predicted noise for the current timestep.
        :param extra_step_kwargs: Additional arguments for the scheduler step.
        :param criteria_guidance_scale: The scale factor for the guidance criteria.
        :param prompt: The text prompt used for conditional guidance.
        :param clip_for_guidance: The CLIP model used for guidance.
        :param regularize: Whether to apply regularization to the rank calculation.
        :param regularize_weight: The weight of the regularization term.
        :param F_M: The feature matrix for the image guidance.
        :param F_T: The feature matrix for the text guidance.
        :param F_M_real: The real feature matrix for the image guidance.
        :param F_T_real: The real feature matrix for the text guidance.
        :param beta: The weight for the real rank in the guidance calculation.
        :return: A tuple containing the computed gradients, updated image feature matrix (F_M),
         and updated text feature matrix (F_T).
        """
        latents = latents.detach().requires_grad_()

        if self.use_latents_for_guidance is False:
            image = self.get_image_from_latent(
                latents=latents,
                noise_pred=noise_pred,
                t=timestep,
                update_latents_from_noise=True,
                postprocess=False,
                **extra_step_kwargs
            )

            image = self.clip_process_image(image)
            features_ = clip_for_guidance.encode_image(image).to(torch.float32)
        else:
            features_ = latents.view(1, -1)
        features_text = None
        if 'cond' in self.algorithm:
            features_text = clip_for_guidance.encode_text(
                clip.tokenize([prompt] * features_.shape[0]).to(features_.device)).to(torch.float32)
            features_text = features_text / features_text.norm(2, dim=1, keepdim=True)
        features_ = features_ / features_.norm(2, dim=1, keepdim=True)
        rank = 0
        if self.F_M is not None:  # and index < 45:
            # print('only first 40')
            # if index < 5:
            #     criteria_guidance_scale = 0.15
            # if index < 10:
            #     criteria_guidance_scale = 0.08
            # elif index < 20:
            # criteria_guidance_scale = 0.08
            F_, M_ = self.get_F_M(M=self.F_M[1], F=self.F_M[0], f=features_, kernel=self.kernel, sigma=self.sigma[0])
            if 'cond' in self.algorithm:
                F_T_, T_ = self.get_F_M(M=self.F_T[1], F=self.F_T[0], f=features_text, kernel=self.kernel, sigma=self.sigma[1])
                rank_fake = self.get_rank(M=M_ / M_.shape[0], M_text=T_ / T_.shape[0], feature_m=F_, feature_t=F_T_,
                                          kernel=self.kernel, sigma_image=self.sigma[0], sigma_text=self.sigma[1])
            else:
                rank_fake = self.get_rank(M=M_ / M_.shape[0], feature_m=F_, kernel=self.kernel, sigma_image=self.sigma[0])

            if beta != 0 and F_M_real is not None:
                F_real_, M_real_ = self.get_F_M(M=F_M_real[1], F=F_M_real[0], f=features_, kernel=self.kernel,
                                                sigma=self.sigma[0])
                if 'cond' in self.algorithm:
                    F_T_real_, T_real_ = self.get_F_M(M=F_T_real[1], F=F_T_real[0], f=features_text, kernel=self.kernel,
                                                      sigma=self.sigma[1])
                    rank_real = self.get_rank(M_real_ / M_real_.shape[0], M_text=T_real_ / T_real_.shape[0],
                                              feature_m=F_real_, feature_t=F_T_real_, kernel=self.kernel,
                                              sigma_image=self.sigma[0], sigma_text=self.sigma[1])
                else:
                    rank_real = self.get_rank(M_real_ / M_real_.shape[0], feature_m=F_real_, kernel=self.kernel,
                                              sigma_image=self.sigma[0])
            else:
                rank_real = 0.0

            rank = rank_fake - beta * rank_real
            if regularize is True:
                # Code borrowed from ReNO paper
                latent_norm = torch.linalg.vector_norm(latents).to(torch.float32)
                log_norm = torch.log(latent_norm)
                latent_dim = math.prod(latents.shape[1:])
                regularization = regularize_weight * (
                        - 0.5 * latent_norm ** 2 + (latent_dim - 1) * log_norm
                )
                rank += regularization.to(rank.dtype)
                print(f'hi regularization is {regularization}')

            if rank.isnan() or rank.isinf():
                logger.warning(f"{index}: Rank is NaN or Inf")
            grads = torch.autograd.grad(rank, latents)[0]
            if self.use_latents_for_guidance is False:
                del image
            torch.cuda.empty_cache()
            if grads.norm(2) > 1e-8:  # Adjust the threshold as necessary
                grads_same_scale = grads / grads.norm(2) * latents.norm(2).detach()
            else:
                logger.warning(f"{index}: Gradient norm is too small. Skipping gradient update.")
                grads_same_scale = grads
            grads = grads_same_scale * criteria_guidance_scale

        else:
            grads = torch.tensor(0.0).to(latents.device)
        # TODO: fix this bug: if if i % guidance_freq != 0 when timestep.item() ==1 it never comes here and F_M is always None
        if timestep.item() in [20, 1]:  # TODO: this is bug when samples are already generated or not!!!!!, this is but to compare with 82, but check it later, 20 for sd and 1 for sdxl
            F_M, F_T = self.update_feature_matrices(features_, features_text)

        return grads, F_M, F_T
