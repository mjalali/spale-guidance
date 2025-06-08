# SPALE Diffusers

**Scalable Prompt-Aware Latent Entropy-Based Diversity Guidance in Diffusion Models**

---

## Overview

This repository contains the official implementation of **SPALE**, a method for improving diversity in prompt-guided diffusion models using **prompt-aware latent entropy-based guidance**. SPALE introduces conditional entropy-guided sampling that dynamically adapts to semantically similar prompts and supports scalable generation across modern text-to-image architectures.

> Project Webpage: [https://mjalali.github.io/SPALE](https://mjalali.github.io/SPALE)

---

## Abstract

Diffusion models have demonstrated exceptional performance in high-fidelity image synthesis and prompt-based generation. However, achieving sufficient diversity—particularly within semantically similar prompts—remains a critical challenge. Prior methods use diversity metrics as guidance signals, but often neglect prompt awareness or computational scalability.

In this work, we propose **SPALE**: _Scalable Prompt-Aware Latent Entropy-based Diversity Guidance_. SPALE leverages **conditional entropy** to guide the sampling process with respect to prompt-localized diversity. By employing **Conditional Latent RKE Score Guidance**, we reduce the computational complexity from $\mathcal{O}(n^3)$ to $\mathcal{O}(n)$, enabling efficient large-scale generation. We integrate SPALE into several popular diffusion pipelines and demonstrate improved diversity without additional inference overhead.

---

## Supported Pipelines

The following `diffusers` pipelines have been extended with SPALE guidance:

| Pipeline Type                             | Implementation                                  |
|------------------------------------------|-------------------------------------------------|
| Stable Diffusion v1.5                    | `SPALEGuidedStableDiffusionPipeline`            |
| Stable Diffusion v2.1                    | `SPALEGuidedStableDiffusionPipeline`            |
| Stable Diffusion XL                      | `SPALEGuidedStableDiffusionXLPipeline`          |
| ControlNet (SD v1.5 + OpenPose)          | `SPALEGuidedStableDiffusionControlNetPipeline`  |
| ControlNet (SDXL + OpenPose)             | `SPALEGuidedStableDiffusionXLControlNetPipeline`|
| PixArt-Sigma (XL)                        | `SPALEGuidedPixArtSigmaPipeline`                |

Each pipeline supports both entropy-based and kernel-based guidance (e.g., Vendi, RKE, Conditional RKE) in a prompt-aware and scalable fashion.

---

## Installation

1. Clone this repository:
```bash
git clone https://github.com/mjalali/spale-guidance.git
cd spale-guidance/spale-diffusers
pip install -r requirements.txt
```

## Usage

You can directly import and use the SPALE-enabled pipelines:

```python
from spale_diffusers.diffusion import get_diffusion_pipeline

pipe = get_diffusion_pipeline(name='sdxl')

image = pipe(
    prompt="a photorealistic portrait of a man with freckles",
    guidance_scale=7.5,
    criteria='vscore_clip',
    algorithm='cond-rke',
    criteria_guidance_scale=0.4,
    num_inference_steps=50,
    kernel='gaussian',
    sigma_image=0.8,
    sigma_text=0.35,
    guidance_freq=10,
    use_latents_for_guidance=True,
    regularize=False,
    regions_list=['face'],
).images[0]

image.save("output.jpg")
```

## Bibtex Citation
If you use this work in your research, please cite it as:

```bibtex
@article{jalali2025spale,
  author    = {Mohammad Jalali and Haoyu Lei and Amin Gohari and Farzan Farnia},
  title     = {SPALE: Scalable Prompt-Aware Latent Entropy-based Diversity Guidance in Diffusion Models},
  journal   = {ArXiv},
  year      = {2025},
}
```
