# SPARKE Diffusers: Improving the Diversity of Diffusion Models in Diffusers

**SPARKE: Scalable Prompt-Aware Diversity Guidance in Diffusion Models via RKE Score**

---

## Overview

This repository contains the official implementation of **SPARKE**, a method for improving diversity in prompt-guided diffusion models using **Scalable Prompt-Aware Diversity Guidance in Diffusion Models via RKE Score**. SPARKE introduces conditional entropy-guided sampling that dynamically adapts to semantically similar prompts and supports scalable generation across modern text-to-image architectures.

> Project Webpage: [https://mjalali.github.io/SPARKE](https://mjalali.github.io/SPARKE)

---

## Abstract

Diffusion models have demonstrated exceptional performance in high-fidelity image synthesis and prompt-based generation. However, achieving sufficient diversity—particularly within semantically similar prompts—remains a critical challenge. Prior methods use diversity metrics as guidance signals, but often neglect prompt awareness or computational scalability.

In this work, we propose **SPARKE**: _Scalable Prompt-Aware Diversity Guidance in Diffusion Models via RKE Score_. SPARKE leverages **conditional entropy** to guide the sampling process with respect to prompt-localized diversity. By employing **Conditional Latent RKE Score Guidance**, we reduce the computational complexity from $\mathcal{O}(n^3)$ to $\mathcal{O}(n)$, enabling efficient large-scale generation. We integrate SPARKE into several popular diffusion pipelines and demonstrate improved diversity without additional inference overhead.

---

## Supported Pipelines

The following `diffusers` pipelines have been extended with SPARKE guidance:

| Pipeline Type                             | Implementation                                    |
|------------------------------------------|---------------------------------------------------|
| Stable Diffusion v1.5                    | `SPARKEGuidedStableDiffusionPipeline`             |
| Stable Diffusion v2.1                    | `SPARKEGuidedStableDiffusionPipeline`             |
| Stable Diffusion XL                      | `SPARKEGuidedStableDiffusionXLPipeline`           |
| ControlNet (SD v1.5 + OpenPose)          | `SPARKEGuidedStableDiffusionControlNetPipeline`   |
| ControlNet (SDXL + OpenPose)             | `SPARKEGuidedStableDiffusionXLControlNetPipeline` |
| PixArt-Sigma (XL)                        | `SPARKEGuidedPixArtSigmaPipeline`                 |

Each pipeline supports both entropy-based and kernel-based guidance (e.g., Vendi, RKE, Conditional RKE) in a prompt-aware and scalable fashion.

---

## Installation

1. Clone this repository:
```bash
git clone https://github.com/mjalali/sparke-diffusers.git
cd sparke-diffusers/sparke_diffusers
pip install -r requirements.txt
```

## Usage

You can directly import and use the SPARKE-enabled pipelines:

```python

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
To cite this work, please use the following BibTeX entries:

SPARKE Diversity Guidance:
```bibtex
@article{jalali2025sparke,
    author = {Mohammad Jalali and Haoyu Lei and Amin Gohari and Farzan Farnia},
    title = {SPARKE: Scalable Prompt-Aware Diversity Guidance in Diffusion Models via RKE Score},
    journal = {arXiv preprint arXiv:2506.10173},
    year = {2025},
    url = {https://arxiv.org/abs/2506.10173},
}
```

RKE Score:
```bibtex
@inproceedings{jalali2023rke,
    author = {Jalali, Mohammad and Li, Cheuk Ting and Farnia, Farzan},
    booktitle = {Advances in Neural Information Processing Systems},
    pages = {9931--9943},
    title = {An Information-Theoretic Evaluation of Generative Models in Learning Multi-modal Distributions},
    url = {https://openreview.net/forum?id=PdZhf6PiAb},
    volume = {36},
    year = {2023}
}
```
