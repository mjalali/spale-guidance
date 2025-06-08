import torch
import os
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_cads import StableDiffusionPipelineCADS
from diffusers import DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"
custom_cache_dir = "cache" 
output_base_dir = "cads" 
num_images_to_generate = 20 


prompt_for_filename = "woman" 
output_dir = os.path.join(output_base_dir, prompt_for_filename.replace(" ", "_").lower())
os.makedirs(output_dir, exist_ok=True)


print(f"Loading model: {model_id} from cache: {custom_cache_dir}")

pipe = StableDiffusionPipelineCADS.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    cache_dir=custom_cache_dir,
    use_safetensors=True # 
)

scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, cache_dir=custom_cache_dir)
pipe.scheduler = scheduler

pipe = pipe.to("cuda")
print("Pipeline loaded and moved to GPU.")

prompt = "A white woman walking on the street."
negative_prompt = "cartoon, blurry, low quality, ugly, text, watermark, signature, deformed"

image_height = 768
image_width = 768
num_inference_steps = 50
guidance_scale = 7.5

cads_active = True
cads_tau1 = 0.6
cads_tau2 = 0.9
cads_noise_scale = 0.25
cads_mixing_factor = 1.0
cads_rescale = True

print(f"Generating {num_images_to_generate} images for the prompt: '{prompt}'")

generator = torch.Generator(device="cuda").manual_seed(42)

results = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=image_height,
    width=image_width,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    num_images_per_prompt=num_images_to_generate,
    generator=generator,
    # --- CADS Parameters ---
    cads_active=cads_active,
    cads_tau1=cads_tau1,
    cads_tau2=cads_tau2,
    cads_noise_scale=cads_noise_scale,
    cads_mixing_factor=cads_mixing_factor,
    cads_rescale=cads_rescale,
    # --- End CADS Parameters ---
)

generated_images = results.images

for i, image in enumerate(generated_images):
    output_filename = os.path.join(output_dir, f"image_{i:02d}.png")
    image.save(output_filename)
    print(f"Image saved as {output_filename}")

print(f"\nSuccessfully generated and saved {len(generated_images)} images to '{output_dir}'.")
