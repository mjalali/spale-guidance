from pathlib import Path
import sys


from diffusers.pipelines.dit.pipeline_dit import DiTPipeline

DiT_XL_2 = '/media/student/data/models/DiT/DiT-XL-2-256/'
def main():
    pipe = DiTPipeline.from_pretrained(DiT_XL_2)

    output = pipe(class_labels=pipe.get_label_ids(["white shark", "umbrella"])).images[0]
    image = output
    image.save("/media/student/data/models/DiT/dit_rke.png")

if __name__ == "__main__":
    main()
