STABLE_DIFFUSION_MODEL = "/media/student/data/models/SD2.1/stable-diffusion-2-1/"
STABLE_DIFFUSION_XL_MODEL = "/media/student/data/models/SDXL.1/stable-diffusion-xl-base-1.0/"
CLIP_Feature_Extractor_PATH = "openai/clip-vit-base-patch32"
DATASET_DIR = '/media/student/data/datasets/geode/'
REAL_HOLD_NOT_IN_EVAL_PATH = f'{DATASET_DIR}index.csv'

proxies = {
    "http": "http://proxy.cse.cuhk.edu.hk:8000",
    "https": "http://proxy.cse.cuhk.edu.hk:8000",
}

REGIONS = {
    'geode': ['Africa', 'EastAsia', 'America', 'Europe'],
    'dollarstreet': ['Europe', 'Africa', 'the Americas', 'Asia'],
}
