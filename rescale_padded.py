import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps
from tqdm import tqdm

SRC_DIR = Path('target_images')
DST_DIR = Path('target_images_padded')
TARGET  = (512, 512)
N_WORKERS = 8

DST_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXT = {'.jpg', '.jpeg'}
files = [p for p in SRC_DIR.iterdir() if p.suffix.lower() in IMG_EXT]

def process(path: Path):
    try:
        with Image.open(path) as im:
            im = im.convert('RGB')

            im.thumbnail(TARGET, Image.LANCZOS)

            delta_w = TARGET[0] - im.width
            delta_h = TARGET[1] - im.height
            padding = (
                delta_w // 2, delta_h // 2,
                delta_w - delta_w // 2, delta_h - delta_h // 2
            )
            im = ImageOps.expand(im, padding, fill=(0, 0, 0))

            out_path = DST_DIR / path.name
            im.save(out_path, format='JPEG', quality=92, subsampling=0)

    except Exception as e:
        print(f'Error {path.name}: {e}')

with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
    list(tqdm(pool.map(process, files), total=len(files), desc='Resizing with padding'))

print(f'Done! {len(files)} images saved to {DST_DIR}')
