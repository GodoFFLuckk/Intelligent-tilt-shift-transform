import os
from PIL import Image
import imagehash
import itertools

def find_similar_images(folder_path, hash_size=8, threshold=10):
    hashes = {}

    valid_exts = ('.jpg', '.jpeg')

    for root, _, files in os.walk(folder_path):
        for fname in files:
            if not fname.lower().endswith(valid_exts):
                continue
            path = os.path.join(root, fname)
            try:
                with Image.open(path) as img:
                    img_hash = imagehash.phash(img, hash_size=hash_size)
                hashes[path] = img_hash
            except Exception as e:
                print(f"Can't process {path}: {e}")

    similar_pairs = []
    for (path1, hash1), (path2, hash2) in itertools.combinations(hashes.items(), 2):
        distance = hash1 - hash2
        if distance <= threshold:
            similar_pairs.append((path1, path2, distance))

    if similar_pairs:
        print("Found simular examples:")
        for p1, p2, dist in sorted(similar_pairs, key=lambda x: x[2]):
            print(f"{p1}  <->  {p2}  (distance: {dist})")
    else:
        print("Didn't found simular examples.")


if __name__ == '__main__':
    find_similar_images('input_images', 8, 15)
