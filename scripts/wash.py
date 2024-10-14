import json
from tqdm import tqdm
from PIL import Image
import hashlib
import requests
import concurrent.futures
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

input_path = "/data/changye/dataset/obelics_10k/obelics_10k.json"
output_path = "/data/changye/dataset/obelics_10k/obelics_10k_washed.json"

# Ensure the images directory exists
images_dir = "images"
os.makedirs(images_dir, exist_ok=True)

# Create a session with retry strategy
session = requests.Session()
retry_strategy = Retry(
    total=10,  # Total number of retries
    backoff_factor=1,  # A delay between retries (exponential backoff)
    status_forcelist=[403, 429, 500, 502, 503, 504],  # Retry on these HTTP statuses
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

with open(input_path, "r") as f:
    data = json.load(f)

print(f"Total items: {len(data)}")

def load_image(url, path):
    try:
        image_name_hash = hashlib.sha256(url.encode()).hexdigest()
        image_path = f"{path}/{image_name_hash}.png"
        if os.path.exists(image_path):
            return image_path
        response = session.get(url, stream=True, timeout=10)  # Use session with retry
        response.raise_for_status()
        image = Image.open(response.raw)
        image_path = f"{path}/{image_name_hash}.png"
        image.save(image_path)
        return image_path
    except Exception as e:
        print(f"Error loading image {url}: {e}")
        return None

def process_piece(piece):
    new_item = piece.copy()
    flag = True
    for i in range(len(piece["images"])):
        if piece['images'][i] is None and piece['texts'][i] is not None:
            continue
        elif piece['images'][i] is not None and piece['texts'][i] is None:
            image_path = load_image(piece['images'][i], images_dir)
            if image_path:
                new_item["images"][i] = image_path
            else:
                flag = False
                break
    return new_item if flag else None

# Use ThreadPoolExecutor for I/O-bound tasks
with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    results = list(tqdm(executor.map(process_piece, data), total=len(data)))

# Filter out None results
outputs = [result for result in results if result is not None]

print(f"Processed items: {len(outputs)}")

with open(output_path, "w") as f:
    json.dump(outputs, f)