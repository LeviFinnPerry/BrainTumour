"""Flattens MRI image pixels with multiple threads due to dataset size"""
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def resize_images(t1_img, t1c_img, t2_img, target_size):
    r1 = t1_img.resize(target_size, Image.Resampling.BILINEAR)
    r2 = t1c_img.resize(target_size, Image.Resampling.BILINEAR)
    r3 = t2_img.resize(target_size, Image.Resampling.BILINEAR)
    return r1, r2, r3

def extract_bytes(r1, r2, r3):
    b1 = np.frombuffer(r1.tobytes(), dtype=np.uint8).astype(np.float32)
    b2 = np.frombuffer(r2.tobytes(), dtype=np.uint8).astype(np.float32)
    b3 = np.frombuffer(r3.tobytes(), dtype=np.uint8).astype(np.float32)
    return b1, b2, b3

def stack_pixels(b1, b2, b3):
    return np.stack([b1, b2, b3], axis=-1)
    
def flatten_pixels(pixels):
    return pixels.ravel()

def create_empty_stack(num_rows):
    return np.empty((num_rows, 3072), dtype=np.float32)

def zip_arrays(df):
    return list(zip(df["t1"].values, df["t1c"].values, df["t2"].values))

def compile_rows(results, full_stack, num_rows):
    for idx, flattened_vector in enumerate(results):
        full_stack[idx] = flattened_vector
        if (idx + 1) % 10000 == 0 or (idx + 1) == num_rows:
            print(f" -> Progress: {idx + 1}/{num_rows} rows compiled successfully.")
    return full_stack    

def normalise_batch(batch, num_rows):
    lo = batch.min(axis=(1, 2), keepdims=True)
    hi = batch.max(axis=(1, 2), keepdims=True)
    normalised_batch = (batch - lo) / (hi - lo + 1e-8)
    return normalised_batch.reshape(num_rows, -1)         

def process_single_row(row_tuple):
    """
    Worker function executed on separate CPU cores.
    Processes a single row's images.
    """
    target_size = (32, 32)
    t1_img, t1c_img, t2_img = row_tuple
    
    r1, r2, r3 = resize_images(t1_img, t1c_img, t2_img, target_size)
    b1, b2, b3 = extract_bytes(r1, r2, r3)
    img_stack = stack_pixels(b1, b2, b3)
    return flatten_pixels(img_stack)


def batch_flatten_pixels_parallel(df, max_workers=None):
    """
    Parallelised image pipeline that splits 77,655 rows across all
    available CPU cores
    """
    num_rows = len(df)
    print(f"Pre-allocating matrix for {num_rows} records...")
    full_stack = create_empty_stack(num_rows)
    print("Preparing image streams for multiprocessing...")
    jobs = zip_arrays(df)
    print(f"Launching parallel processing...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_single_row, jobs, chunksize=500)
        full_stack = compile_rows(results, full_stack, num_rows)
    print("Normalization across the whole dataset...")
    batch = full_stack.reshape(num_rows, 32, 32, 3)
    full_stack = normalise_batch(batch, num_rows)
    print(f"Final matrix shape: {full_stack.shape}")
    return full_stack
