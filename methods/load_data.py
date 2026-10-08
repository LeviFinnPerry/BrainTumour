import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from PIL import Image

def load_ucsf(debug=False, classification=True, basic_mask=True, image_size=32):
    """Loads the UCSF_PDGM dataset
    Args:
        debug (bool, optional): smaller dataset for testing. Defaults to False.
        classification (bool, optional): classification or segmentation model. Defaults to True.
        basic_mask (bool, optional): single or multiple tumour mask pixel types. Defaults to True.
        image_size (int, optional): size of the square image. Defaults to 32.
    Returns:
        complete split: x_train, y_train, x_val, y_val, x_test, y_test
    """
    print("loading dataset...")
    if debug:
        ds = load_dataset("chehablab/UCSF_PDGM", split="train[:15500]")
    else:
        ds = load_dataset("chehablab/UCSF_PDGM", split="train")
    df = pd.DataFrame(ds, columns=["volume_id", "slice_id", "t1", "t1c", "t2", "tumor_mask", "is_tumorous"])
    print("dataset loaded.")
    train_df, val_df, test_df = split_dfs(df)
    if classification:
        return split_classification(train_df, val_df, test_df, image_size)
    else:
        return split_segmentation(train_df, val_df, test_df, basic_mask, image_size)

def split_ids(df):
    """Splits the dataset by volume id"""
    patient_ids = np.asarray(df["volume_id"].unique(), dtype=object)
    train_ids, temp_ids = train_test_split(patient_ids, test_size=0.4, random_state=1606009, shuffle=False)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.4, random_state=1606009, shuffle=False)
    return train_ids, val_ids, test_ids

def split_df(df, ids):
    """Splits the dataframe for given ids"""
    t_df = df[df["volume_id"].isin(ids)]
    return t_df.reset_index(drop=True)
    
def split_dfs(df):
    """Splits all dataframes for the given ids"""
    train_ids, val_ids, test_ids = split_ids(df)
    train_df = split_df(df, train_ids)
    val_df = split_df(df, val_ids)
    test_df = split_df(df, test_ids)
    validate_dfs(train_df, val_df, test_df)
    return train_df, val_df, test_df

def get_sizes(df):
    """Gets the number of patients and size of dataframe"""
    patients = df["volume_id"].unique()
    length = df.shape[0]
    return patients, length

def validate_dfs(train_df, val_df, test_df):
    """Validates all dataframes are expected lengths for amount of patients"""
    train_patients, train_length = get_sizes(train_df)
    val_patients, val_length = get_sizes(val_df)
    test_patients, test_length = get_sizes(test_df)
    
    if len(train_patients) * 155 != train_length:
        print("Training size does not match patients")
    elif len(val_patients) * 155 != val_length:
        print("Validation size does not match patients")
    elif len(test_patients) * 155 != test_length:
        print("Test size does not match patients")
    else:
        print("No data leakage of patient ids in train, validation and test splits")
    

def split_x(df, image_size):
    """Splits the MRI images for x sets while preserving image shape."""
    x_cols = ["t1", "t1c", "t2"]
    target_size = (image_size, image_size)
    stacked = []

    for _, row in df[x_cols].iterrows():
        channels = []
        for col in x_cols:
            img = row[col]
            if img is None:
                raise ValueError(f"Missing MRI image in column '{col}'")
            arr = np.asarray(img.resize(target_size, Image.Resampling.BILINEAR), dtype=np.float32)
            channels.append(arr)
        stacked.append(np.stack(channels, axis=-1))

    return np.stack(stacked, axis=0).astype(np.float32)

def split_y(df, classification, basic_mask, image_size):
    """Splits the predicted column"""
    if classification:
        return np.asarray(df["is_tumorous"])
    else:
        return update_tumor_mask(df, basic_mask, image_size)
    
def get_binary_tumor_mask(df, image_size):
    """
    Binary tumour vs. background mask, single channel.
    Any nonzero label becomes tumour = 1.
    """
    label_indices = np.stack(df["tumor_mask"].values).astype(np.uint8)

    binary_masks = []
    for mask in label_indices:
        img = Image.fromarray(mask).resize((image_size, image_size), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        arr = (arr > 0).astype(np.float32)
        binary_masks.append(arr[..., np.newaxis])  # (H, W) -> (H, W, 1)

    return np.stack(binary_masks)  # (N, H, W, 1)


def get_multiclass_tumor_mask(df, image_size):
    """
    Multi-compartment RGB mask, distinguishing NCR/NET, ED, and ET.
    """
    custom_palette = np.array([
        [0, 0, 0],      # 0 - Background: Black
        [0, 255, 0],    # 1 - NCR/NET: Green
        [255, 255, 0],  # 2 - ED: Yellow
        [0, 0, 0],      # Placeholder
        [255, 0, 0]     # 4 - ET: Red
    ])

    label_indices = np.clip(np.stack(df["tumor_mask"].values).astype(int), 0, 4).astype(np.uint8)
    coloured_batch = custom_palette[label_indices].astype(np.uint8)

    resized_batch = []
    for img in coloured_batch:
        resized = Image.fromarray(img, mode="RGB").resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
        resized_batch.append(np.array(resized, dtype=np.uint8))

    return np.stack(resized_batch)  # (N, H, W, 3)


def update_tumor_mask(df, basic_mask, image_size):
    """Returns the tumour mask, binary if basic_mask, else 4 coloured."""
    if basic_mask:
        return get_binary_tumor_mask(df, image_size)
    else:
        return get_multiclass_tumor_mask(df, image_size)
    
def split_tumorous(df):
    """Splits the dataframe to only tumourous samples"""
    return df[df["is_tumorous"] == True].reset_index(drop=True)

def split_groups(train_df, val_df, test_df, classification, basic_mask=False, image_size=32):
    """Splits all the groups of data to the specifications given"""
    x_train = split_x(train_df, image_size)
    x_val = split_x(val_df, image_size)
    x_test = split_x(test_df, image_size)
    y_train = split_y(train_df, classification, basic_mask, image_size)
    y_val = split_y(val_df, classification, basic_mask, image_size)
    y_test = split_y(test_df, classification, basic_mask, image_size)
    return x_train, y_train, x_val, y_val, x_test, y_test

def split_classification(train_df, val_df, test_df, image_size=32):
    """Splits the data for classification"""
    return split_groups(train_df, val_df, test_df, True, image_size=image_size)
    
def split_segmentation(train_df, val_df, test_df, basic_mask=False, image_size=32):
    """Splits the data for segmentation"""
    train_df = split_tumorous(train_df)
    val_df = split_tumorous(val_df)
    test_df = split_tumorous(test_df)
    return split_groups(train_df, val_df, test_df, False, basic_mask=basic_mask, image_size=image_size)
