import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from PIL import Image

def load_ucsf(debug=False, classification=True, basic_mask=False):
    """Loads the UCSF_PDGM dataset
    Args:
        debug (bool, optional): smaller dataset for testing. Defaults to False.
        classification (bool, optional): classification or segmentation model. Defaults to True.
        basic_mask (bool, optional): single or multiple tumour mask pixel types. Defaults to False.
    Returns:
        complete split: x_train, y_train, x_val, y_val, x_test, y_test
    """
    print("loading dataset...")
    if debug:
        ds = load_dataset("chehablab/UCSF_PDGM", split="train[:1550]")
    else:
        ds = load_dataset("chehablab/UCSF_PDGM", split="train")
    df = pd.DataFrame(ds, columns=["volume_id", "slice_id", "t1", "t1c", "t2", "tumor_mask", "is_tumorous"])
    train_df, val_df, test_df = split_dfs(df)
    if classification:
        return split_classification(train_df, val_df, test_df)
    else:
        return split_segmentation(train_df, val_df, test_df, basic_mask)

def split_ids(df):
    """Splits the dataset by volume id"""
    patient_ids = np.asarray(df["volume_id"].unique(), dtype=object)
    train_ids, temp_ids = train_test_split(patient_ids, test_size=0.5, random_state=1606009, shuffle=False)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=1606009, shuffle=False)
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
    return train_df, val_df, test_df

def split_x(df):
    """Splits the MRI images for x sets"""
    x_cols = ["t1", "t1c", "t2"]
    return df[x_cols].reset_index(drop=True)

def split_y(df, classification, basic_mask):
    """Splits the predicted column"""
    if classification:
        return np.asarray(df["is_tumorous"])
    else:
        return update_tumor_mask(df, basic_mask)
    
def update_tumor_mask(df, basic_mask):
    """Updates the tumour mask so it is visually readable"""
    if not basic_mask:
        custom_palette = np.array([
            [0, 0, 0],      # 0 - Background: Black
            [0, 255, 0],    # 1 - NCR/NET: Green
            [255, 255, 0],  # 2 - ED: Yellow
            [0, 0, 0],      # Placeholder
            [255, 0, 0]     # 4 - ET: Red
        ])
    else:
        custom_palette = np.array([
            [0, 0, 0],      # 0 - Background: Black
            [0, 255, 0],    # 1 - NCR/NET: Red
            [0, 255, 0],    # 2 - ED: Red
            [0, 0, 0],      # Placeholder
            [0, 255, 0]     # 4 - ET: Red
        ])
    
    label_indices = np.stack(df["tumor_mask"].values).astype(int)
    label_indices = np.clip(label_indices, 0, 4)
    coloured_batch = custom_palette[label_indices]
    return pd.Series([Image.fromarray(img) for img in coloured_batch])
    
def split_tumorous(df):
    """Splits the dataframe to only tumourous samples"""
    return df[df["is_tumorous"] == True].reset_index(drop=True)

def split_groups(train_df, val_df, test_df, classification, basic_mask=False):
    """Splits all the groups of data to the specifications given"""
    x_train = split_x(train_df)
    x_val = split_x(val_df)
    x_test = split_x(test_df)
    y_train = split_y(train_df, classification, basic_mask)
    y_val = split_y(val_df, classification, basic_mask)
    y_test = split_y(test_df, classification, basic_mask)
    return x_train, y_train, x_val, y_val, x_test, y_test

def split_classification(train_df, val_df, test_df):
    """Splits the data for classification"""
    return split_groups(train_df, val_df, test_df, True)
    
def split_segmentation(train_df, val_df, test_df, basic_mask):
    """Splits the data for segmentation"""
    train_df = split_tumorous(train_df)
    val_df = split_tumorous(val_df)
    test_df = split_tumorous(test_df)
    return split_groups(train_df, val_df, test_df, False, basic_mask)
    