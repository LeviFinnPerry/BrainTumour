# Tumour Detection and Segmentation from Multi-Sequence 2D Brain MRI Slices (UCSF-PDGM)

|                          |                          |                          |
|------------------------|------------------------|------------------------|
| ![](images/paste-2.jpeg) | ![](images/paste-3.jpeg) | ![](images/paste-4.jpeg) |

## Problem

To detect and localise tumours in preoperative MRI scans of patients with brain cancer. Split into two stages to first detect if there is a tumour present in the image slices followed by locating the tumour on the brain image slice.

1.  Classification:
    -   Detections of tumour in each slice using `is_tumorous`
2.  Segmentation:
    -   Pixel localisation using `tumor_mask`.

## Data Structure

### Dataset

| Field         | Description                                      |
|---------------|--------------------------------------------------|
| `volume_id`   | Unique patient identifier                        |
| `slice_id`    | Index of slice of MRI (155 slices)               |
| `t1`          | 2D Pre-contrast T1 weighted MRI slice            |
| `t1c`         | 2D Post-contrast T1 weighted MRI slice           |
| `t2`          | 2D T2 weighted MRI slice                         |
| `tumor_mask`  | 2D tumor segmentation mask                       |
| `is_tumorous` | Whether any tumour label is present on the slice |
| `tumor_type`  | Final diagnosis                                  |
| `who_grade`   | WHO tumour grade                                 |
| `sex`         | Patient sex M or F                               |
| `age`         | Patient age at MRI                               |

### Tumour mask

| Value | Label      | Description                                       |
|-------|------------|---------------------------------------------------|
| `0`   | Background | No tumour                                         |
| `1`   | NCR/NET    | Necrotic and non-enhancing tumour core            |
| `2`   | ED         | Peritumoral edema / surrounding FLAIR abnormality |
| `4`   | ET         | GD-enhancing tumour                               |

### Tumour Grades

| Tumour Region of Interest | Values        |
|---------------------------|---------------|
| No Tumour                 | `0`           |
| Whole Tumour              | `1`, `2`, `4` |
| Tumour Core               | `1`, `4`      |
| Enhancing Tumour          | `4`           |

All patients have a confirmed diagnosis, after removing missing values there is 414 patients x 155 slices = 64170 slices. `is_tumorous` varies per slice with 38,715 no tumour slices and 25,455 tumour slices. Each slice comes with three MRI sequences (`t1`, `t1c`, `t2`) as well as a `tumor_mask` for segmentation.

[UCSF_PDGM Dataset](https://huggingface.co/datasets/chehablab/UCSF_PDGM) - Hugging Face

