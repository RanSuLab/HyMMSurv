# HyM²Surv

This repository provides a complete pipeline for multimodal survival prediction by integrating Whole Slide Images (WSIs) and tumor microbiome features. 

## 1. Downloading TCGA Data

Download WSIs and clinical data from the Genomic Data Commons (GDC) portal: 
🔗 https://portal.gdc.cancer.gov

Select the following cancer types:
- STAD (Stomach Adenocarcinoma)
- COAD (Colon Adenocarcinoma)
- ESCA (Esophageal Carcinoma)
- HNSC (Head and Neck Squamous Cell Carcinoma)

## 2. Download Microbiome Abundance Data

Download tumor microbiome abundance data from: 
🔗 https://research.repository.duke.edu/record/126?v=zip
```bash
'./dataset/abundance/bacteria.unambiguous.decontam.tissue.sample.rpm.relabund.txt'
```

## 3. Processing Whole Slide Images 

We use **Trident**, a scalable toolkit for large-scale WSI processing:
🔗 https://github.com/mahmoodlab/TRIDENT

 Feature Extraction Configuration:
- Encoder: `UNI-h2`
- Patch Feature Dimension: `1536`
- Magnifications: `20x` and `10x`

```bash
## 20x
python ./TRIDENT/run_batch_of_slides.py \
  --task all \
  --wsi_dir ./dataset/TCGA-WSI/STAD \
  --job_dir ./dataset/TCGA-Features/STAD/trident_processed_20x \
  --patch_encoder uni_v2 \
  --mag 20 \
  --patch_size 256 \
  --gpu 2

# 10x
python ./TRIDENT/run_batch_of_slides.py \
  --task all \
  --wsi_dir ./dataset/TCGA-WSI/STAD \
  --job_dir ./dataset/TCGA-Features/STAD/trident_processed_10x \
  --patch_encoder uni_v2 \
  --mag 10 \
  --patch_size 256 \
  --gpu 2
```
## 4. Data Preprocessing

### 4.1 Construct Multimodal `.pkl` Dataset

Extract and align the following data sources:

- Pathology features (`.h5`)
- Survival labels
- Microbiome abundance

All patch-level features are retained at this stage (no sampling applied) and saved into a unified `.pkl` file for downstream processing.

```bash
python ./Datasets_builder/data_builder.py
# output:'./dataset/dataset_orignal/STAD_OS_WGS.pkl'
```

### 4.2 Construct Initial Hypergraph (per Magnification)

Build a KNN-based hypergraph using pathology features.

- `radius`: number of neighboring nodes used to construct WSI hyperedges

```bash
python -u ./Datasets_builder/creat_graph.py \
  --data_path ./dataset/dataset_orignal/STAD_OS_WGS.pkl \
  --radius 12
  # output:'./dataset/dataset_orignal/STAD_OS_WGS_radius_12.pkl'
  ```


  ### 4.3 Patch Sampling on Hypergraph

Perform patch sampling on the constructed hypergraph for each magnification.

- `num_patches`: number of sampled pathology patches per slide

```bash
python -u ./Datasets_builder/sampling_patch.py \
  --data_path ./dataset/dataset_orignal/STAD_OS_WGS_radius_12.pkl \
  --num_patches 2000
 # output:'./dataset/dataset_orignal/STAD_OS_WGS_radius_12_2000.pkl'
  ```

  ## 5. Running Experiments

Train the model with cross-magnification hypergraph construction.

- `k_init`: number of patches selected across magnifications to build multimodal hyperedges

```bash
python -u ./main.py \
  --device 4 \
  --batchsize 1 \
  --epochs 50 \
  --model M2Surv \
  --k_init 8 \
  --data_path ./dataset/dataset_orignal/STAD_OS_WGS_radius_12_2000.pkl \
  --results_base ./dataset/Experiments_results/kinit_cross_model
```