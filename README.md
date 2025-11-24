# Image Captioning for the Visually Impaired  
**Dataset:** [VizWiz Image Captioning](https://vizwiz.org/tasks-and-datasets/image-captioning/) *(~23k training images)*

---

## Overview  
This project builds an image-captioning model tailored to the VizWiz dataset — photos captured by blind and low-vision users. These images are often blurry, low-light, poorly framed, or occluded, making them far more challenging than standard captioning datasets.

The goal is to generate clear, informative captions that remain robust under real-world visual noise.

---

## Model Architecture  
- **Encoder:** InceptionV3 pretrained on ImageNet, using features from the `mixed7` layer  
- **Projection Layer:** 1×1 convolution projecting features into a 512-dimensional embedding and reshaped into a spatial sequence  
- **Decoder:** GRU-based sequence decoder (embedding size = 256, hidden size = 512) with teacher forcing  
- **Attention:** Bahdanau (additive) attention applied over encoder features  
- **Training Schedule:**  
  - Train decoder with encoder frozen  
  - After epoch 2, unfreeze upper encoder layers (BatchNorm layers remain frozen)  
  - Gradient clipping + ReduceLROnPlateau for training stability  

---

## Data Pipeline  
- Images and captions stored in **Google Cloud Storage (GCS)**  
- Synced to local Colab disk for high-throughput access  
- `tf.data` pipeline with:  
  - Parallelised preprocessing  
  - On-disk caching (`train.cache`, `val.cache`)  
  - Prefetching for maximum GPU utilisation  
- Captions cleaned, tokenised, and padded  
- Final vocabulary size ≈ **4.7k**

---

## Training  
- Environment: Google Colab (NVIDIA A100 GPU)  
- Objective: next-token prediction (negative log-likelihood)  
- Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau  
- Staged fine-tuning improves generalisation on noisy images  

---

## Validation Results  
Evaluation uses the **official VizWiz multi-reference BLEU** scoring.

| Metric | Score |
|--------|-------|
| **BLEU-1** | 0.57 |
| **BLEU-2** | 0.38 |
| **BLEU-3** | 0.25 |
| **BLEU-4** | 0.16 |

> Test-set captions are not publicly released, so validation-set metrics are reported.

---
