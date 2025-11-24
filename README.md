# Image Captioning for the Visually Impaired  
**Dataset:** [VizWiz Image Captioning](https://vizwiz.org/tasks-and-datasets/image-captioning/) 

---

## Overview  
This project builds an image–captioning model designed specifically for the VizWiz dataset — photos taken by people who are blind or have low vision. These images are often blurry, poorly framed, low-light, or partially occluded, making them significantly more challenging than typical captioning datasets.

The goal is to produce informative, reliable captions under real-world conditions that mainstream captioning models often struggle with.

---

## Model Architecture  
- **Encoder:** InceptionV3 pretrained on ImageNet, using features from the `mixed7` layer  
- **Projector:** 1×1 convolution reducing features to a 512-dim embedding, reshaped into a sequence  
- **Decoder:** GRU-based decoder (embedding = 256, hidden size = 512) with teacher forcing  
- **Attention:** Bahdanau (additive) attention applied over spatial encoder features  
- **Training schedule:**  
  - Train decoder while encoder is frozen  
  - After epoch 2, unfreeze only the top encoder layers (BatchNorm layers remain frozen)  
  - Mixed precision and gradient clipping used for training stability  

---

## Data Pipeline  
- Images and captions stored in Google Cloud Storage (GCS)  
- Pipeline built with `tf.data.Dataset` using parallel mapping, caching, and prefetching  
- **Caching behaviour:**  
  - During the first epoch, the pipeline writes a local cache file (`train.cache`)  
  - All subsequent epochs load **from the cache**, avoiding repeated GCS reads  
  - This provides a ~20× speedup on Colab  
- Captions cleaned, tokenised, padded; final vocabulary size ≈ 4.7k  
- Training performed on Google Colab with an NVIDIA A100 GPU  

---

## Validation Results  
| Metric | Score |
|--------|-------|
| BLEU-1 | 0.57  |
| BLEU-2 | 0.38  |
| BLEU-3 | 0.25  |
| BLEU-4 | 0.16  |

> These metrics reflect performance on the validation split. The VizWiz test-set annotations are not publicly released.
