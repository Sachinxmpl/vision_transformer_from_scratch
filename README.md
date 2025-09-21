# Vision Transformer (ViT) from Scratch

This repository contains a **PyTorch implementation of the Vision Transformer (ViT)**, built completely from scratch without relying on high-level Transformer libraries.  
It follows the approach introduced in the paper:

📄 [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)  
*Alexey Dosovitskiy et al., ICLR 2021*

---

## 🚀 Features
- Custom implementation of core ViT components:
  - Patch Embedding
  - Multi-Head Self Attention (MHSA)
  - Position Embeddings
  - Transformer Encoder Layers
  - Classification Head
- Training pipeline for CIFAR-10 dataset
- 
---

## 📂 Project Structure
vision_transformers_from_scratch/
│── dataset.py # CIFAR-10 dataloaders
│── train.py # Training loop 
│── models/
│ └── ViT.py # Vision Transformer implementation
│── best_vit_cifar10.pth # Saved best model after training 
