# 🫁 Generating Radiologically Realistic Lung Images with GANs

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-GAN-purple?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

> A deep learning project that leverages **Generative Adversarial Networks (GANs)** to synthesize radiologically realistic lung X-ray images — addressing the critical challenge of limited medical imaging data in AI-driven diagnostics.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Motivation](#-motivation)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Technologies](#-technologies)
- [Future Work](#-future-work)
- [Author](#-author)
- [License](#-license)

---

## 🔍 Overview

Medical AI models require vast amounts of labeled data to perform reliably — but real patient data is scarce, expensive to annotate, and privacy-restricted. This project tackles that problem by training a **GAN** to generate **synthetic lung X-ray images** that preserve radiological characteristics, enabling:

- ✅ Data augmentation for diagnostic AI models
- ✅ Privacy-safe medical research datasets
- ✅ Realistic simulation of lung conditions

---

## 💡 Motivation

- Real patient imaging data is limited by **privacy regulations (HIPAA, GDPR)**
- Annotated medical datasets are **expensive and time-consuming** to create
- Synthetic data can **supplement training datasets** without compromising patient privacy
- GANs have shown strong results in medical imaging literature

---

## 🧠 Architecture

The system consists of two competing neural networks trained adversarially:

### Generator

```
Random Noise (z)
    → Dense Layer
    → Reshape
    → Conv2DTranspose + BatchNorm + ReLU  (×N)
    → Conv2DTranspose + Tanh
    → Synthetic Lung Image
```

### Discriminator

```
Input Image (real or fake)
    → Conv2D + LeakyReLU + Dropout  (×N)
    → Flatten
    → Dense
    → Sigmoid
    → Real / Fake
```

### Training Objective

```
min_G max_D  E[log D(x)]  +  E[log(1 - D(G(z)))]
```

Both networks are trained in alternating steps:

1. Train **Discriminator** on real + generated images
2. Train **Generator** to fool the Discriminator
3. Repeat until convergence

---

## 📁 Dataset

| Property | Details |
|----------|---------|
| Type | Lung X-ray images (grayscale / RGB) |
| Source | NIH Chest X-ray, Kaggle lung datasets |
| Image Size | 64×64 / 128×128 |
| Normalization | Pixel values scaled to `[-1, 1]` |

**Preprocessing steps:**

- Resized to uniform dimensions
- Normalized pixel values to `[-1, 1]`
- Augmented with random flipping and rotation to improve generalization

---

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/Pratik1603/Generating-Radiologically-Realistic-Lung-Images-with-Generative-Adversarial-Networks-Major-Project.git
cd Generating-Radiologically-Realistic-Lung-Images-with-Generative-Adversarial-Networks-Major-Project
```

**2. Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
tensorflow>=2.10
numpy
matplotlib
opencv-python
scikit-learn
Pillow
jupyter
tqdm
```

---

## 🚀 Usage

### Run the Notebook

```bash
jupyter notebook
```

Open `GAN_Lung_Image_Generation.ipynb` and run all cells sequentially.

### Configure Hyperparameters

```python
EPOCHS        = 100
BATCH_SIZE    = 32
LATENT_DIM    = 128
IMAGE_SIZE    = (64, 64)
LEARNING_RATE = 0.0002
BETA_1        = 0.5
```

### Generate Synthetic Images After Training

```python
import tensorflow as tf
import matplotlib.pyplot as plt

noise = tf.random.normal([num_samples, LATENT_DIM])
generated_images = generator(noise, training=False)

fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
for i, ax in enumerate(axes):
    ax.imshow(generated_images[i] * 0.5 + 0.5, cmap='gray')
    ax.axis('off')
plt.suptitle("Generated Lung X-Ray Samples")
plt.show()
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Generator Loss | ~0.6 – 0.9 (converged) |
| Discriminator Loss | ~0.4 – 0.6 (converged) |
| Training Epochs | 50 – 100 |
| Image Resolution | 64×64 / 128×128 |
| Batch Size | 32 |
| Latent Dimension | 128 |

> 📷 Sample outputs are visualized within the notebook at regular epoch intervals to monitor GAN convergence and image quality.

---

## 📂 Project Structure

```
├── GAN_Lung_Image_Generation.ipynb     # Main project notebook
├── data/
│   ├── raw/                            # Original X-ray images
│   └── processed/                      # Preprocessed & normalized images
├── models/
│   ├── generator.h5                    # Saved generator weights
│   └── discriminator.h5                # Saved discriminator weights
├── outputs/
│   └── generated_samples/              # Synthetic images saved per epoch
├── utils/
│   ├── data_loader.py                  # Dataset loading & preprocessing
│   ├── model.py                        # Generator & Discriminator definitions
│   └── train.py                        # Training loop logic
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| Deep Learning | TensorFlow / Keras |
| Image Processing | OpenCV, Pillow |
| Visualization | Matplotlib |
| Environment | Jupyter Notebook |
| Architecture | DCGAN (Deep Convolutional GAN) |
| Data Handling | NumPy, scikit-learn |

---

## 🔭 Future Work

- [ ] Implement **Conditional GAN (cGAN)** to generate class-specific lung conditions (e.g., pneumonia, COVID-19)
- [ ] Integrate **FID Score** (Fréchet Inception Distance) for quantitative image quality evaluation
- [ ] Scale resolution to **256×256** using Progressive Growing GAN
- [ ] Explore **CycleGAN** for unpaired image-to-image translation between healthy and diseased lungs
- [ ] Deploy a **Streamlit / Flask web app** for on-demand synthetic image generation
- [ ] Extend to **CT scan** volumetric data generation

---

## 👨‍💻 Author

**Pratik** — [GitHub Profile](https://github.com/Pratik1603)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

> ⚠️ **Disclaimer:** Synthetic images generated by this model are intended strictly for **research and educational purposes only** and must not be used as a substitute for professional medical diagnosis or clinical decision-making.
