# Image-Super-Resolution-Enhancement-Using-a-Hybrid-Model-Framework

**Overview-**
This project focuses on image super-resolution using a hybrid deep learning framework that combines **Convolutional Neural Networks (CNNs)**, **Transformers**, and **Generative Adversarial Networks (GANs)**. The objective is to enhance low-resolution images by improving spatial detail, structural consistency, and perceptual quality.

**Problem Statement-**
Existing super-resolution approaches struggle to restore all aspects of image quality simultaneously. CNNs often produce overly smooth outputs, Transformers may miss fine textures, and GANs can introduce visual artifacts. No single architecture consistently delivers high-quality results across diverse image datasets.

**Implementation Overview-**
The implementation follows a modular pipeline:

1. **CNN** module extracts local spatial features such as edges and textures

2. **Transformer** module captures long-range dependencies and global contextual information

3. **GAN-based generator** enhances perceptual sharpness and realistic texture details

4. A **fusion mechanism** integrates outputs from all components to generate the final high-resolution image

Each component is trained sequentially, and the fused output aims to balance structural accuracy and perceptual realism.

**Dataset-**
The model is trained and evaluated using the **DIV2K** dataset, which contains paired low-resolution and high-resolution images commonly used for image super-resolution research.
The dataset is not included in this repository due to size constraints and can be downloaded directly from Kaggle.

**Project Structure & Code Organization-**

**1. Data Preprocessing**

**prepare_data.py-**
This script prepares the super-resolution dataset by generating low-resolution images from high-resolution inputs. It reads HR images, applies bicubic downsampling at multiple scale factors (×2, ×3, ×4), and saves the corresponding LR images with matching filenames, enabling consistent LR–HR pairing for model training and evaluation.

**make_splits.py-**
This script creates reproducible train, validation, and test splits for the super-resolution dataset. It shuffles the list of high-resolution images using a fixed random seed and saves the filenames into separate text files, enabling consistent dataset splits across experiments.

**01_preview.py-** 
This script is used to preview and visually verify the dataset by loading one low-resolution (LR) and high-resolution (HR) image pair and displaying them side by side, ensuring that data loading, scaling, and preprocessing are working correctly before training the model.

**2. Dataset Pipeline**

**sr_dataset.py-**
This module implements a custom Py Torch dataset for image super-resolution. It dynamically generates aligned low-resolution and high-resolution image pairs by extracting random patches from high-resolution images, applying data augmentation, and optionally using pre-computed low-resolution images. The output tensors are normalized and prepared for efficient model training.

**3. Model Building**

**Model.py-**
This module implements the core neural network architectures used in the hybrid super-resolution system, including CNN, Transformer, GAN Generator, and FusionNet models, each contributing complementary spatial, contextual, and perceptual features for improved image reconstruction.

**4. Training & Evaluation**

**Metrics.py-**
This file provides evaluation utilities for the super-resolution models, including PSNR and SSIM computation, batch-wise evaluation across multiple models, and saving visual output comparisons (LR, HR, CNN, Transformer, GAN, and Fusion results).

**Training_utils.py-**
This file contains helper functions for training super-resolution models, including mixed-precision (AMP) training, checkpoint saving, and automatic backup of model weights during training.

**train.py-**
This script implements the complete training pipeline for the hybrid image super-resolution framework. It trains individual CNN, Transformer, and GAN models using mixed-precision training, followed by training a Fusion Network that combines their outputs. The script performs validation using PSNR and SSIM metrics and saves model checkpoints during training.

**Results & Evaluation-**
The proposed hybrid image super-resolution framework was evaluated using both quantitative metrics and qualitative visual analysis. The evaluation focuses on assessing reconstruction quality and perceptual consistency rather than claiming state-of-the-art performance.


**Quantitative Evaluation-**
Model performance was measured using standard image quality metrics:

1. PSNR (Peak Signal-to-Noise Ratio)

2. SSIM (Structural Similarity Index)

Average validation metrics are reported in the results/metrics_summary.txt file for reference. These values indicate consistent reconstruction behavior across the evaluated samples.

Note: The reported metrics are provided for reference purposes. Further tuning and extended training are expected to improve quantitative performance.

**Qualitative Evaluation-**
Visual comparisons demonstrate that:

1. The CNN model produces stable reconstructions with good structural consistency.

2. The Transformer model captures global contextual information more effectively.

3. The GAN-based generator enhances perceptual sharpness and texture realism.

4. The Fusion model combines complementary features from all models, resulting in more balanced and visually consistent super-resolved images.

Due to repository size constraints, qualitative results are shown for one representative sample. The complete visual comparison (LR, CNN, Transformer, GAN, and Fusion outputs) is available in the results/ directory.

**🚧 Project Status-**
This project is currently a **work in progress**. Future improvements include extended evaluation, hyperparameter optimization, and experimentation with advanced fusion strategies.
