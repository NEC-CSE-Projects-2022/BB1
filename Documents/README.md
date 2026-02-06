# Project Documents

This folder contains all official documents related to the **Skin Cancer Classification using Deep Learning** project, including the abstract, paper, presentations, and detailed project documentation submitted for academic evaluation.

## Summary

This project focuses on automated skin lesion classification using deep learning models on the HAM10000 (Skin Cancer MNIST) dataset. The system aims to assist early detection of skin cancer by accurately classifying dermoscopic images into multiple lesion categories. The work addresses real-world challenges such as class imbalance, visual similarity between lesions, and dataset variability.

**Key objectives**

- Multiclass skin lesion classification using CNN and Transformer-based models
- Performance evaluation using standard medical imaging metrics
- Analysis of class imbalance and generalization challenges

## Repository Contents

- `CAMERA_READY_PAPER.pdf`
  Final camera-ready research paper describing the methodology, models, experiments, and results.

- `PROJECT_ABSTRACT.pdf`
  Concise abstract outlining the problem statement, approach, and key findings.

- `COLLEGE_REVIEW_PRESENTATION.pptx`
  Editable presentation used for internal college/project reviews.

- `CONFERENCE_PRESENTATION.ppt` / `CONFERENCE_PRESENTATION.pptx`
  Presentation prepared for conference or external evaluation.

- `PROJECT_DOCUMENTATION.pdf`
  Complete project documentation including system design, dataset description, preprocessing, model architecture, experimental setup, and results.

## Dataset Reference

- **Dataset:** Skin Cancer MNIST – HAM10000
- **Classes:** 7 skin lesion categories
- **Images:** ~10,000 dermoscopic images
- **Source:** Kaggle (`kmader/skin-cancer-mnist-ham10000`)
  https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

The dataset is discussed in detail in the project documentation, including preprocessing steps and train/validation/test splits.

## System Description

**Input:** Dermoscopic skin lesion images

**Processing**

- Image resizing and normalization
- Data augmentation for class imbalance handling
- Feature extraction using deep learning models

**Models Used**

- Convolutional Neural Networks (CNN-based architectures)
- Transformer-based lightweight and hybrid models

**Output**

- Predicted skin lesion class
- Performance metrics for evaluation

## Tools & Technologies

- **Programming Language:** Python 3.x
- **Frameworks:** PyTorch, Torchvision
- **Libraries:** NumPy, OpenCV, Matplotlib, scikit-learn
- **Development Environment:**
  - Local system (Windows 11)
  - Google Colab (GPU-enabled training)

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC–AUC (macro-average)

Detailed quantitative results are provided in the camera-ready paper and project documentation.

## Reproducibility

For full reproducibility, refer to `PROJECT_DOCUMENTATION.pdf`, which includes:

- Dataset preprocessing steps
- Data split strategy
- Model architectures and hyperparameters
- Training configuration and evaluation protocol

## Notes

- All documents are intended for academic and research purposes only.
- The dataset and pretrained models used in this project are subject to their respective licenses.
- Results may vary depending on hardware and training configuration.

