# 🩺 Skin Lesion Classification Using Deep Learning (HAM10000)

A deep learning–based system for **automated multiclass classification of skin lesions** using dermoscopic images from the **HAM10000 (Skin Cancer MNIST)** dataset. The project evaluates **CNN and Transformer-based architectures** and focuses on building a **robust, reproducible, and research-ready** medical image classification pipeline.

---

## 👥 Team Information

### Yelchuri Siva Rama Krishna Vinay

- LinkedIn: https://www.linkedin.com/in/yelchurivinay28/
- **Role & Contribution:** Project lead; responsible for model selection, training, evaluation, and overall system integration. Implemented deep learning pipelines for multiclass skin lesion classification and handled performance analysis.

### Hari Lakshmi Sai Manikanta

- LinkedIn: https://www.linkedin.com/in/manihari7/
- **Role & Contribution:** Worked on dataset preprocessing, augmentation strategies, and handling class imbalance. Assisted in experimentation and result validation.

### Prajapathi Satyanarayana

- LinkedIn: https://www.linkedin.com/in/prajapathi-satyanarayana
- **Role & Contribution:** Contributed to exploratory data analysis, documentation, and visualization of results. Assisted in model comparison and report preparation.

---

## 📌 Abstract

Early detection of skin cancer significantly improves patient survival rates; however, manual diagnosis using dermoscopic images is time-consuming and prone to inter-observer variability. This project presents a deep learning–based system for automated multiclass skin lesion classification using the **HAM10000 (Skin Cancer MNIST)** dataset. The proposed approach leverages convolutional and transformer-based architectures to classify dermoscopic images into seven lesion categories. The system addresses key challenges such as **class imbalance**, **intra-class similarity**, and **dataset variability**, achieving reliable classification performance suitable for computer-aided diagnosis support.

---

## 🧩 About the Project

This project implements an **end-to-end deep learning pipeline** for skin cancer classification using dermoscopic images. The system takes a skin lesion image as input and predicts the corresponding lesion class. The primary goal is to build a **robust, efficient, and reproducible** classification system suitable for academic research and future clinical decision-support applications.

### Applications

- Computer-aided dermatology diagnosis
- Medical image analysis research
- Early skin cancer screening support systems

---

## 🔁 System Workflow

```text
Input Dermoscopic Image
→ Image Preprocessing
→ Data Augmentation
→ Deep Learning Model (CNN / Transformer)
→ Multiclass Classification Output
```

---

## 📊 Dataset Used

### 👉 Skin Cancer MNIST – HAM10000 Dataset

#### 🗂 Dataset Details

- **Total Images:** ~10,000 dermoscopic images
- **Number of Classes:** 7
- **Image Type:** RGB images (JPG)
- **Data Source:** Kaggle (Harvard Dataverse + ISIC Archive)

#### Class Labels

- Actinic keratoses and intraepithelial carcinoma (**akiec**)
- Basal cell carcinoma (**bcc**)
- Benign keratosis-like lesions (**bkl**)
- Dermatofibroma (**df**)
- Melanoma (**mel**)
- Melanocytic nevi (**nv**)
- Vascular lesions (**vasc**)

---

## 🧰 Tools & Technologies Used

- **Programming Language:** Python
- **Frameworks:** PyTorch, Torchvision
- **Libraries:** NumPy, OpenCV, Matplotlib, scikit-learn

### Development Environment

- Windows 11 (local system)
- Google Colab (GPU-based training)

---

## 🔍 Data Preprocessing & EDA

- All images converted to **RGB format** for consistency
- Images resized to fixed dimensions suitable for model input
- **Data augmentation** applied to reduce overfitting and mitigate class imbalance
- Corrupted and low-quality images removed
- Dataset split into **training, validation, and testing** sets

---

## 🧪 Model Training Information

- Deep learning models trained using **supervised learning**
- **Transfer learning** applied using pretrained weights
- Loss functions and **class-weighting** used to address class imbalance
- Hyperparameters tuned experimentally for optimal performance

---

## 🧾 Model Evaluation

### Metrics Used

- Accuracy
- Precision
- Recall
- F1-score
- ROC–AUC (macro-average)
- Confusion Matrix

Evaluation is performed on **unseen test data** to assess generalization capability.

---

## 🏆 Results (Summary)

- Achieved strong multiclass classification performance across all lesion categories
- Improved detection of minority classes through augmentation and balanced training
- Demonstrated robustness on real-world dermoscopic images

**Note:** Detailed numerical results are provided in the project documentation.

---

## 📄 Documentation

Detailed explanations of system design, dataset handling, model architecture, experiments, and results are available in the `Documents/` folder:

- Abstract
- Project documentation
- Review and final presentations
- Camera-ready paper

---

## ⚠️ Notes

- This project is intended for **academic and research purposes only**.
- The dataset and pretrained models are subject to their respective licenses.
- The system is **not a replacement for professional medical diagnosis**.
