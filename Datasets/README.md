
## Dataset (HAM10000)

<img width="1584" height="896" alt="image" src="https://github.com/user-attachments/assets/5baf84be-0252-48e7-bb34-ab84a4c1af40" />


- **Total images:** 10,015
- **Classes:** 7
- **Image type:** RGB dermoscopic images (`.jpg`)
- **Official source:** Kaggle (recommended)

### Download Links

- **Kaggle (official):**
  https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

- **Project mirror (Google Drive):**
  https://drive.google.com/drive/folders/1r4kdnYIAATud-PUYCgaq6jjpMrdAD76y?usp=sharing

> Important: Do **not** commit the dataset images to this repository.
> Download the dataset locally and place it under `Datasets/` using the structure below.

### Kaggle Download (CLI)

1. Install and configure Kaggle API credentials:
   https://github.com/Kaggle/kaggle-api#api-credentials

2. Download:
   ```bash
   kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
   ```

3. Unzip and arrange images into the structure below.

### Expected Dataset Structure

Create the following structure inside the project root:

```text
Datasets/
  README.md
  train/
    akiec/
    bcc/
    bkl/
    df/
    mel/
    nv/
    vasc/
  val/
    akiec/
    bcc/
    bkl/
    df/
    mel/
    nv/
    vasc/
  test/
    akiec/
    bcc/
    bkl/
    df/
    mel/
    nv/
    vasc/
```

This layout is compatible with **PyTorch `torchvision.datasets.ImageFolder`**.

### Class Labels

| Label | Description |
|------:|------------|
| `akiec` | Actinic keratoses and intraepithelial carcinoma |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic nevi |
| `vasc` | Vascular lesions |

## Usage Example (PyTorch)

```python
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder("Datasets/train", transform=transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

## Notes

- The dataset is known to have **strong class imbalance**.
- Ensure your splits (`train/`, `val/`, `test/`) are created consistently and do not overlap.

## Acknowledgements / License

- Dataset authors: **HAM10000 contributors**
- Kaggle hosting + documentation:
  https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

This repository is intended for **academic and research purposes**. Please refer to the official dataset page for **licensing and citation** requirements.
 - **Classes:** 7
 - **Image type:** RGB dermoscopic images (`.jpg`)
 - **Official source:** Kaggle (recommended)
 
