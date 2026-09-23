# version1 — MobileNetV2 + Transfer Learning (TensorFlow / Keras)

การทดลองที่ 1 ของโปรเจกต์รู้จำตัวอักษรและตัวเลขภาษาไทย 72 คลาส

> เวอร์ชันนี้ **ไม่ใช่** CNN ที่สร้างจากศูนย์ — ใช้ **MobileNetV2 ที่ pretrained บน
> ImageNet** เป็น backbone แล้ว fine-tune แบบ 2 เฟส พร้อม **Focal Loss** เพื่อรับมือ
> ข้อมูลที่ไม่สมดุล สิ่งที่ต่างจาก [version2](../version2/) คือ backbone
> (MobileNetV2 vs ResNet50), ขนาดภาพเข้าโมเดล (96×96 vs 224×224), เทคนิคจัดการ
> ความไม่สมดุล (Focal Loss vs Balanced Augmentation) และเฟรมเวิร์ก (Keras vs PyTorch)

## ผลลัพธ์

วัดบนชุด validation 12,545 ภาพ ด้วย `best_model.keras`

| ตัวชี้วัด | ค่า |
|---|---|
| Accuracy | **0.8542** |
| Macro F1 | **0.6804** |
| Weighted F1 | 0.8458 |
| Balanced accuracy | 0.7120 |
| Top-5 accuracy | 0.9856 |

ช่องว่างระหว่าง accuracy 0.85 กับ macro F1 0.68 คือผลของความไม่สมดุลของข้อมูลโดยตรง
คลาสใหญ่ฉุดตัวเลขรวมขึ้น ขณะที่ **28 จาก 70 คลาสที่วัดได้ยังมี F1 ต่ำกว่า 0.80**

จุดที่พังชัดที่สุด: คลาส `ว` ถูกทำนายเป็น `า` **323 ครั้ง = 93.6% ของคลาส** (F1 = 0.10)
และมี 7 คลาสที่ F1 = 0 สนิท (`ฉ ฎ ฤ ฬ ฯ ธ ๗`) รายละเอียดครบอยู่ใน
`outputs/reports/04_classification_report.csv`

## ติดตั้ง

ต้องใช้ **Python 3.11** (TensorFlow 2.15 ไม่รองรับ 3.12 ขึ้นไป)

```bash
cd version1
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

ชุดข้อมูลไม่ได้อยู่ใน repo (ขนาดใหญ่) ให้วางไว้ที่ `<repo root>/ThaiCharacter Dataset/round2/`
หรือชี้ path เองด้วยตัวแปรสภาพแวดล้อม

```bash
set THAI_DATASET_ROOT=D:\path\to\round2        # Windows
export THAI_DATASET_ROOT=/path/to/round2       # macOS / Linux
```

## วิธีรัน

รัน notebook ตามลำดับ

```bash
jupyter lab notebooks/
```

| notebook | ทำอะไร |
|---|---|
| `00_data_exploration.ipynb` | สำรวจข้อมูล นับจำนวนภาพต่อคลาส ดูการกระจายขนาดภาพ |
| `01_preprocessing_augmentation.ipynb` | แบ่ง train/val 80:20, สร้าง cache ภาพ 96×96, กำหนด augmentation |
| `02_model_transfer_learning.ipynb` | สร้างโมเดล MobileNetV2, อธิบาย Focal Loss, สาธิต Grad-CAM |
| `03_training.ipynb` | เทรน 2 เฟส (freeze → fine-tune) และพล็อต learning curve |

จากนั้นประเมินผลและสร้าง confusion matrix

```bash
python 04_evaluate_confusion_matrix.py              # ทั้ง validation split
python 04_evaluate_confusion_matrix.py --max-images 400   # ทดลองรันเร็ว
python 04_evaluate_confusion_matrix.py --help       # ดูตัวเลือกทั้งหมด
```

## โครงสร้างโค้ด

```
version1/
  notebooks/00–03            ขั้นตอนหลักของการทดลอง
  04_evaluate_confusion_matrix.py   ประเมินผล + confusion matrix + รายงานต่อคลาส
  src/data_utils.py          สำรวจโฟลเดอร์, label mapping, stratified split
  src/augmentation.py        augmentation สำหรับลายมือไทย (ไม่ใช้ flip)
  src/pipeline.py            cache ภาพเป็น .npz และสร้าง tf.data.Dataset
  src/model_utils.py         MobileNetV2 / EfficientNetB0 builder + Focal Loss
  src/gradcam.py             Grad-CAM สำหรับดูว่าโมเดลมองที่ส่วนไหนของตัวอักษร
  src/viz_utils.py           ฟังก์ชันพล็อตกราฟ (ตั้งฟอนต์ไทยไว้แล้ว)
  label.json                 รหัสคลาส -> ตัวอักษรไทย
  outputs/figures/           กราฟทั้งหมด (.png)
  outputs/reports/           ตารางผลลัพธ์ (.csv / .json)
  outputs/models/            best_model.keras (ไม่ commit — ดู .gitignore)
```

## รายละเอียดทางเทคนิค

**สถาปัตยกรรม** (`src/model_utils.py`)

```
Input 96×96×3
  → Lambda(mobilenet_v2.preprocess_input)
  → MobileNetV2 (include_top=False, weights="imagenet")
  → GlobalAveragePooling2D
  → Dense(256, relu)
  → Dropout(0.3)
  → Dense(72, softmax)
```

**การเทรน 2 เฟส** (`notebooks/03_training.ipynb`)

| เฟส | backbone | learning rate | epochs |
|---|---|---|---|
| 1 | freeze ทั้งหมด | 1e-3 | 8 |
| 2 | unfreeze 30 ชั้นท้าย (BatchNorm ยัง freeze) | 1e-5 | 6 |

callback: `EarlyStopping(patience=3)`, `ModelCheckpoint(save_best_only)`,
`ReduceLROnPlateau(factor=0.5, patience=2)`

**Focal Loss** — เทคนิคที่เลือกใช้จัดการความไม่สมดุล

cross-entropy ปกติให้น้ำหนักทุกตัวอย่างเท่ากัน ไม่ว่าโมเดลจะทายถูกอย่างมั่นใจอยู่แล้ว
หรือไม่ บน dataset ที่คลาสใหญ่สุดมีภาพมากกว่าคลาสเล็กสุดหลายพันเท่า loss จะถูกครอบงำ
ด้วยตัวอย่างง่าย ๆ ของคลาสใหญ่ที่โมเดลทายถูกแล้ว Focal Loss เพิ่มตัวคูณ
`(1 − p_t)^γ` (γ = 2.0) ที่ลดน้ำหนักตัวอย่างง่าย ทำให้ gradient ยังโฟกัสที่ตัวอย่างยาก
และคลาสเล็ก โดยไม่ต้องทิ้งข้อมูล (ต่างจาก undersampling) และไม่ต้องเพิ่มขนาด dataset
(ต่างจาก oversampling)

**การแบ่งข้อมูล**: train/val = 80/20 แบบ stratified (`src/data_utils.stratified_split`)
มีกฎรองรับคลาสที่มีภาพ 1–4 ภาพซึ่ง `sklearn` จัดการไม่ได้

> เวอร์ชันนี้ **ไม่มีชุด test แยก** — ตัวเลขทั้งหมดวัดบน validation ที่ใช้เลือก
> checkpoint ด้วย ถ้าต้องการชุด test ที่ไม่เคยถูกใช้เลย ดู [version2](../version2/)
> ซึ่งกัน test 5% ออกก่อนเป็นอย่างแรก
