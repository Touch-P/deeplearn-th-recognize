# การรู้จำตัวอักษรและตัวเลขภาษาไทย 72 คลาส ด้วย CNN + Transfer Learning

โปรเจกต์นี้เก็บการทดลอง **2 เวอร์ชัน** ที่แยกกันสมบูรณ์ ทั้งคู่แก้โจทย์เดียวกัน
(จำแนกภาพลายมือตัวอักษร/ตัวเลขไทย 72 คลาส จากชุดข้อมูลที่ไม่สมดุล 5,025 : 1)
แต่ใช้สถาปัตยกรรม เฟรมเวิร์ก และเทคนิคจัดการความไม่สมดุลต่างกัน

| | [**version1**](version1/) | [**version2**](version2/) |
|---|---|---|
| Backbone | MobileNetV2 (pretrained ImageNet) | **ResNet50** (pretrained ImageNet) |
| เฟรมเวิร์ก | TensorFlow 2.15 / Keras | **PyTorch 2.6** |
| ภาพเข้าโมเดล | 96 × 96 | **224 × 224** |
| Transfer learning | 2 เฟส: freeze → unfreeze 30 ชั้นท้าย | **fine-tune ทั้งโมเดล** |
| จัดการความไม่สมดุล | Focal Loss (γ = 2.0) | **Error-driven Balanced Augmentation** |
| การแบ่งข้อมูล | train/val = 80/20 | **train/val/test = 76/19/5** |
| ชุด test ที่กันไว้ | ไม่มี | **3,136 ภาพ (5%)** |
| เกณฑ์เลือก best model | val accuracy | **val macro F1** |
| รายละเอียด | [version1/README.md](version1/README.md) | [version2/README.md](version2/README.md) |

## ผลลัพธ์เปรียบเทียบ

วัดบนชุด validation ของแต่ละเวอร์ชัน (72 คลาส)

| ตัวชี้วัด | version1 | version2 | ส่วนต่าง |
|---|---|---|---|
| Accuracy | 0.8542 | **0.9859** | +0.132 |
| **Macro F1** | 0.6804 | **0.9584** | **+0.278** |
| Weighted F1 | 0.8458 | **0.9859** | +0.140 |
| Balanced accuracy | 0.7120 | **0.9860** | +0.274 |
| Top-5 accuracy | 0.9856 | **0.9985** | +0.013 |
| จำนวนภาพที่วัด | 12,545 | 11,913 | |

version2 วัดบนชุด **test 5% ที่ไม่เคยถูกใช้เทรนหรือ augment** ได้อีกชุดหนึ่ง:
accuracy **0.9837** · macro F1 **0.9455** (3,136 ภาพ) — ตัวเลขใกล้กับ validation
แสดงว่าโมเดล generalize ได้จริง ไม่ได้ overfit ไปกับชุดที่ใช้เลือก checkpoint

> **ข้อควรระวังในการอ่านตารางนี้**: ทั้งสองเวอร์ชันแบ่งข้อมูลด้วยกฎที่ต่างกัน
> (version1 แบ่ง 80/20 ไม่มี test, version2 กัน test 5% ออกก่อนแล้วแบ่ง 80/20 จาก
> ที่เหลือ) ชุด validation จึงไม่ใช่ภาพชุดเดียวกันเป๊ะ ส่วนต่างในตารางสะท้อน
> "ผลรวมของทุกอย่างที่เปลี่ยน" (backbone + ความละเอียดภาพ + เทคนิคจัดการความ
> ไม่สมดุล + จำนวน epoch) ไม่ใช่การวัดผลของปัจจัยใดปัจจัยเดียว
>
> การเปรียบเทียบที่คุมตัวแปรได้จริงอยู่ **ภายใน version2** ซึ่งเทียบโมเดลเดียวกัน
> บน split เดียวกัน ก่อน-หลังทำ augmentation: macro F1 0.9538 → 0.9858 (**+0.032**)
> และเฉพาะ 14 คลาสที่ baseline ทายผิดบ่อย F1 ขึ้น **+0.166** ขณะที่คลาสอื่น −0.002

## ชุดข้อมูล

ไม่ได้เก็บใน repo เพราะขนาดใหญ่ (62,707 ภาพ ~200MB) ให้วางไว้ที่

```
<repo root>/ThaiCharacter Dataset/round2/<รหัสคลาส>/*.jpg
```

ทั้งสองเวอร์ชันอ่านข้อมูลชุดเดียวกันจากตำแหน่งนี้ (ไม่ทำสำเนาสองชุด) และรองรับ
การชี้ path เองด้วยตัวแปรสภาพแวดล้อม

```bash
set THAI_DATASET_ROOT=D:\path\to\round2        # Windows
export THAI_DATASET_ROOT=/path/to/round2       # macOS / Linux
```

รหัสคลาสเป็นเลข 3 หลัก (161–249) ซึ่ง map เป็นตัวอักษรไทยด้วย `label.json`
(แต่ละเวอร์ชันมีสำเนาของตัวเอง)

| สถิติ | ค่า |
|---|---|
| จำนวนคลาส | 72 (พยัญชนะ สระ วรรณยุกต์ เลขไทย) |
| จำนวนภาพ | 62,707 |
| คลาสที่มีภาพมากสุด | `า` 5,025 ภาพ |
| คลาสที่มีภาพน้อยสุด | `ฃ` 1 ภาพ |
| ขนาดภาพต้นฉบับ | ประมาณ 12×18 ถึง 30×45 px |

## เริ่มใช้งาน

แต่ละเวอร์ชันมี `requirements.txt` ของตัวเองและ **ไม่ import โค้ดข้ามโฟลเดอร์กัน**
จึงติดตั้งแยก environment ได้

```bash
# version1 — TensorFlow (ต้องใช้ Python 3.11)
cd version1
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# version2 — PyTorch (Python 3.11+, ใส่ --extra-index-url ถ้าต้องการ CUDA)
cd version2
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

หรือถ้าต้องการ environment เดียวที่ลงทั้งสองเวอร์ชันพร้อมกัน (สะดวกตอนพัฒนา)
ใช้ `pyproject.toml` ที่ root กับ [uv](https://docs.astral.sh/uv/)

```bash
uv sync        # ลง TensorFlow 2.15 + PyTorch 2.6 (cu124) ในชุดเดียว
```

## โครงสร้าง repo

```
.
├── README.md                  ไฟล์นี้ — ภาพรวมและตารางเปรียบเทียบ
├── pyproject.toml / uv.lock   environment รวมสำหรับพัฒนา (ทางเลือก)
├── ThaiCharacter Dataset/     ชุดข้อมูล (ไม่อยู่ใน repo) ใช้ร่วมกันสองเวอร์ชัน
│
├── version1/                  การทดลองที่ 1 — MobileNetV2 + Focal Loss (Keras)
│   ├── README.md
│   ├── requirements.txt
│   ├── label.json
│   ├── notebooks/00–03        สำรวจข้อมูล, preprocessing, สร้างโมเดล, เทรน
│   ├── 04_evaluate_confusion_matrix.py
│   ├── src/                   data_utils, augmentation, pipeline, model_utils, gradcam, viz_utils
│   └── outputs/               figures/ reports/ models/
│
└── version2/                  การทดลองที่ 2 — ResNet50 + Balanced Augmentation (PyTorch)
    ├── README.md
    ├── RESULTS.md             รายงานผลแบบเต็ม สร้างอัตโนมัติจากไฟล์ผลลัพธ์
    ├── notebook.ipynb         สรุปทั้งการทดลองพร้อมกราฟ (สำหรับนำเสนอ)
    ├── requirements.txt
    ├── label.json
    ├── run_step1..6.py        6 ขั้นตอนของ pipeline
    ├── run_all.sh             รันทั้งสายต่อเนื่อง
    ├── make_results.py        สร้าง RESULTS.md จากตัวเลขจริง
    ├── torch_pipeline/        config, data, augment, model, reporting, viz
    └── outputs/               figures/ reports/ models/ logs/ splits/
```

ไฟล์โมเดลที่เทรนแล้ว (`.keras` ของ version1 และ `.pth` ของ version2) ไม่ได้ commit
เพราะขนาด 25–284MB ซึ่งเกินลิมิต 100MB ต่อไฟล์ของ GitHub — ดูวิธีตั้ง Git LFS
ใน [`.gitignore`](.gitignore) ถ้าต้องการเก็บไว้ใน repo

## การพัฒนาต่อ

โฟลเดอร์ทั้งสองไม่ผูกกันเลย (ไม่มี import ข้าม ไม่มี path ข้าม นอกจากตำแหน่งของ
ชุดข้อมูลตั้งต้น) ดังนั้นสามารถ:

- เพิ่ม `version3/` เป็นการทดลองถัดไปได้โดยไม่กระทบสองเวอร์ชันเดิม
- ก็อป `version2/` ไปทับ `version1/` ถ้าต้องการยกเลิกการทดลองที่ 1
- ลบเวอร์ชันใดออกได้โดยอีกเวอร์ชันยังรันได้ตามปกติ
