# version2 — ResNet50 + Transfer Learning + Error-driven Balanced Augmentation (PyTorch)

การทดลองที่ 2 ของโปรเจกต์รู้จำตัวอักษรและตัวเลขภาษาไทย 72 คลาส
จากชุด `ThaiCharacter Dataset/round2/` (62,707 ภาพ, ไม่สมดุล 5,025 : 1)

เทียบกับ [version1](../version1/) (MobileNetV2 96×96 + Focal Loss, Keras)
เวอร์ชันนี้เปลี่ยนสี่อย่าง: **backbone เป็น ResNet50**, **ภาพเข้าโมเดล 224×224**,
**จัดการความไม่สมดุลด้วย balanced augmentation แทน Focal Loss** และ
**เฟรมเวิร์กเป็น PyTorch**

## ผลลัพธ์

| ชุดข้อมูล | Accuracy | **Macro F1** | Weighted F1 | Balanced Acc | Top-5 | จำนวนภาพ |
|---|---|---|---|---|---|---|
| Validation | 0.9859 | **0.9584** | 0.9859 | 0.9860 | 0.9985 | 11,913 |
| **Test 5%** (ไม่เคยถูกเทรนหรือ augment) | **0.9837** | **0.9455** | 0.9835 | 0.9707 | 0.9990 | 3,136 |

**ผลของ Error-driven Balanced Augmentation** (เทียบกับ baseline ที่ไม่ augment)

| กลุ่มคลาส | Macro F1 ก่อน | Macro F1 หลัง | ส่วนต่าง |
|---|---|---|---|
| ทุกคลาส | 0.9538 | 0.9858 | **+0.032** |
| เฉพาะ 14 คลาสที่ baseline ทายผิดบ่อย | 0.7922 | 0.9582 | **+0.166** |
| คลาสอื่น ๆ | – | – | −0.002 |

ประโยชน์ไปลงที่คลาสเล็กตามที่ออกแบบไว้ โดยแทบไม่กระทบคลาสที่เดิมทำได้ดีอยู่แล้ว
ขณะที่ accuracy รวมขยับเพียง +0.001 — ถ้าวัดด้วย accuracy อย่างเดียวจะสรุปผิดว่า
augmentation ไม่ช่วยอะไร

รายงานผลแบบเต็ม: [RESULTS.md](RESULTS.md) · สรุปพร้อมนำเสนอ: [notebook.ipynb](notebook.ipynb)

## ติดตั้ง

```bash
cd version2
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

ที่ทดสอบไว้คือ torch 2.6.0+cu124 บน RTX 3050 Laptop 4GB (Python 3.11)
ถ้าไม่มี GPU ตัดส่วน `--extra-index-url` ออกได้ แต่การเทรนจะช้ากว่ามาก

ชุดข้อมูลไม่ได้อยู่ใน repo ให้วางไว้ที่ `<repo root>/ThaiCharacter Dataset/round2/`
หรือชี้ path เองด้วย `THAI_DATASET_ROOT`

```bash
set THAI_DATASET_ROOT=D:\path\to\round2        # Windows
export THAI_DATASET_ROOT=/path/to/round2       # macOS / Linux
```

## ลำดับการรัน

แต่ละขั้นเซฟผลไว้ให้ขั้นถัดไปอ่าน รันต่อกันตามลำดับนี้

```bash
python run_step1_explore.py          # ~1 นาที   สำรวจ + แบ่ง train/val/test
python run_step2_baseline.py         # ~15 นาที  baseline 3 epochs -> หาคลาสที่ทายผิดบ่อย
python run_step3_augment_plan.py     # ~1 นาที   สร้าง balanced manifest + กราฟ
python run_step4_train.py            # ~5.7 ชม.  เทรนจริง 10 epochs (บน RTX 3050)
python run_step5_evaluate.py         # ~1 นาที   ประเมิน + เทียบก่อน-หลัง augment
python run_step6_test_inference.py   # ~1 นาที   ทำนายชุด test 5%
python make_results.py               # สร้าง RESULTS.md จากตัวเลขจริง
```

หรือรันทั้งสายต่อเนื่องด้วย `bash run_all.sh` (หยุดทันทีถ้ามีขั้นใดพลาด
และเก็บ log แยกไฟล์ต่อขั้นใน `outputs/logs/`)

ทุกสคริปต์มี `--help` และมีโหมดทดสอบเร็ว:

```bash
python run_step2_baseline.py --epochs 1 --limit-per-class 20
python run_step3_augment_plan.py --target-count 300
python run_step4_train.py --epochs 1 --max-steps 30
python run_step4_train.py --resume          # ต่อจาก checkpoint ถ้าเทรนหลุด
```

พารามิเตอร์ทั้งหมด (path, seed, batch size, lr, epoch, เกณฑ์เลือกคลาสอ่อนแอ)
อยู่ใน `torch_pipeline/config.py` ไฟล์เดียว

## โครงสร้างโค้ด

```
torch_pipeline/
  config.py      path + hyperparameter ทุกตัว, set_seed(), get_device()
  data.py        สำรวจโฟลเดอร์, split 3 ทาง, balanced manifest, Dataset/DataLoader
  augment.py     transform 3 ระดับ (light/normal/intense) สำหรับลายมือไทย
  model.py       ResNet50 builder + train/eval loop + checkpoint + log
  reporting.py   confusion matrix, classification report, เลือกคลาสอ่อนแอ
  viz.py         กราฟทุกใบ (ฟอนต์ไทย, เซฟ .png dpi 300)
run_step1..6.py  ขั้นตอนตามสเปก 5 ข้อ
```

## การตัดสินใจที่ต่างจากโจทย์ และเหตุผล

### 1. ชุด test 5% ต้องสร้างขึ้นใหม่

โจทย์บอกว่า "มีชุด test แยกต่างหาก 5%" แต่ในโปรเจกต์ไม่มีไฟล์ไหนที่เป็นชุด test
(`version1/outputs/split.csv` ของการทดลองที่ 1 มีแค่ train/val) `run_step1_explore.py`
จึงตัด test 5% แบบ stratified ออกจาก dataset ทั้งก้อนก่อนเป็นอย่างแรก ล็อกด้วย
`SEED = 42` และเขียนลง `outputs/splits/splits.csv` เป็นคอลัมน์ `split`

**การกันชุด test ออกจริงตรวจสอบได้จาก**: manifest ของ augmentation สร้างจาก
แถวที่ `split == "train"` เท่านั้น (`run_step3_augment_plan.py` มี `assert` กำกับ)
และ `run_step4_train.py` validate ด้วยแถวที่ `split == "val"` เท่านั้น

จำนวนที่ได้: train 47,658 (76%) / val 11,913 (19%) / test 3,136 (5%)

**2 คลาสไม่มีภาพในชุด test เลย** คือ `163 ฃ` และ `177 ฑ` เพราะทั้งคลาสมีภาพแค่
1 ภาพ (ไปอยู่ train) — เรื่องนี้ต้องเขียนกำกับในรายงาน ไม่ใช่ bug

### 2. รูปแบบไฟล์ผลทำนาย

โจทย์บอก "บันทึกผลตาม format ที่โจทย์กำหนด" แต่ไม่ได้ระบุ format มาด้วย
`run_step6_test_inference.py` จึงเขียน `06_test_predictions.csv` ที่มีคอลัมน์
ครอบคลุมทุกแบบที่พบทั่วไป (`filename`, `true_class_id`, `pred_class_id`,
`pred_char`, `confidence`, `correct`, top-3) แล้วตัดคอลัมน์ที่ไม่ต้องการออกได้
ถ้าอาจารย์ให้โฟลเดอร์ test มาแยก ใช้ `--test-dir "path"` ได้เลย (รองรับทั้งแบบ
มีโฟลเดอร์ย่อยเป็นคลาส และแบบภาพกองเดียวไม่มี label)

### 3. augmentation ไม่เขียนไฟล์ภาพลงดิสก์ แต่ใช้ manifest + seed ที่ล็อกไว้

balance ทุกคลาสถึง 3,819 ภาพ = 274,968 ภาพต่อ epoch ถ้าเซฟเป็น .jpg ที่ 224×224
จะกินดิสก์ ~20GB และเสียเวลาเขียน/อ่านฟรี ๆ

`build_balanced_manifest()` จึงสร้างตารางที่หนึ่งแถว = หนึ่งภาพที่โมเดลจะเห็น
โดยเก็บ (ภาพต้นฉบับ, ระดับความเข้ม, seed ประจำภาพ) แล้วแปลงตอน `__getitem__`
ภายใน `torch.random.fork_rng()` ที่ตั้ง seed จาก manifest

ผลที่ได้ **เทียบเท่ากับการมีชุดภาพ augment จริงหนึ่งชุดที่คงที่** (ไม่ใช่การสุ่ม
ใหม่ทุก epoch แบบ on-the-fly ทั่วไป) — `FIXED_AUGMENTATION = True` ใน config
ถ้าต้องการสุ่มใหม่ทุก epoch ใช้ `--fresh-aug-each-epoch`

กราฟ "ก่อน-หลัง augment" นับจาก manifest ซึ่งคือจำนวนภาพจริงที่โมเดลเห็นต่อ epoch

### 4. ค่าที่วัดได้บนเครื่องนี้ (RTX 3050 Laptop 4GB) — สำคัญถ้าย้ายเครื่อง

**`channels_last` ทำให้การเทรนช้าลง 8 เท่า** บน GPU ตัวนี้ ซึ่งตรงข้ามกับที่
เอกสาร PyTorch แนะนำทั่วไป วัดจริงด้วย ResNet50 @224 bs16 AMP:

| memory format | train | forward เท่านั้น |
|---|---|---|
| `channels_last` | 24 img/s | 675 img/s |
| contiguous (NCHW) | **189 img/s** | 556 img/s |

(ไม่ใช่ปัญหา VRAM ล้น — peak allocation แค่ 1.7GB จาก 4GB และการ์ดทำ
fp16 matmul ได้ 13.2 TFLOPS ปกติ)

จึงตั้ง `CHANNELS_LAST_TRAIN = False` และเปิดเฉพาะตอน inference ล้วน ๆ
(`CHANNELS_LAST_EVAL = True`) **ถ้าย้ายไปรัน GPU ตัวอื่นควรวัดซ้ำ** เพราะผล
อาจกลับกัน

batch size ที่วัดแล้วดีสุดคือ 48 (200 img/s, VRAM peak 2.3GB) จึงเป็นค่าเริ่มต้น

### 5. ปรับพิสัย augmentation หลังดูภาพที่สร้างออกมาจริง

รอบแรกตั้ง `scale=(0.85, 1.15)` + `translate=0.10` + `ColorJitter(0.30)` แล้วเปิด
`03_augmentation_examples.png` ดู พบสองปัญหา:

- **ซูมเข้าเกิน 1.05 ทำให้เส้นถูกตัดหายที่ขอบ** เพราะตัวอักษรกินพื้นที่เกือบเต็ม
  เฟรมอยู่แล้ว การตัดเส้นหัวหรือหางทิ้งอาจเปลี่ยนตัวอักษรเป็นตัวอื่น = ป้อน
  label ผิด เพดาน scale จึงเหลือ 1.05 (ซูมออกยังปลอดภัยเพราะได้ขอบขาวเพิ่ม)
- **brightness/contrast 0.30 ทำให้พื้นหลังกลายเป็นเทากลางภาพ** ซึ่งไม่เคย
  เกิดใน val/test ที่พื้นหลังขาวเสมอ = เทรนบน distribution ที่ผิด ลดเหลือ 0.20

### 6. ห้าม flip — และเหตุผลที่ต้องเขียนในรายงาน

ตัวอักษรไทยมีทิศทาง พลิกซ้าย-ขวาแล้วได้รูปที่เป็นอักขระอื่นหรือไม่มีในภาษา
(เช่น "ก" พลิกแล้วคล้าย "ฎ/ฏ") การ flip คือการป้อน label ผิดให้โมเดลโดยตรง
`augment.py` จึงไม่มี `RandomHorizontalFlip`/`RandomVerticalFlip` เลย

## ไฟล์ output ที่ได้ (ใช้ในรายงาน/สไลด์)

```
outputs/
  figures/
    01_class_distribution.png              bar chart จำนวนภาพต่อคลาส
    01_split_per_class.png                 แท่งซ้อน train/val/test ต่อคลาส
    01_sample_images.png                   ตัวอย่าง 4 ภาพ × 72 คลาส
    02_baseline_confusion_matrix.png       CM ของ baseline (+ _normalized)
    02_baseline_recall_per_class.png       recall รายคลาส เรียงต่ำ->สูง
    03_counts_before_after_augmentation.png  ก่อน-หลัง augment (ชื่อคลาสสีแดง = คลาสอ่อนแอ)
    03_augmentation_examples.png           ต้นฉบับ vs augment 5 แบบ
    04_learning_curves.png                 loss/accuracy train-val ตาม epoch
    05_confusion_matrix_val.png            CM โมเดลสุดท้าย (+ _normalized)
    05_f1_per_class_val.png                F1 รายคลาส
    05_recall_before_after_augmentation.png  recall ก่อน-หลัง (★ = คลาสอ่อนแอ)
    06_test_confusion_matrix.png           CM ชุด test 5% (+ _normalized)
  reports/     CSV/JSON ทุกตาราง (utf-8-sig เปิดใน Excel ไทยไม่เพี้ยน)
  models/      best_model.pth, last_checkpoint.pth, baseline_model.pth, class_to_idx.json
  splits/      splits.csv, train_manifest.csv.gz
  logs/        02_baseline_log.jsonl, 04_training_log.jsonl (หนึ่งบรรทัดต่อ epoch)
```
