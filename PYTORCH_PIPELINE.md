# PyTorch track: ResNet50 + Error-driven Balanced Augmentation

จำแนกตัวอักษร/ตัวเลขภาษาไทย 72 คลาส จากชุด `ThaiCharacter Dataset/round2/`
(62,707 ภาพ, imbalanced 5,025 : 1)

โปรเจกต์นี้มีสอง track ที่แยกกันสมบูรณ์ ไม่ทับไฟล์กัน:

| track | โค้ด | โมเดล | output |
|---|---|---|---|
| TensorFlow (เดิม) | `notebooks/`, `src/`, `04_evaluate_confusion_matrix.py` | MobileNetV2 + focal loss | `outputs/` |
| **PyTorch (อันนี้)** | `torch_pipeline/`, `run_step*.py` | **ResNet50 fine-tune ทั้งตัว** | `outputs_torch/` |

## ติดตั้ง

```bash
uv sync    # ลง torch 2.6.0+cu124 + torchvision ให้เรียบร้อยแล้วใน pyproject.toml
```

## ลำดับการรัน

แต่ละขั้นเซฟผลไว้ให้ขั้นถัดไปอ่าน รันต่อกันตามลำดับนี้

```bash
uv run python run_step1_explore.py          # ~1 นาที   สำรวจ + แบ่ง train/val/test
uv run python run_step2_baseline.py         # ~10 นาที  baseline 5 epochs -> หาคลาสที่ทายผิดบ่อย
uv run python run_step3_augment_plan.py     # ~1 นาที   สร้าง balanced manifest + กราฟ
uv run python run_step4_train.py            # ~4 ชั่วโมง เทรนจริง 10 epochs
uv run python run_step5_evaluate.py         # ~1 นาที   ประเมิน + เทียบก่อน-หลัง augment
uv run python run_step6_test_inference.py   # ~1 นาที   ทำนายชุด test 5%
```

ทุกสคริปต์มี `--help` และมีโหมดทดสอบเร็ว:

```bash
uv run python run_step2_baseline.py --epochs 1 --limit-per-class 20
uv run python run_step3_augment_plan.py --target-count 300
uv run python run_step4_train.py --epochs 1 --max-steps 30
uv run python run_step4_train.py --resume          # ต่อจาก checkpoint ถ้าเทรนหลุด
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
(`outputs/split.csv` ของ track TensorFlow มีแค่ train/val) `run_step1_explore.py`
จึงตัด test 5% แบบ stratified ออกจาก dataset ทั้งก้อนก่อนเป็นอย่างแรก ล็อกด้วย
`SEED = 42` และเขียนลง `outputs_torch/splits/splits.csv` เป็นคอลัมน์ `split`

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
outputs_torch/
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
