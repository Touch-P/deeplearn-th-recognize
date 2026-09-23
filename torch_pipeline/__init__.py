"""PyTorch pipeline: ResNet50 + Error-driven Balanced Augmentation
สำหรับจำแนกตัวอักษร/ตัวเลขภาษาไทย 72 คลาส

รันตามลำดับ (แต่ละขั้นเซฟผลไว้ให้ขั้นถัดไปอ่าน):
  run_step1_explore.py        สำรวจข้อมูล + แบ่ง train/val/test
  run_step2_baseline.py       baseline 5 epochs -> หาคลาสที่ทายผิดบ่อย
  run_step3_augment_plan.py   สร้าง balanced manifest + กราฟก่อน-หลัง
  run_step4_train.py          เทรนจริง 10 epochs
  run_step5_evaluate.py       ประเมิน + เทียบก่อน-หลัง augmentation
  run_step6_test_inference.py ทำนายชุด test 5%
"""
