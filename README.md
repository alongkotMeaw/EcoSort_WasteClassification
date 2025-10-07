
# EcoSort Waste Classification

## Overview

EcoSort เป็นระบบวิสัยทัศน์คอมพิวเตอร์สำหรับคัดแยกขยะครัวเรือน 21 คลาส โดยบอกผลลัพธ์ถังหลัก 5 ประเภท ได้แก่ Recyclable, General, Biodegradable, Hazardous และ E-waste ภายในรีโพนี้มีสคริปต์ฝึกโมเดลสองสถาปัตยกรรม (ResNet50 และ EfficientNet-B1), ชุดข้อมูลตัวอย่าง, และเว็บแอป Flask ที่ให้ผู้ใช้ถ่ายภาพหรืออัปโหลดไฟล์เพื่อรับคำแนะนำการทิ้งที่เหมาะสม

## Repository Layout

- `Code/` รวมสคริปต์ฝึก เทสต์ และโค้ดเว็บแอป
  - `RestNet50/train/` ท่อฝึก ResNet50 แบบ stratified 5-fold พร้อมสคริปต์เอนเซมเบิล
  - `EfficientB0/train/` ท่อฝึก EfficientNet-B1 (3-fold) และสคริปต์ประเมิน
  - `webpage/web/` โค้ด Flask UI, ไฟล์ข้อความคำแนะนำผู้ใช้
- `Dataset_ClusterGroup_Version/` โครงสร้างตัวอย่างที่แบ่งเป็น `train/`, `val/`, `test/` สำหรับ 21 คลาส
- `Raw Data/` ภาพต้นฉบับที่ผ่านการคัดโดยมนุษ โดยคัดจากdataset ต่าง ๆ ก่อนนำไป preprocess
- `Resources/` เอกสารสรุปสถาปัตยกรรม งานวิจัยที่ใช้ประกอบในโปรเจ็กต์

## Dataset

สคริปต์ทุกตัวคาดหวังชุดข้อมูลที่จัดโฟลเดอร์ย่อยตามคลาสภายใต้ `train/`, `val/`, `test/` ภาพอินพุตต้องเป็น RGB ขนาด 224x224 สำหรับ ResNet50 และ 260x260 สำหรับ EfficientNet-B1 ควรบาลานซ์จำนวนภาพต่อคลาสให้ใกล้เคียงกัน และตั้งชื่อโฟลเดอร์ให้ตรงกับ mapping ที่ใช้ในโปรเจ็กต์

## Models and Training

### ResNet50 Pipeline

- สคริปต์หลัก: `Code/RestNet50/train/train_kfold_resnet50.py`
- เริ่มจากสุ่มน้ำหนัก (ไม่มีพรีเทรน) แล้วฝึกต่อด้วย stratified k-fold 5 ส่วน
- Augmentation: resize 224x224, random horizontal flip, color jitter และ normalize ตามสถิติ ImageNet
- ใช้ Adam (lr 1e-4), batch size 32 และ early stopping patience 5
- บันทึกเช็คพอยต์ รายงาน CSV และ confusion matrix ต่อ fold ไว้ใน `Code/RestNet50/output/`
- ผลลัพธ์เฉลี่ย macro-F1 ต่อ fold ประมาณ 0.98 (accuracy validation 97.7%-98.3%)

### EfficientNet-B1 Pipeline

- สคริปต์หลัก: `Code/EfficientB0/train/train_kfold_log_b3_mixup_weighted.py`
- ฝึกแบบ 3-fold พร้อมคลาสเวทและ logging สำหรับ mixup augmentation (เริ่มจากน้ำหนักสุ่มในสคริปต์นี้เช่นกัน)
- อินพุต 260x260 พร้อม augmentation เทียบเคียงท่อ ResNet50
- เอาต์พุตโมเดลและรายงานอยู่ใน `Code/EfficientB0/output/`

## Evaluation Workflow

- สคริปต์เอนเซมเบิล ResNet50: `Code/RestNet50/train/TestRestnet50_Kfold.py`
- สคริปต์เอนเซมเบิล EfficientNet-B1: `Code/EfficientB0/train/test_ensemble_efficientnet.py`
- ปรับ `DATA_DIR` และพาธเช็คพอยต์ให้ตรงก่อนรัน จากนั้นตรวจผลรายงาน CSV และ confusion matrix ในโฟลเดอร์ `output/`

## Web Application

### Setup

1. สร้าง virtual environment ที่ `Code/webpage/web`
2. ติดตั้ง dependencies จาก `requirements.txt`
3. ตรวจสอบว่า `MODEL_PATHS` ใน `main/main.py` ชี้ไปที่น้ำหนัก ResNet50
4. รัน `python main/main.py` เพื่อเปิดเซิร์ฟเวอร์ (default port 5000)
5. ถ้าต้องการให้เข้าถึงเซิร์ฟเวอร์จากภายนอก ให้รัน **ngrok** ด้วยคำสั่ง ngrok http 5000

### Inference Flow

เมื่อผู้ใช้อัปโหลดภาพ ระบบจะบันทึกไฟล์ ทำ preprocess แล้วรันเอนเซมเบิล ResNet50 เพื่อให้ความน่าจะเป็น 21 คลาส ผลลัพธ์ถูกแม็ปสู่กลุ่มถังขยะพร้อมดึงข้อความคำแนะนำจาก `info_txt/`

## Research References

- *Solid Waste Classification Using Modified ResNet-50 Model with Transfer Learning Approach* (2023) — PDF อยู่ใน `Resources/SOLID_WASTE_CLASSIFICATION_USING_MODIFIED_RESNET_50_MODEL_WITH_TRANSFER_LEARNING_APPROACH.pdf` นำเสนอการประยุกต์ ResNet-50 กับงานคัดแยกขยะแบบสองกลุ่ม (ใช้เป็นแรงบันดาลใจ แม้เวอร์ชันปัจจุบันฝึกจากสุ่มน้ำหนัก)
- *EfficientNet-Based Deep Learning Model for Advanced Waste Classification* (2024) — PDF อยู่ใน `Resources/EfficientNet-Based Deep Learning Model for Advanced Waste Classification.pdf` ศึกษาการปรับ EfficientNet สำหรับขยะแบบหลายคลาสและผลลัพธ์เทียบกับสถาปัตยกรรมอื่น

## Dataset Credits

EcoSort ใช้ชุดข้อมูลผสมจากหลายแหล่งต่อไปนี้:

- **BDWaste - A Comprehensive Image Dataset for Smart Waste Classification** — (https://github.com/BDWaste/BDWaste) ชุดภาพสาธารณะ MIT License ครอบคลุมขยะรีไซเคิล ย่อยสลายได้ และอันตราย ใช้เป็นฐานสำหรับหมวดหลัก
- **Waste Classification Data (Kaggle)** — (https://www.kaggle.com/datasets/techsash/waste-classification-data) รวมภาพขยะครัวเรือนประเภทแก้ว โลหะ กระดาษ พลาสติก นำมาปรับขนาดและรีเลเบลให้ตรงกับ 21 คลาส
- **TrashNet / TrashBox Aggregations** — (https://github.com/open-trash/TrashBox) เติมภาพขยะทั่วไปและรีไซเคิล พร้อมคัด duplicate ออกก่อนใช้งาน
- **Dense Waste Segmentation Dataset (DWSD)** — (https://github.com/DWSD/Dense-Waste-Segmentation) ใช้ภาพบางส่วนสำหรับหมวด E-waste และขยะอันตราย โดยเลือกเฉพาะครอปที่พร้อมสำหรับการจำแนก
- **RealWaste (self-collected photos)** — ภาพที่ทีม EcoSort ถ่ายเองและดึงจากสคริปต์ Bing (ดู `Code/find_some_class/real_images_bing_colab_fixed.ipynb`) เพื่อเพิ่มคลาสที่ขาด เช่น สายชาร์จ หูฟัง แบตเตอรี่พกพา
- **Thai Waste Category Guidelines** — เอกสารกรมทรัพยากรธรรมชาติและสิ่งแวดล้อมที่ใช้กำหนดการแม็ปผลคาดการณ์สู่ 5 กลุ่มถัง

ชุดข้อมูลภายนอกทุกชุดยังคงอยู่ภายใต้สัญญาอนุญาตต้นทาง AIVIA แจกจ่ายเฉพาะข้อมูลที่ผ่านการประมวลผลเพื่อการศึกษาและวิจัยเท่านั้น

## How to Reproduce

1. เตรียมสภาพแวดล้อม Python ที่ติดตั้ง PyTorch, torchvision, pandas, scikit-learn, matplotlib, seaborn, tqdm และ efficientnet-pytorch
2. แก้ค่า `DATA_DIR` ในสคริปต์ฝึกและทดสอบให้ตรงกับที่เก็บข้อมูลของคุณ
3. รันสคริปต์ฝึกตามโมเดล เช่น `python Code/RestNet50/train/train_kfold_resnet50.py`
4. ตรวจสอบรายงานและเช็คพอยต์ในโฟลเดอร์ `output/` เพื่อนำไปใช้กับสคริปต์เอนเซมเบิลหรือเว็บแอป
