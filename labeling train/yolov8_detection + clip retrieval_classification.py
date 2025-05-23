# 0) 필수 설치 (Colab 셀에 한 번만)
!pip install --quiet ultralytics transformers torch torchvision scikit-learn pillow
!pip install ultralytics ftfy regex tqdm git+https://github.com/openai/CLIP.git

from google.colab import drive
drive.mount('/content/drive')

# 2) 임포트 및 경로 설정
import os, csv, numpy as np
from PIL import Image
import torch, clip
from ultralytics import YOLO
from sklearn.linear_model import LogisticRegression

DRIVE       = "/content/drive/MyDrive/prometheus"
TRAIN_IMG   = os.path.join(DRIVE, "dataset/images/train")
TEST_IMG    = os.path.join(DRIVE, "dataset/images/test")
BEST_PT     = os.path.join(DRIVE, "runs/train/exp/weights/best.pt")
CLASSES_TXT = os.path.join(DRIVE, "food-101_classes.txt")  # 100개 한식 메뉴

# 3) 클래스 리스트와 맵 생성
with open(CLASSES_TXT) as f:
    classes = [l.strip() for l in f if l.strip()]
label2idx = {c:i for i,c in enumerate(classes)}

# 4) 디바이스 + 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP 모델
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# 1. YOLOv8 설치 및 드라이브 연결
!pip install ultralytics

import os, shutil

# Data root: 첫 단계 클래스 폴더들이 모여 있는 곳
base_path = '/content/drive/MyDrive/prometheus/archive (30)'

# YOLO용 폴더
train_img = '/content/dataset/images/train'
train_lbl = '/content/dataset/labels/train'
test_img  = '/content/dataset/images/test'

# 기존에 있던 dataset 폴더 완전 삭제 & 재생성
!rm -rf /content/dataset
os.makedirs(train_img, exist_ok=True)
os.makedirs(train_lbl, exist_ok=True)
os.makedirs(test_img,  exist_ok=True)

# (Step B2) 클래스 리스트 로드
with open(CLASSES_TXT) as f:
    classes = [l.strip() for l in f if l.strip()]
label2idx = {c:i for i,c in enumerate(classes)}
print("✔ Classes:", len(classes))

for cls in class_dirs:
    # 2단계 하위 폴더
    cls_folder = os.path.join(base_path, cls, cls)
    if not os.path.isdir(cls_folder):
        # 만약 두 번째 폴더명이 클래스명과 다르면,
        # 그냥 base_path/cls 내부에 파일이 있을 수도 있으니 backup
        cls_folder = os.path.join(base_path, cls)

    files = sorted(os.listdir(cls_folder))

    # .jpg–.txt 쌍만 모으기
    paired = []
    for f in files:
        if f.lower().endswith('.jpg'):
            txt = f[:-4] + '.txt'
            if txt in files:
                paired.append((f, txt))
    # 학습용 30쌍
    for img, txt in paired[:30]:
        shutil.copy(
            os.path.join(cls_folder, img),
            os.path.join(train_img, f'{cls}_{img}')
        )
        shutil.copy(
            os.path.join(cls_folder, txt),
            os.path.join(train_lbl, f'{cls}_{txt}')
        )
    # 나머지 .jpg → test
    learned = {img for img, _ in paired[:30]}
    for f in files:
        if f.lower().endswith('.jpg') and f not in learned:
            shutil.copy(
                os.path.join(cls_folder, f),
                os.path.join(test_img, f'{cls}_{f}')
            )

print('▶ train images count:', len(os.listdir(train_img)))
print('▶ train labels count:', len(os.listdir(train_lbl)))
print('▶ test  images count:', len(os.listdir(test_img)))
# 예상: train ~ (30 × 클래스수), test ~ (나머지)

yaml_path = '/content/dataset.yaml'
with open(yaml_path, 'w') as f:
    f.write(f"""
path: /content/dataset
train: images/train
val: images/train
names: [{", ".join(f'"{c}"' for c in class_dirs)}]
""")

from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.model.names = class_dirs

model.train(
  data=yaml_path,
  epochs=10,
  imgsz=640,
  batch=32,
  lr0=0.001,
  lrf=0.2,
  cos_lr=True,
  augment=True,
  patience=5,
  val=True,
  split=0.15
)

import os

# (1) 30샷으로 라벨링된 이미지 폴더
train_img = '/content/dataset/images/train'

# (2) Food-101 또는 archive(30) 폴더로부터 클래스 리스트 로드
#    만약 food-101_classes.txt 를 쓰고 싶다면 이 줄로:
# with open('/content/drive/MyDrive/prometheus/food-101_classes.txt') as f:
#     classes = [l.strip() for l in f if l.strip()]

#    archive(30) 폴더에 직접 있는 클래스명을 쓰려면:
base_path = '/content/drive/MyDrive/prometheus/archive (30)'
classes = [
    d for d in sorted(os.listdir(base_path))
    if os.path.isdir(os.path.join(base_path, d))
]
print("✔ Classes:", len(classes), classes[:5])

# (3) 클래스명 → 인덱스 맵
label2idx = {c:i for i,c in enumerate(classes)}


import numpy as np
import torch, clip
from PIL import Image
from sklearn.linear_model import LogisticRegression

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

X_train, y_train = [], []
for fn in sorted(os.listdir(train_img)):
    if not fn.lower().endswith(".jpg"): continue
    cls = fn.split("_",1)[0]
    idx = label2idx.get(cls)
    if idx is None: continue

    img = Image.open(os.path.join(train_img, fn)).convert("RGB")
    inp = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(inp)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    X_train.append(feat.cpu().numpy()[0])
    y_train.append(idx)

X_train = np.vstack(X_train)
y_train = np.array(y_train)
print(f"✔ Loaded {X_train.shape[0]} training samples.")

clf = LogisticRegression(max_iter=1000, n_jobs=-1)
clf.fit(X_train, y_train)
print("✅ Few-Shot LogisticRegression trained")

# 루트(현재 작업 디렉터리) 아래에서 .pt 파일 검색
!find . -type f -name "*.pt"

from ultralytics import YOLO
yolo = YOLO('runs/detect/train/weights/best.pt').to(device)

import os
import csv
from PIL import Image
import torch
from ultralytics import YOLO

# ── 1) 모델 로드 ─────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo = YOLO('runs/detect/train/weights/best.pt')  # YOLOv8 불러오기
# clip_model, clip_preprocess, clf, classes 는 기존 그대로

# ── 2) 결과 저장할 CSV 파일 열기 ────────────────────────────────
output_csv = "/content/drive/MyDrive/prometheus/yolo_clip_fewshot_results.csv"
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image_name", "gt_label", "pred_label", "pred_conf",
        "x0","y0","x1","y1"
    ])

    # ── 3) 폴더 내 모든 JPG 처리 ────────────────────────────────
    for fn in sorted(os.listdir(test_img)):
        if not fn.lower().endswith(".jpg"):
            continue

        img_path = os.path.join(test_img, fn)
        img = Image.open(img_path).convert("RGB")

        # ── 4) YOLO 예측 (출력 억제) ───────────────────────────
        try:
            res = yolo.predict(
                source=img_path,     # 입력 이미지
                conf=0.3,            # confidence threshold
                verbose=False,       # 콘솔 출력 억제
                show=False           # 시각화 억제
            )[0]
        except Exception:
            # 읽기 오류 등은 조용히 건너뛰기
            continue

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()

        # ── 5) Crop → CLIP-LR 분류 → CSV 기록 ───────────────
        for (x0, y0, x1, y1), det_conf in zip(boxes, confs):
            if det_conf < 0.3:
                continue

            crop = img.crop((x0, y0, x1, y1))
            inp  = clip_preprocess(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = clip_model.encode_image(inp)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            probs = clf.predict_proba(feat.cpu().numpy())[0]
            idx   = probs.argmax()
            pred  = classes[idx]
            conf  = probs[idx]
            gt    = fn.split("_", 1)[0]

            writer.writerow([
                fn, gt,
                pred, f"{conf:.4f}",
                f"{x0:.1f}", f"{y0:.1f}",
                f"{x1:.1f}", f"{y1:.1f}"
            ])

# ── 6) (선택) 완료 메시지 한 줄만 출력 ────────────────────────
print(f"✅ Inference complete. Results saved to {output_csv}")
