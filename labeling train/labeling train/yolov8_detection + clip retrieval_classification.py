# 0) 필수 설치 (Colab 셀에 한 번만)
!pip install --quiet ultralytics transformers torch torchvision scikit-learn pillow
!pip install ultralytics ftfy regex tqdm git+https://github.com/openai/CLIP.git

# 1) 드라이브 마운트 (필요시)
from google.colab import drive
drive.mount('/content/drive')

# 2) 임포트 및 경로 설정
import os, csv, numpy as np
from PIL import Image
import torch, clip
from ultralytics import YOLO
from sklearn.linear_model import LogisticRegression

# 2) 경로 & 디바이스 설정
DRIVE        = "/content/drive/MyDrive/prometheus"
TRAIN_IMG    = os.path.join(DRIVE, "dataset/images/train")
TEST_IMG     = os.path.join(DRIVE, "dataset/images/test")
YOLO_CHECKPT = os.path.join(DRIVE, "yolo_ab_porridge.pt")   # 저장해 둔 .pt
CLASSES_TXT  = os.path.join(DRIVE, "food-101_classes.txt")  # 100개 메뉴

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
# archive (30) 아래 첫 번째 레벨의 폴더들만 선택
class_dirs = [
    d for d in sorted(os.listdir(base_path))
    if os.path.isdir(os.path.join(base_path, d))
]
print("✔ 클래스 개수:", len(class_dirs), "개")
print("✔ 샘플:", class_dirs[:5])
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

model = YOLO('yolov8s.pt')                     # 작은→중형 모델 업그레이드
model.model.names = class_dirs                  # 이름 매핑 덮어쓰기
model.train(
  data=yaml_path,
  epochs=120,
  imgsz=640,
  batch=32,
  lr0=0.001,
  lrf=0.2,
  cos_lr=True,
  augment=True,
  mosaic=1.0,
  mixup=0.2,
  copy_paste=0.3,
  auto_augment='albumentations',
  patience=30,
  val=True,
  split=0.15
)
!ls runs/detect/train/weights
# best.pt  last.pt
# Google Drive를 이미 마운트했다고 가정
!cp runs/detect/train/weights/best.pt /content/drive/MyDrive/yolo_ab_porridge_best.pt
!cp runs/detect/train/weights/last.pt  /content/drive/MyDrive/yolo_ab_porridge_last.pt
import shutil

shutil.copy(
    'runs/detect/train/weights/best.pt',
    '/content/drive/MyDrive/yolo_ab_porridge_best.pt'
)
shutil.copy(
    'runs/detect/train/weights/last.pt',
    '/content/drive/MyDrive/yolo_ab_porridge_last.pt'
)
# model: 이미 train() 까지 마친 YOLO 객체
model.save('/content/drive/MyDrive/yolo_ab_porridge.pt')

import os

# ─── Step B2 앞에 추가 ──────────────────────────────

# (1) 30샷으로 라벨링된 이미지 폴더
TRAIN_IMG = '/content/dataset/images/train'
TEST_IMG  = '/content/dataset/images/test'

base_path = '/content/drive/MyDrive/prometheus/archive (30)'
classes = sorted([
    d for d in os.listdir(base_path)
    if os.path.isdir(os.path.join(base_path, d))
])
label2idx = {c:i for i,c in enumerate(classes)}
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# 30-shot 데이터로 CLIP 특징 벡터 모으기
X, y = [], []
for fn in sorted(os.listdir(TRAIN_IMG)):
    if not fn.lower().endswith(".jpg"): continue
    cls = fn.split("_",1)[0]
    idx = label2idx.get(cls)
    if idx is None: continue

    img = Image.open(os.path.join(TRAIN_IMG, fn)).convert("RGB")
    inp = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(inp)
        feat = feat / feat.norm(dim=-1, keepdim=True)

    X.append(feat.cpu().numpy()[0])
    y.append(idx)

X = np.vstack(X)
y = np.array(y)

clf = LogisticRegression(max_iter=1000, n_jobs=-1).fit(X, y)
print(f"✔ CLIP-LR trained on {X.shape[0]} samples")
# 루트(현재 작업 디렉터리) 아래에서 .pt 파일 검색
!find . -type f -name "*.pt"
from ultralytics import YOLO

# device 변수는 이미 정의되어 있다고 가정
model_path = '/content/drive/MyDrive/yolo_ab_porridge_best.pt'
yolo = YOLO(model_path).to(device)
import os
import csv
from PIL import Image
import torch
from ultralytics import YOLO

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

import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import math

# 1) 설정
CSV_PATH = "/content/drive/MyDrive/prometheus/yolo_clip_fewshot_results.csv"
TEST_IMG = "/content/dataset/images/test"

# 2) CSV 로드
df = pd.read_csv(CSV_PATH)

# 3) 시각화할 이미지 샘플 선택 (여기선 처음 20개)
sample_imgs = df['image_name'].unique()[:20]
n = len(sample_imgs)

# 4) 그리드 설정 (5열 기준)
cols = 5
rows = math.ceil(n / cols)

# 5) 각 셀 크기 8×8인치로 설정
fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 8))
axes = axes.flatten()

# 6) 이미지 및 박스 그리기
for ax, img_name in zip(axes, sample_imgs):
    img_path = os.path.join(TEST_IMG, img_name)
    img = Image.open(img_path).convert("RGB")
    ax.imshow(img)
    ax.set_title(img_name, fontsize=12)
    ax.axis('off')

    # 해당 이미지 예측 결과 불러오기
    sub = df[df['image_name'] == img_name]
    for _, row in sub.iterrows():
        x0, y0 = row['x0'], row['y0']
        x1, y1 = row['x1'], row['y1']
        # 신뢰도 제거: 음식명만 출력
        label = row['pred_label']
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                             fill=False, linewidth=3, edgecolor='yellow')
        ax.add_patch(rect)
        ax.text(x0, y0, label, color='yellow', fontsize=12,
                verticalalignment='top',
                bbox=dict(facecolor='black', alpha=0.7, pad=2))

# 7) 빈 축 숨기기
for ax in axes[n:]:
    ax.axis('off')

plt.tight_layout()
plt.show()
