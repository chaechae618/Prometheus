# 1. YOLOv8 설치 및 드라이브 연결
!pip install ultralytics

from google.colab import drive
drive.mount('/content/drive')

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

model = YOLO('yolov8n.pt')
model.train(
  data=yaml_path,
  epochs=80,
  imgsz=640,
  batch=16,
  lr0=0.01,
  lrf=0.1,
  cos_lr=True,
  augment=True,
  mosaic=1.0,
  mixup=0.1,
  copy_paste=0.2,
  patience=20,
  # 검증 수행 & 비율 지정
  val=True,        # 검증 수행 여부
  split=0.15       # 학습 데이터의 15%를 val로 분리
)

from PIL import Image
from IPython.display import display

# 1) 예측만 하고 저장은 건너뛰기
results = model.predict(
    source=test_img,
    save=False,   # 저장 생략
    conf=0.25
)

# 2) 첫 5개 결과만 화면에 렌더링
for i, res in enumerate(results[:5]):
    # res.plot()이 BGR numpy array 리턴 → RGB로 변환
    im = res.plot()[:, :, ::-1]
    display(Image.fromarray(im))
