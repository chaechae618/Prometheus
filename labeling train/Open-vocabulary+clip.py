# 1) CLIP 및 numpy 임포트
import torch, clip
import numpy as np
from PIL import Image
import os, csv

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 2) Food-101 클래스 사전 로드 & 임베딩
classes_path = "/content/drive/MyDrive/prometheus/food-101_classes.txt"
with open(classes_path, "r") as f:
    food101_labels = [l.strip() for l in f if l.strip()]

# 텍스트 토큰화 → 임베딩 → L2 정규화
text_tokens = clip.tokenize(food101_labels).to(device)            # [101, token_len]
with torch.no_grad():
    text_feats = model.encode_text(text_tokens)                   # [101, D]
    text_feats /= text_feats.norm(dim=-1, keepdim=True)

# 3) Retrieval 함수 정의
def clip_retrieve(crop_img, top_k=1):
    """
    crop_img: PIL.Image 크롭
    top_k: 반환할 후보 개수 (여기선 1로 설정)
    반환: [(label, similarity), ...]
    """
    img_input = preprocess(crop_img).unsqueeze(0).to(device)       # [1,3,H,W]
    with torch.no_grad():
        img_feat = model.encode_image(img_input)                  # [1, D]
        img_feat /= img_feat.norm(dim=-1, keepdim=True)

    # 코사인 유사도 계산
    sims = (img_feat @ text_feats.T).squeeze(0).cpu().numpy()      # [101]
    idxs = np.argpartition(-sims, top_k-1)[:top_k]                 # top_k 인덱스
    topk = sorted(
        [(food101_labels[i], float(sims[i])) for i in idxs],
        key=lambda x: x[1], reverse=True
    )
    return topk

# 4) 기존 OWL-ViT detection + CLIP Retrieval 통합
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# 이미 로드된 OWL-ViT 프로세서/모델이 없다면 다시 로드
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
ovd_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)

def owlvit_detect_boxes(img_pil, threshold=0.3):
    inputs = processor(text=["food"], images=img_pil, return_tensors="pt").to(device)
    outputs = ovd_model(**inputs)
    target_sizes = torch.tensor([img_pil.size[::-1]]).to(device)
    results = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]
    return results["boxes"].cpu().numpy()  # [[x0,y0,x1,y1], ...]

# 5) inference & CSV 저장
base_path  = "/content/drive/MyDrive/prometheus/archive (30)"
output_dir = "/content/drive/MyDrive/ovd_clip_retrieval"
os.makedirs(output_dir, exist_ok=True)
csv_path   = os.path.join(output_dir, "retrieval_results.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["class_folder","image_name","pred_label","score","x0","y0","x1","y1"])

    for cls in sorted(os.listdir(base_path)):
        cls_dir = os.path.join(base_path, cls, cls)
        if not os.path.isdir(cls_dir):
            cls_dir = os.path.join(base_path, cls)
        for img_name in sorted(os.listdir(cls_dir)):
            if not img_name.lower().endswith(".jpg"):
                continue
            img_path = os.path.join(cls_dir, img_name)
            img = Image.open(img_path).convert("RGB")

            # 5.1) 영역 감지 (food 쿼리)
            boxes = owlvit_detect_boxes(img, threshold=0.25)

            # 5.2) 각 박스별로 Retrieval
            for (x0,y0,x1,y1) in boxes:
                crop = img.crop((x0,y0,x1,y1))
                top1 = clip_retrieve(crop, top_k=1)[0]   # (label, sim)
                writer.writerow([
                    cls, img_name,
                    top1[0], f"{top1[1]:.4f}",
                    f"{x0:.1f}", f"{y0:.1f}", f"{x1:.1f}", f"{y1:.1f}"
                ])
print("✅ Retrieval 결과 저장 완료:", csv_path)
