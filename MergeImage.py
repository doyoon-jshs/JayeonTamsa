import os
import math
import numpy as np
from PIL import Image
import exifread

# 설정값
ALTITUDE_M = 50
PIXEL_RESOLUTION_CM = 2.5
PIXEL_RESOLUTION_M = PIXEL_RESOLUTION_CM / 100
OVERLAP_X = 0.7
OVERLAP_Y = 0.8
MAX_WIDTH = 10000
MAX_HEIGHT = 10000

# 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(base_dir, "input_jpgs")
output_file = os.path.join(base_dir, "gps_mosaic_blended.jpg")

# GPS 추출 함수
def get_gps(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f, details=False)
        def to_deg(value):
            d, m, s = [float(x.num) / float(x.den) for x in value.values]
            return d + m / 60 + s / 3600
        try:
            lat = to_deg(tags["GPS GPSLatitude"])
            if tags["GPS GPSLatitudeRef"].values != "N":
                lat = -lat
            lon = to_deg(tags["GPS GPSLongitude"])
            if tags["GPS GPSLongitudeRef"].values != "E":
                lon = -lon
            return lat, lon
        except:
            return None

# 이미지 및 GPS 정보 수집
images_with_gps = []
for fname in sorted(os.listdir(input_folder)):
    if fname.lower().endswith(".jpg"):
        path = os.path.join(input_folder, fname)
        gps = get_gps(path)
        if gps:
            images_with_gps.append((path, gps))

if not images_with_gps:
    print("GPS 포함된 이미지가 없습니다.")
    exit()

# 좌표 범위 계산
latitudes = [gps[0] for _, gps in images_with_gps]
longitudes = [gps[1] for _, gps in images_with_gps]
min_lat, max_lat = min(latitudes), max(latitudes)
min_lon, max_lon = min(longitudes), max(longitudes)
center_lat = sum(latitudes) / len(latitudes)
meters_per_deg_lat = 111_000
meters_per_deg_lon = 111_000 * math.cos(math.radians(center_lat))

# 이미지 크기 및 자를 영역 계산
sample_img = Image.open(images_with_gps[0][0])
img_w, img_h = sample_img.size
crop_w = int(img_w * (1 - OVERLAP_X))
crop_h = int(img_h * (1 - OVERLAP_Y))
offset_x = (img_w - crop_w) // 2
offset_y = (img_h - crop_h) // 2
crop_box = (offset_x, offset_y, offset_x + crop_w, offset_y + crop_h)

# 픽셀 위치 계산
positions = []
for path, (lat, lon) in images_with_gps:
    dx = (lon - min_lon) * meters_per_deg_lon
    dy = (max_lat - lat) * meters_per_deg_lat
    x = int(dx / PIXEL_RESOLUTION_M)
    y = int(dy / PIXEL_RESOLUTION_M)
    positions.append((path, x, y))

# 캔버스 크기 계산
max_x = max(x for _, x, _ in positions)
max_y = max(y for _, _, y in positions)
canvas_w = max_x + crop_w
canvas_h = max_y + crop_h

print(f"캔버스 크기: {canvas_w} x {canvas_h}")
print(f"자를 영역: {crop_w} x {crop_h}")

# NumPy로 빈 캔버스 + 마스크 생성
canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
mask = np.zeros((canvas_h, canvas_w), dtype=np.float32)

# 블렌딩 적용하며 이미지 붙이기
for path, x, y in positions:
    img = Image.open(path).crop(crop_box).convert('RGB')
    arr = np.asarray(img).astype(np.float32)
    h, w, _ = arr.shape

    # 범위
    x1, y1 = x, y
    x2, y2 = x + w, y + h

    # 해당 위치에 블렌딩
    canvas[y1:y2, x1:x2, :] += arr
    mask[y1:y2, x1:x2] += 1.0

    print(f" {os.path.basename(path)} → ({x}, {y})")

# 마스크를 나눠서 평균 내기 (블렌딩)
mask = np.clip(mask, 1e-3, None)[:, :, np.newaxis]
blended = (canvas / mask).astype(np.uint8)

# PIL 이미지로 변환
final_image = Image.fromarray(blended)

# 자동 축소
scale = min(MAX_WIDTH / canvas_w, MAX_HEIGHT / canvas_h, 1.0)
if scale < 1.0:
    resized = final_image.resize((int(canvas_w * scale), int(canvas_h * scale)), resample=Image.LANCZOS)
    resized.save(output_file, format="JPEG", quality=90)
    print(f"축소된 JPG 저장 완료: {output_file} ({resized.size[0]} x {resized.size[1]})")
else:
    final_image.save(output_file, format="JPEG", quality=90)
    print(f"JPG 저장 완료: {output_file} ({canvas_w} x {canvas_h})")
