"""Ілюстрація до тез: варіанти СХЕМАТИЧНОГО ескізу (створений програмно, не з вибірки).
Перетворення ідентичні тим, що використані в experiment.py."""
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

CANVAS = 256
rng = np.random.default_rng(42)

def paste_center(obj):
    c = np.full((CANVAS, CANVAS), 255, np.uint8)
    h, w = obj.shape
    y0, x0 = (CANVAS - h) // 2, (CANVAS - w) // 2
    c[y0:y0 + h, x0:x0 + w] = obj
    return c

def crop_object(img, thr=200):
    ys, xs = np.where(img < thr)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def shift(img, d):
    return cv2.warpAffine(img, np.float32([[1, 0, d], [0, 1, d]]), (CANVAS, CANVAS), borderValue=255)

def scale(img, s):
    return cv2.warpAffine(img, cv2.getRotationMatrix2D((CANVAS / 2, CANVAS / 2), 0, s), (CANVAS, CANVAS), borderValue=255)

def add_noise_lines(img, n):
    out = img.copy()
    for _ in range(n):
        x1, y1 = rng.integers(20, CANVAS - 20, 2)
        ang = rng.uniform(0, 2 * np.pi); L = rng.integers(20, 60)
        x2 = int(np.clip(x1 + L * np.cos(ang), 0, CANVAS - 1)); y2 = int(np.clip(y1 + L * np.sin(ang), 0, CANVAS - 1))
        cv2.line(out, (int(x1), int(y1)), (x2, y2), 170, 1, cv2.LINE_AA)
    return out

def stretch_width(img, k):
    obj = crop_object(img); h, w = obj.shape
    return paste_center(cv2.resize(obj, (max(1, round(w * k)), h)))

def stretch_top(img, k):
    obj = crop_object(img); h, w = obj.shape
    top, bottom = obj[:h // 2], obj[h // 2:]
    return paste_center(np.vstack([cv2.resize(top, (w, max(1, round(top.shape[0] * k)))), bottom]))

# схематичний будинок (160 px по найбільшій стороні)
b = np.full((CANVAS, CANVAS), 255, np.uint8)
cv2.rectangle(b, (68, 118), (188, 190), 0, 2, cv2.LINE_AA)           # стіни
cv2.line(b, (60, 118), (128, 62), 0, 2, cv2.LINE_AA)                 # дах
cv2.line(b, (128, 62), (196, 118), 0, 2, cv2.LINE_AA)
cv2.rectangle(b, (150, 66), (164, 90), 0, 2, cv2.LINE_AA)            # димар
cv2.rectangle(b, (116, 146), (140, 190), 0, 2, cv2.LINE_AA)          # двері
cv2.rectangle(b, (80, 136), (100, 156), 0, 2, cv2.LINE_AA)           # вікно
cv2.line(b, (90, 136), (90, 156), 0, 1, cv2.LINE_AA); cv2.line(b, (80, 146), (100, 146), 0, 1, cv2.LINE_AA)
base = paste_center(crop_object(b))

panels = [base, shift(base, 16), scale(base, 0.85), add_noise_lines(base, 10),
          stretch_width(base, 0.8), stretch_top(base, 0.8)]
labels = ["а", "б", "в", "г", "д", "е"]

pad = 46
sheet = Image.new("L", (CANVAS * 6, CANVAS + pad), 255)
d = ImageDraw.Draw(sheet)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 34)
for i, (p, lab) in enumerate(zip(panels, labels)):
    sheet.paste(Image.fromarray(p), (i * CANVAS, 0))
    d.rectangle([i * CANVAS, 0, (i + 1) * CANVAS - 1, CANVAS - 1], outline=200)
    d.text((i * CANVAS + CANVAS // 2 - 8, CANVAS + 4), lab + ")", fill=0, font=font)
sheet.save("figure_variants.png")
print("Збережено figure_variants.png")
