import os, glob
import numpy as np
import cv2
import pandas as pd
from skimage.metrics import structural_similarity

# ================== НАЛАШТУВАННЯ ==================
SKETCH_DIR = "sketches"                 # папка з ескізами: sketches/<категорія>/1.png, 2.png ...
CATEGORIES = ["home", "cat", "cup", "tree"]
PER_CATEGORY = 6                        # скільки ескізів брати з кожної категорії
CANVAS, OBJ = 256, 160                  # розмір полотна і максимальний розмір об'єкта (px)
SEED = 42
USE_VIT = True                          # False - порахувати лише класичні методи
VIT_MODEL = "google/vit-base-patch16-224-in21k"
# ==================================================

rng = np.random.default_rng(SEED)

# ---------- 1. Підготовка ескізу ----------
def load_gray(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    if g.mean() < 128:                  # раптом чорний фон
        g = 255 - g
    return g

def paste_center(obj):
    canvas = np.full((CANVAS, CANVAS), 255, np.uint8)
    h, w = obj.shape
    y0, x0 = (CANVAS - h) // 2, (CANVAS - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = obj
    return canvas

def crop_object(img, thr=200):
    ys, xs = np.where(img < thr)
    if len(ys) == 0:
        return None
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

INK_THR = 110   # піксель темніший за це значення вважається штрихом (0 = чорний, 255 = білий)

def normalize_sketch(g):
    ink = (g < INK_THR).astype(np.uint8)                 # 1 = штрих, 0 = фон (будь-якого кольору)
    ink = cv2.dilate(ink, np.ones((3, 3), np.uint8))      # трохи потовщуємо штрихи
    ys, xs = np.where(ink > 0)
    if len(ys) < 50:
        return None
    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = ink.shape
    s = OBJ / max(h, w)
    obj = cv2.resize((255 - ink * 255).astype(np.uint8),
                     (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    obj = np.clip(255 - (255 - obj.astype(np.float32)) * 3, 0, 255).astype(np.uint8)
    return paste_center(obj)                              # чорні штрихи на білому тлі

# ---------- 2. Перетворення ----------
def shift(img, d):
    M = np.float32([[1, 0, d], [0, 1, d]])
    return cv2.warpAffine(img, M, (CANVAS, CANVAS), borderValue=255)

def scale(img, s):
    M = cv2.getRotationMatrix2D((CANVAS / 2, CANVAS / 2), 0, s)
    return cv2.warpAffine(img, M, (CANVAS, CANVAS), borderValue=255)

def add_noise_lines(img, n):
    out = img.copy()
    for _ in range(n):
        x1, y1 = rng.integers(20, CANVAS - 20, 2)
        ang = rng.uniform(0, 2 * np.pi)
        L = rng.integers(20, 60)
        x2 = int(np.clip(x1 + L * np.cos(ang), 0, CANVAS - 1))
        y2 = int(np.clip(y1 + L * np.sin(ang), 0, CANVAS - 1))
        cv2.line(out, (int(x1), int(y1)), (x2, y2), 170, 1, cv2.LINE_AA)
    return out

def stretch_width(img, k):                 # зміна співвідношення сторін
    obj = crop_object(img)
    h, w = obj.shape
    return paste_center(cv2.resize(obj, (max(1, round(w * k)), h)))

def stretch_top(img, k):                   # зміна висоти верхньої половини
    obj = crop_object(img)
    h, w = obj.shape
    top, bottom = obj[:h // 2], obj[h // 2:]
    top2 = cv2.resize(top, (w, max(1, round(top.shape[0] * k))))
    return paste_center(np.vstack([top2, bottom]))

def make_variants(base):
    v = []   # (група, тип, рівень, зображення)
    for d in (8, 16, 24):
        v.append(("nuisance", "shift", d, shift(base, d)))
    for s in (0.85, 0.90, 1.10, 1.15):
        v.append(("nuisance", "scale", s, scale(base, s)))
    for n in (5, 10, 20):
        v.append(("nuisance", "noise", n, add_noise_lines(base, n)))
    for k in (0.80, 0.90, 1.10, 1.20):
        v.append(("deformation", "aspect", k, stretch_width(base, k)))
    for k in (0.80, 0.90, 1.10, 1.20):
        v.append(("deformation", "top_part", k, stretch_top(base, k)))
    return v

# ---------- 3. Метрики (більше = більш схоже) ----------
def m_mse(a, b):
    d = (a.astype(np.float32) - b.astype(np.float32)) / 255.0
    return 1.0 - float(np.mean(d ** 2))

def m_ssim(a, b):
    return float(structural_similarity(a, b, data_range=255))

def canny_edges(a):
    return cv2.Canny(a, 100, 200) > 0

def sobel_edges(a):
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    return mag > 0.25 * mag.max() if mag.max() > 0 else mag > 0

def dice(e1, e2):
    s = e1.sum() + e2.sum()
    return 1.0 if s == 0 else float(2 * np.logical_and(e1, e2).sum() / s)

if USE_VIT:
    import torch
    from transformers import ViTImageProcessor, ViTModel
    proc = ViTImageProcessor.from_pretrained(VIT_MODEL)
    vit = ViTModel.from_pretrained(VIT_MODEL).eval()

    @torch.no_grad()
    def vit_feats(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        inp = proc(images=rgb, return_tensors="pt")
        h = vit(**inp).last_hidden_state[0]
        return h[0].numpy(), h[1:].reshape(-1).numpy()      # CLS, усі патчі

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

# ---------- 4. Основний цикл ----------
bases = []
for cat in CATEGORIES:
    files = sorted(glob.glob(os.path.join(SKETCH_DIR, cat, "*.png")))
    n = 0
    for f in files:
        g = load_gray(f)
        b = normalize_sketch(g) if g is not None else None
        if b is not None:
            bases.append((cat, os.path.basename(f), b))
            n += 1
        if n == PER_CATEGORY:
            break
    print(cat, "->", n, "ескізів")
cv2.imwrite("bases_check.png", np.vstack([np.hstack([b[2] for b in bases[i:i + 6]] + [np.full((CANVAS, CANVAS), 255, np.uint8)] * (6 - len(bases[i:i + 6]))) for i in range(0, len(bases), 6)]))
assert bases, "Не знайдено ескізів. Перевірте SKETCH_DIR і назви категорій."

rows = []
for bid, (cat, name, base) in enumerate(bases):
    ref_c, ref_s = canny_edges(base), sobel_edges(base)
    if USE_VIT:
        ref_cls, ref_patch = vit_feats(base)
    for group, kind, level, img in make_variants(base):
        r = dict(base_id=bid, category=cat, file=name, group=group, kind=kind, level=level)
        r["MSE"] = m_mse(base, img)
        r["SSIM"] = m_ssim(base, img)
        r["Canny"] = dice(ref_c, canny_edges(img))
        r["Sobel"] = dice(ref_s, sobel_edges(img))
        if USE_VIT:
            c, p = vit_feats(img)
            r["ViT_CLS"] = cos(ref_cls, c)
            r["ViT_patch"] = cos(ref_patch, p)
        rows.append(r)
    print("оброблено", bid + 1, "/", len(bases))

df = pd.DataFrame(rows)
df.to_csv("results.csv", index=False)
metrics = [m for m in ["MSE", "SSIM", "Canny", "Sobel", "ViT_CLS", "ViT_patch"] if m in df.columns]

# ---------- 5. Результати ----------
order = ["shift", "scale", "noise", "aspect", "top_part"]
table = df.groupby("kind")[metrics].mean().loc[order].round(3)
print("\nТаблиця A. Середня подібність до еталона (1 = ідентично)")
print(table)
table.to_csv("table_mean.csv")

def pair_acc(g, col):
    n = g[g.group == "nuisance"][col].values
    d = g[g.group == "deformation"][col].values
    return np.mean([(x > y) + 0.5 * (x == y) for x in n for y in d])

auc = {m: np.mean([pair_acc(g, m) for _, g in df.groupby("base_id")]) for m in metrics}
auc = pd.Series(auc).round(3)
print("\nТаблиця B. Частка пар, де зміна без впливу на пропорції оцінена як БІЛЬШ схожа, ніж зміна пропорцій")
print("(1.0 - ідеально, 0.5 - випадково)")
print(auc)
auc.to_csv("table_auc.csv")

# ---------- 6. Приклад для ілюстрації ----------
b = bases[0][2]
vs = make_variants(b)
pick = [b] + [vs[i][3] for i in (1, 4, 8, 10, 14)]   # еталон, зсув, масштаб, шум, ширина, верх
cv2.imwrite("examples.png", np.hstack(pick))
print("\nЗбережено: results.csv, table_mean.csv, table_auc.csv, examples.png")
