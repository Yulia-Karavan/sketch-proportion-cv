"""Відтворення таблиць 1-2 і довірчих інтервалів з results.csv (результат experiment.py)."""
import sys
import numpy as np
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
df = pd.read_csv(path)
M = ["MSE", "SSIM", "Canny", "Sobel", "ViT_CLS", "ViT_patch"]
order = ["shift", "scale", "noise", "aspect", "top_part"]

def pair_acc(g, col, kind=None):
    n = g[g.group == "nuisance"] if kind is None else g[g.kind == kind]
    d = g[g.group == "deformation"][col].values
    n = n[col].values
    return np.mean([(x > y) + 0.5 * (x == y) for x in n for y in d])

t1 = df.groupby("kind")[M].mean().loc[order].round(3)
print("Таблиця 1. Середня подібність до еталона\n", t1, "\n")

t2 = pd.DataFrame({m: {**{k: np.mean([pair_acc(g, m, k) for _, g in df.groupby("base_id")])
                          for k in ["shift", "scale", "noise"]},
                       "all": np.mean([pair_acc(g, m) for _, g in df.groupby("base_id")])}
                   for m in M}).T.round(3)
print("Таблиця 2. Частка правильно впорядкованих пар\n", t2, "\n")

rng = np.random.default_rng(0)
print("95 % бутстреп-ДІ для частки по всіх змінах (5000 повторень, ресемплінг ескізів)")
for m in M:
    per = np.array([pair_acc(g, m) for _, g in df.groupby("base_id")])
    bs = [rng.choice(per, len(per)).mean() for _ in range(5000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"{m:10s} {per.mean():.3f}  [{lo:.3f}; {hi:.3f}]")

t1.to_csv("table1_mean.csv"); t2.to_csv("table2_pairs.csv")
