import pandas as pd
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
df = pd.read_csv("evaluation_results.csv")
print("COLUMNS:", df.columns.tolist())
pred = [c for c in df.columns if "pred" in c.lower()][0]
true = [c for c in df.columns if any(k in c.lower() for k in ("truth", "gt", "actual", "true"))][0]
df["err"] = df[pred] - df[true]
print("\nDENSE-SCENE CANDIDATES (biggest undercounts):")
print(df.sort_values("err").head(5))
print("\nSPARSE-SCENE CANDIDATES (closest to truth):")
c = df[df[true] > 0].copy()
c["abs_err"] = c["err"].abs()
print(c.sort_values(["abs_err", true]).head(5))
