import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import seaborn as sns
from pathlib import Path

csv_path = Path("csrspmm_vs_cusparse.csv")
#  subprocess.run(["bash", "csrspmm_vs_cusparse.sh"], check=True)
if not csv_path.exists():
    print("CSV not found. Running csrspmm_vs_cusparse.sh ...")
    subprocess.run(["bash", "csrspmm_vs_cusparse.sh"], check=True)
else:
    print("CSV already exists. Skipping rerun experiment.")

df = pd.read_csv(csv_path)

df["density"] = pd.to_numeric(df["density"])

csrs = df[df["lib"] == "csrspmm"]

#  plt.figure(figsize=(10, 6))
#  sns.lineplot(
#      data=csrs,
#      x="density",
#      y="gflops",
#      hue="algo",
#      marker="o"
#  )
#
#  plt.title("CSRSpMM Performance vs Density")
#  plt.xlabel("Density")
#  plt.ylabel("GFLOPS")
#  plt.grid(True, linestyle="--", alpha=0.5)
#  plt.legend(title="Algorithm")
#
#  plt.tight_layout()
#  plt.show()

# Performance comparision in different mat sizes
for case in df["case"].unique():
    subset = df[df["case"] == case]

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=subset,
        x="density",
        y="gflops",
        hue="algo",
        marker="o",
        linewidth=2,
        alpha=0.8
    )

    # Highlight the baseline: cuSPARSE
    cus = subset[subset["algo"] == "cuSPARSE"]
    if not cus.empty:
        sns.lineplot(
            data=cus,
            x="density",
            y="gflops",
            marker="o",
            linewidth=4,
            color="black",
            label="cuSPARSE"
        )

    plt.title(f"SpMM Performance for Size: {case}")
    plt.xlabel("Density")
    plt.ylabel("GFLOPS")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), title="Algorithm")

    plt.savefig(f"csrspmm_{case}.png", dpi=300)
    plt.show()
    plt.close()


print("\n========= Summary =========")

mean_gflops = df.groupby("algo")["gflops"].mean().rename("mean_gflops")

cus = df[df["algo"] == "cuSPARSE"][["case", "density", "gflops"]]
cus = cus.rename(columns={"gflops": "cusparse_gflops"})

merged = df.merge(cus, on=["case", "density"], how="left")
merged["speedup"] = merged["gflops"] / merged["cusparse_gflops"]

mean_speedup = merged.groupby("algo")["speedup"].mean().rename("mean_speedup")

mean_speedup["cuSPARSE"] = 1.0

summary = pd.concat([mean_gflops, mean_speedup], axis=1)
summary = summary.sort_values("mean_gflops", ascending=False)

print(summary.to_string())
