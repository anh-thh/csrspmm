import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import seaborn as sns

subprocess.run(["bash", "csrspmm_vs_cusparse.sh"], check=True)

df = pd.read_csv("csrspmm_vs_cusparse.csv")

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
        marker="o"
    )
    plt.title(f"SpMM Performance for Size: {case}")
    plt.xlabel("Density")
    plt.ylabel("GFLOPS")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Algorithm")
    plt.tight_layout()

    plt.savefig(f"csrspmm_{case}.png", dpi=300)

    plt.show()
    plt.close()


