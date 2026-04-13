import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_1samp
from pathlib import Path

def analyze_cluster_types(path, df):
    is_3d = df["Folder_Name"].str.contains("arenaH|arenaL|arenaM", na=False)
    df["Type"] = is_3d.map({True: "3D", False: "2D"})

    df = df.drop(columns=["Timestamp", "Folder_Name"])
    groups = df.groupby("ClusterIdx")

    type_ratios = (
        groups["Type"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .rename(columns={"2D": "ratio_2d", "3D": "ratio_3d"})
    )

    print(type_ratios)

    null_mean = 2 / 5
    ratio_2d = type_ratios["ratio_2d"]
    t_statistic, p_value = ttest_1samp(ratio_2d, popmean=null_mean)
    contingency_table = pd.crosstab(df["ClusterIdx"], df["Type"])
    chi2_statistic, chi2_p_value, chi2_dof, _ = chi2_contingency(contingency_table)

    output = [
        f"Null mean: {null_mean:.3f}",
        f"Sample mean ratio_2d: {ratio_2d.mean():.6f}",
        f"One-sample t statistic: {t_statistic:.6f}",
        f"p-value: {p_value:.6g}",
        f"Degrees of freedom: {len(ratio_2d) - 1}",
        f"Chi-square statistic: {chi2_statistic:.6f}",
        f"Chi-square p-value: {chi2_p_value:.6g}",
        f"Chi-square degrees of freedom: {chi2_dof}",
    ]

    for line in output:
        print(line)
    print("\n" + "=" * 50 + "\n")

    with open(path / "statistical_analysis.txt", "w") as f:
        f.write("\n".join(output))

    sns.set_theme(style="whitegrid")
    ax = sns.histplot(type_ratios["ratio_2d"], bins=20)
    ax.set(
        title="Distribution of 2D Ratios by Cluster",
        xlabel="Ratio within cluster",
        ylabel="Count of clusters",
    )

    plt.tight_layout()
    plt.savefig(path / f"ratio_histogram_{path.name}.png", dpi=200)
    plt.close()

    return type_ratios

# Main analysis loop

folder_path = Path("./data/")
data_paths = [d for d in folder_path.iterdir() if d.is_dir()]
df_collection = []
per_folder_results = {}
offset = 0

for path in data_paths:
    df = pd.read_csv(path / "Cluster_detail_results.csv")
    df["ClusterIdx"] += offset
    offset = df["ClusterIdx"].max() + 1
    df_collection.append(df)
    per_folder_results[path.name] = analyze_cluster_types(path, df)

fig, ax = plt.subplots(figsize=(10, 6))
for folder_name, type_ratios in per_folder_results.items():
    sns.ecdfplot(type_ratios["ratio_2d"], label=folder_name, alpha=1)
ax.set(
    title="Fraction of 2D samples within clusters (ECDF)",
    xlabel="Fraction of Data",
    ylabel="Count of clusters",
)
ax.legend(title="Dataset")
plt.tight_layout()
plt.savefig(folder_path / "ratio_ecdf_overlay.png", dpi=200)
plt.close()

combined_df = pd.concat(df_collection, ignore_index=True)
analyze_cluster_types(folder_path, combined_df)