import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_1samp


df = pd.read_csv("./data/mouse_2/Cluster_detail_results.csv")

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

print(f"Null mean: {null_mean:.3f}")
print(f"Sample mean ratio_2d: {ratio_2d.mean():.6f}")
print(f"One-sample t statistic: {t_statistic:.6f}")
print(f"p-value: {p_value:.6g}")
print(f"Degrees of freedom: {len(ratio_2d) - 1}")
print(f"Chi-square statistic: {chi2_statistic:.6f}")
print(f"Chi-square p-value: {chi2_p_value:.6g}")
print(f"Chi-square degrees of freedom: {chi2_dof}")

sns.set_theme(style="whitegrid")
ax = sns.histplot(type_ratios["ratio_2d"], bins=20)
ax.set(
    title="Distribution of 2D Ratios by Cluster",
    xlabel="Ratio within cluster",
    ylabel="Count of clusters",
)

plt.tight_layout()
plt.savefig("ratio_histogram.png", dpi=200)
