import matplotlib.pyplot as plot
import numpy as np 
import pandas as pd 
import scikit_posthocs as sp
from pathlib import Path


#mouse_no = input("Input mouse number: ")
mouse_no = 9
data_path = Path(f".\\data\\mouse_{mouse_no}")

cluster_details = pd.read_csv(data_path / "Cluster_detail_results.csv")
idx3D = cluster_details["Folder_Name"].str.contains("arenaH", case=False) | cluster_details["Folder_Name"].str.contains("arenaL", case=False) | cluster_details["Folder_Name"].str.contains("arenaM", case=False)

cluster_details["Folder_Name"].loc[idx3D] = "3D"
cluster_details["Folder_Name"].loc[~idx3D] = "2D"

ratio = np.ones((np.max(cluster_details["ClusterIdx"]),)) * 2

for i in range(1,np.max(cluster_details["ClusterIdx"])+1):
    idx = cluster_details['ClusterIdx'] == i
    num_3D = np.sum((cluster_details["Folder_Name"].loc[idx] == "3D"))
    num_2D = np.sum((cluster_details["Folder_Name"].loc[idx] == "2D"))
    
    ratio[i-1] = num_2D / (num_3D + num_2D)
ratio = np.log(ratio / (1.0000000001 - ratio))
#ratio = ratio[:-4]
print(ratio)

"""
gesd_analysis = sp.outliers_gesd(ratio, 10, report=True, hypo=True)
print(ratio[gesd_analysis])
"""
#plot.xkcd()
plot.hist(ratio)
plot.show()


q1 = np.percentile(ratio, 25)
q3 = np.percentile(ratio, 75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = ratio[(ratio < lower) | (ratio > upper)]
print(np.exp(outliers) / (1 + np.exp(outliers)))

x=ratio
med = np.median(x)
mad = np.median(np.abs(x - med))
z = 0.6745 * (x - med) / mad
outliers = np.abs(z) > 3.5
print(np.exp(ratio[outliers]) / (1 + np.exp(ratio[outliers])))
