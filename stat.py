# 设置你的文件夹路径，'.' 表示当前文件夹
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置你的主文件夹路径，'.' 表示当前文件夹下的所有目录
folder_path = "/cache/yanko/dataset/abc-splited-bezier-all/organized"

# 存储提取的数据
data = []
print("Scanning folders, please wait...")

# 使用 os.walk 进行递归查找
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(".npz"):
            name_without_ext = filename[:-4]
            parts = name_without_ext.split("_")

            if len(parts) >= 4:
                try:
                    faces = int(parts[-2])
                    edges = int(parts[-1])
                    full_path = os.path.join(root, filename)
                    data.append(
                        {
                            "filepath": full_path,
                            "filename": filename,
                            "faces": faces,
                            "edges": edges,
                        }
                    )
                except ValueError:
                    pass

# 转换为 pandas DataFrame
df = pd.DataFrame(data)

if df.empty:
    print("No valid .npz files found.")
else:
    # --- 1. 打印基础统计信息 ---
    print(f"\nScan complete! Found {len(df)} valid files.")
    print("\n========== Dataset Statistics ==========")
    print(df[["faces", "edges"]].describe())
    print("========================================")

    # --- 2. 数据可视化 (纯英文，避免字体报错) ---
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 图 1: 面数的直方图
    sns.histplot(df["faces"], bins=50, kde=True, ax=axes[0, 0], color="skyblue")
    axes[0, 0].set_title("Distribution of Faces")
    axes[0, 0].set_xlabel("Number of Faces")
    axes[0, 0].set_ylabel("Frequency")

    # 图 2: 边数的直方图
    sns.histplot(df["edges"], bins=50, kde=True, ax=axes[0, 1], color="salmon")
    axes[0, 1].set_title("Distribution of Edges")
    axes[0, 1].set_xlabel("Number of Edges")
    axes[0, 1].set_ylabel("Frequency")

    # 图 3: 面数 vs 边数的散点图
    sns.scatterplot(data=df, x="faces", y="edges", ax=axes[1, 0], alpha=0.5)
    axes[1, 0].set_title("Relationship: Faces vs Edges")
    axes[1, 0].set_xlabel("Number of Faces")
    axes[1, 0].set_ylabel("Number of Edges")

    # 图 4: 箱线图 (查看异常值)
    sns.boxplot(data=df[["faces", "edges"]], ax=axes[1, 1])
    axes[1, 1].set_title("Boxplot of Faces and Edges (Outlier Detection)")

    plt.tight_layout()

    # 【重点修改】如果你在服务器上，弹不出窗口，就把图保存成文件
    save_path = "dataset_distribution.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n可视化图表已保存至: {save_path}，请将该图片下载到本地查看。")

    # 如果你在带图形界面的系统，依然可以展示
    # plt.show()
