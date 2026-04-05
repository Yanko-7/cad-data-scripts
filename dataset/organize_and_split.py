import glob
import json
import multiprocessing
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

# ================= Global Configuration =================

# 1. 输入：源文件夹列表
SOURCE_DIRS = [
    r"/cache/yanko/dataset/abc_preprocessed/",
]

# 2. 输出：整理后存放的根目录
OUTPUT_BASE_DIR = r"/cache/yanko/dataset/abc_preprocessed/organized_by_face_count"

# 3. 数据集切分配置
SPLIT_JSON_NAME = "dataset_split.json"
# 仅将以下前缀的文件夹纳入训练集 (例如排除  06 等高面数模型)
TRAIN_TARGET_PREFIXES = ("01_", "02_", "03_", "04_", "05_")
SPLIT_RATIOS = (0.90, 0.05, 0.05)  # Train / Val / Test

# 4. 通用配置
FILE_EXTENSION = "*.npz"  # 统一文件后缀
ACTION_MODE = "move"  # 'move' or 'copy'
MAX_WORKERS = max(1, os.cpu_count() - 2)
RANDOM_SEED = 42  # 固定随机种子，保证切分结果可复现

# =======================================================


def get_category_folder(face_count):
    if face_count <= 10:
        return "01_Faces_0_to_10"
    elif face_count <= 30:
        return "02_Faces_10_to_30"
    elif face_count <= 50:
        return "03_Faces_30_to_50"
    elif face_count <= 70:
        return "04_Faces_50_to_70"
    elif face_count <= 90:
        return "05_Faces_70_to_90"
    else:
        return "06_Faces_90_plus"


def parse_face_count_from_name(filename_stem):
    """从文件名解析面数: ..._step_000_{face}_{edge}"""
    try:
        parts = filename_stem.split("_")
        return int(parts[-2])
    except (IndexError, ValueError):
        return None


def process_one_file(file_path_str):
    """Worker: 处理单个文件的移动/复制"""
    try:
        path_obj = Path(file_path_str)
        face_num = parse_face_count_from_name(path_obj.stem)

        if face_num is None:
            sub_folder = "99_Unparsed_Filename"
        else:
            sub_folder = get_category_folder(face_num)

        target_dir = os.path.join(OUTPUT_BASE_DIR, sub_folder)
        os.makedirs(target_dir, exist_ok=True)

        target_path = os.path.join(target_dir, path_obj.name)

        if ACTION_MODE == "move":
            shutil.move(file_path_str, target_path)
        else:
            shutil.copy2(file_path_str, target_path)

        return True, "Success"
    except Exception as e:
        return False, f"Error: {str(e)}"


def run_organization():
    """阶段一：文件整理"""
    print(f"\n[Phase 1] Organizing Files ({ACTION_MODE})...")

    if not os.path.exists(OUTPUT_BASE_DIR):
        os.makedirs(OUTPUT_BASE_DIR)

    all_files = []
    for source_dir in SOURCE_DIRS:
        pattern = os.path.join(source_dir, "**", FILE_EXTENSION)
        all_files.extend(glob.glob(pattern, recursive=True))

    total_files = len(all_files)
    print(f"Found {total_files} files to organize.")

    if total_files == 0:
        return False

    success_count = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(
            tqdm(
                executor.map(process_one_file, all_files),
                total=total_files,
                unit="file",
            )
        )

    for success, _ in results:
        if success:
            success_count += 1

    print(f"Organization Complete. Success: {success_count}/{total_files}")
    return True


def run_splitting():
    """阶段二：生成数据集 Split JSON"""
    print("\n[Phase 2] Generating Dataset Split JSON...")

    valid_files = []

    # 扫描 Output 目录
    # 注意：这里扫描的是整理后的目录，确保文件真实存在
    if not os.path.exists(OUTPUT_BASE_DIR):
        print("Error: Output directory does not exist.")
        return

    subfolders = [
        f
        for f in os.listdir(OUTPUT_BASE_DIR)
        if os.path.isdir(os.path.join(OUTPUT_BASE_DIR, f))
    ]

    print(f"Scanning target folders: {TRAIN_TARGET_PREFIXES}")

    for folder_name in subfolders:
        # 过滤文件夹前缀
        if folder_name.startswith(TRAIN_TARGET_PREFIXES):
            folder_path = os.path.join(OUTPUT_BASE_DIR, folder_name)
            # 搜索整理后的文件
            files = glob.glob(os.path.join(folder_path, FILE_EXTENSION))
            valid_files.extend(files)
            print(f"  -> Included {folder_name}: {len(files)} files")
        else:
            print(f"  -> Skipped  {folder_name} (Not in target prefixes)")

    total = len(valid_files)
    if total == 0:
        print("No valid files found for splitting.")
        return

    # 打乱
    random.seed(RANDOM_SEED)
    random.shuffle(valid_files)

    # 计算索引
    r_train, r_val, r_test = SPLIT_RATIOS
    n_train = int(total * r_train)
    n_val = int(total * r_val)
    # 剩余归 test

    # 生成相对路径列表
    def to_rel(path_list):
        return [
            os.path.relpath(p, OUTPUT_BASE_DIR).replace("\\", "/") for p in path_list
        ]

    split_data = {
        "meta": {
            "total": total,
            "prefixes_included": TRAIN_TARGET_PREFIXES,
            "ratios": SPLIT_RATIOS,
        },
        "train": to_rel(valid_files[:n_train]),
        "val": to_rel(valid_files[n_train : n_train + n_val]),
        "test": to_rel(valid_files[n_train + n_val :]),
    }

    output_json_path = os.path.join(OUTPUT_BASE_DIR, SPLIT_JSON_NAME)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=4)

    print(f"\nSaved split configuration to: {output_json_path}")
    print(f"Total Files: {total}")
    print(f"Train: {len(split_data['train'])}")
    print(f"Val:   {len(split_data['val'])}")
    print(f"Test:  {len(split_data['test'])}")


def main():
    # 1. 执行整理
    org_success = run_organization()

    # 2. 如果整理（或扫描）成功，执行切分
    # 即使 org_success 为 False (比如第一次运行没文件，或者已经整理过了想重新生成JSON)，
    # 我们也可以加一个判断，或者直接允许运行 Phase 2
    if org_success or os.path.exists(OUTPUT_BASE_DIR):
        run_splitting()
    else:
        print("Skipping split generation due to lack of data.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
