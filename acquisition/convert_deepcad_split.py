import json
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset_loader import load_dataset_fast

deepcad_json_path = "train_val_test_split.json"

dataset = load_dataset_fast(
    deepcad_json_path,
    root_dir="/cache/yanko/dataset/abc_preprocessed/organized_by_face_count/",
)
print(dataset.keys())

dataset["val"] = dataset.pop("validation")

with open("deepcad.json", "w") as f:
    json.dump(dataset, f, indent=4)
