import json
from pathlib import Path

#
# with open("filtered_brep_abc_data_split_6bit_paths.json", "rb") as f:
#     data = json.load(f)
#     pickle.dump(data, open("filtered_brep_abc_data_split_6bit_paths.pkl", "wb"))

SEARCH_ROOT = "/cache/yanko/dataset/abc-splited-bezier-all/organized"
with open("train_val_test_split.json", "r") as f:
    split = json.load(f)
    dataset = {}
    new_dataset = {"train": [], "validation": [], "test": []}
    for key, value in split.items():
        dataset[key] = [str(x).split("/")[-1] for x in value]
        print(len(dataset[key]))
    search_path = Path(SEARCH_ROOT)
    with open("train_val_test_split_new.json", "w") as f:
        json.dump(dataset, f, indent=4)
for file in search_path.rglob("*.npz"):
    name = file.stem
    if name.split("_")[0] in dataset["train"]:
        new_dataset["train"].append(name)
    elif name.split("_")[0] in dataset["validation"]:
        new_dataset["validation"].append(name)
    elif name.split("_")[0] in dataset["test"]:
        new_dataset["test"].append(name)
with open("train_val_test_split_new.json", "w") as f:
    for key, value in new_dataset.items():
        print(key, len(value))
    json.dump(new_dataset, f, indent=4)
