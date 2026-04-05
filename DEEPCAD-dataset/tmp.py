import pickle
import json

with open("brepgen_deepcad_data_split_6bit.pkl", "rb") as g:
    data2 = pickle.load(g)
    result = {}
    for key, value in data2.items():
        result[key] = [v.split("/")[-1].split(".")[0] for v in value]
    with open("brepgen_deepcad_data_split_6bit.json", "w") as f:
        json.dump(result, f, indent=4)