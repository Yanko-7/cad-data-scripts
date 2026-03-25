import json

with open("abc_data_split_6bit.json") as f:
    data = json.load(f)
    for key, value in data.items():
        print(len(value))
