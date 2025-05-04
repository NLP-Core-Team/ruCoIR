import json
import os
import sys

if len(sys.argv) < 2:
    print("Please provide the folder path as a command-line argument.")
    sys.exit(1)

folder = sys.argv[1]
res = {}

for filename in os.listdir(folder):
    if filename.endswith('.json'):
        if filename == "model_meta.json":
            continue
        file_path = os.path.join(folder, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
            res[filename[:-4]] = data['metrics']['NDCG']['NDCG@10']

# You can print or use the `res` dictionary as needed.
print(res)
