import json
import os
import sys
import pandas as pd

if len(sys.argv) < 2:
    print("Please provide the folder path as a command-line argument.")
    sys.exit(1)

folder = sys.argv[1]
res = {}


for f_name in os.listdir(folder):
    if not os.path.isdir(os.path.join(folder, f_name)):
        continue
    for model_name in  os.listdir(os.path.join(folder, f_name)):
        if not os.path.isdir(os.path.join(folder, f_name, model_name)):
            continue
        print(f_name, model_name)
        res[f_name+'/'+model_name] = {}
        r = res[f_name+'/'+model_name]
        fld = os.path.join(folder, f_name, model_name)
        for filename in os.listdir(fld):
            if filename.endswith('.json'):
                if filename == "model_meta.json":
                    continue
                file_path = os.path.join(fld, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    r[filename[:-4]] = data['metrics']['NDCG']['NDCG@10']

# You can print or use the `res` dictionary as needed.
print(res)

df = pd.DataFrame(res)
df.to_csv(os.path.join(folder,'results.csv'))