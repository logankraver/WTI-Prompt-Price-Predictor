import pandas as pd

with open('results.txt', 'r') as f:
    lines = f.readlines()

flatten = ""
for line in lines:
    flatten += line
flatten = flatten.strip().replace("\n", "")

start = 0
results = []
params = ["hid_dim", "final_dim", "num_layers", "num_epochs", "lr", "test_loss"]
while start < len(flatten) and flatten.find(":", start) != -1:
    result = {}
    for param in params:
        param_start, param_end = flatten.find(":", start), flatten.find(",", start)
        val = float(flatten[param_start+1:param_end])
        start = param_end + 1
        result[param] = val

    preds_start, preds_end = flatten.find("[", start), flatten.find("]", start)
    result["preds"] = flatten[preds_start:preds_end+1]
    start = flatten.find("{", preds_end)

    results.append(result)

df = pd.DataFrame(results)
df.to_csv("results.csv")
