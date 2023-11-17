import pandas as pd

df = pd.read_csv("hyperparam_results.csv")
best_models = df[df['test_loss'] < 4]
hid_dim, final_dim, num_layers = {}, {}, {}
layer_agg = {}
num_epochs, lr = {}, {}
for _, row in best_models.iterrows():
    hid = row.get("hid_dim")
    final = row.get("final_dim")
    n_layers = row.get("num_layers")
    n_epochs = row.get("num_epochs")
    learn = row.get("lr")

    hid_dim[hid] = 1 + hid_dim.get(hid, 0)
    final_dim[final] = 1 + final_dim.get(final, 0)
    num_layers[n_layers] = 1 + num_layers.get(n_layers, 0)

    layer_agg[(hid, final, n_layers)] = 1 + layer_agg.get((hid, final, n_layers), 0)

    num_epochs[n_epochs] = 1 + num_epochs.get(n_epochs, 0)
    lr[learn] = 1 + lr.get(learn, 0)

print(hid_dim)
print(final_dim)
print(num_layers)

print(layer_agg)

print(num_epochs)
print(lr)
