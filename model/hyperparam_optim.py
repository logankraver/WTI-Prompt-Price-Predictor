from FFNN import FFNN, format_dataloader, train_model, eval_model
import torch.optim as optim
import torch.nn as nn
import pandas as pd

train = format_dataloader("../data/scripts/train.csv")
val = format_dataloader("../data/scripts/val.csv")
test = format_dataloader("../data/scripts/test.csv")

input_dim = 10 

hyperparams = {
    "hid_dim": [50, 75, 100, 125, 150],
    "final_dim": [10, 20, 30],
    "num_layers": [2,3,4,5,6],
    "num_epochs": [20,30,40,50,60],
    "lr": [0.0000005, 0.000001, 0.000005, 0.00001, 0.00005]
}

hyperparams = {
    "hid_dim": [50, 75],
    "final_dim": [20, 30],
    "num_layers": [4,5,6],
    "num_epochs": [30,40,50],
    "lr": [0.000001, 0.000005, 0.00001]
}
total_iter = 30
loss_fn = nn.L1Loss()

results = []
for hid_dim in hyperparams["hid_dim"]:
    for final_dim in hyperparams["final_dim"]:
        for num_layers in hyperparams["num_layers"]:
            for num_epochs in hyperparams["num_epochs"]:
                for lr in hyperparams["lr"]:
                    test_loss = 0
                    preds = [0. for _ in range(10)]
                    for _ in range(total_iter):
                        model = FFNN(input_dim, hid_dim, final_dim, num_layers)
                        optimizer = optim.SGD(model.parameters(), lr=lr)
                        train_model(model, train, val, num_epochs, optimizer, loss_fn)
                        res, loss = eval_model(model, test, loss_fn)
                        test_loss += loss
                        for i in range(10):
                            preds[i] += res[i]

                    for i in range(10):
                        preds[i] = preds[i] / total_iter
                    results.append({
                        "hid_dim": hid_dim,
                        "final_dim": final_dim,
                        "num_layers": num_layers,
                        "num_epochs": num_epochs,
                        "lr": lr,
                        "test_loss": test_loss / total_iter,
                        "preds": preds
                    })

print(results)
df = pd.DataFrame(results)
df.to_csv("hyperparam_results.csv")
