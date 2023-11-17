from FFNN import FFNN, format_dataloader, train_model, pred_model, save_model, load_model
import torch.optim as optim
import torch.nn as nn

model_file = True
proj_file = "proj_vals.csv"
input_dim = 10 
hid_dim = 75
final_dim = 20
num_layers = 5
num_epochs = 30
lr = 0.000001 

# load data
proj = format_dataloader(proj_file)
train = format_dataloader("../data/scripts/train.csv")
val = format_dataloader("../data/scripts/val.csv")
test = format_dataloader("../data/scripts/test.csv")

models = []
loss_fn = nn.L1Loss()
preds = [0. for _ in range(10)]

if not model_file:
    for n in range(30):
        model = FFNN(input_dim, hid_dim, final_dim, num_layers)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        train_model(model, train + val, test, num_epochs, optimizer, loss_fn)

        save_model(model, f"models/oil_ffnn_{n}.pth")

        preds[0] += pred_model(model, proj[1]['input'])
        for i in range(2, len(proj)):
            # insert prev price into 
            proj[i]['input'][-1] = preds[-1]
            preds[i-1] += pred_model(model, proj[i]['input'])
    for i in range(len(preds)):
        preds[i] = preds[i] / 30

else:
    for n in range(30):
        model = FFNN(input_dim, hid_dim, final_dim, num_layers)
        model = load_model(f"models/oil_ffnn_{n}.pth")

        preds[0] += pred_model(model, proj[1]['input'])
        for i in range(2, len(proj)):
            # insert prev price into 
            proj[i]['input'][-1] = preds[-1]
            preds[i-1] += pred_model(model, proj[i]['input'])
    for i in range(len(preds)):
        preds[i] = preds[i] / 30

print(preds)
