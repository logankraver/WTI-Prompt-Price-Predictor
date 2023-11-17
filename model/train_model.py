from FFNN import FFNN, format_dataloader, train_model, eval_model
import torch.optim as optim
import torch.nn as nn

train = format_dataloader("../data/scripts/train.csv")
val = format_dataloader("../data/scripts/val.csv")
test = format_dataloader("../data/scripts/test.csv")

input_dim = 10 
hid_dim = 100
final_dim = 30
num_layers = 5
num_epochs = 20
lr = 0.0000005


model = FFNN(input_dim, hid_dim, final_dim, num_layers)
optimizer = optim.SGD(model.parameters(), lr=lr)
loss_fn = nn.L1Loss()
train_model(model, train, val, num_epochs, optimizer, loss_fn)
res, test_loss = eval_model(model, test, loss_fn)
print(res)
