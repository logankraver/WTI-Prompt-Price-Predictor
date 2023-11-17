import pandas as pd

val_size, test_size = 20, 10
df = pd.read_csv("dataset.csv")
test = df.head(test_size)
val = df.iloc[test_size: test_size + val_size]
train = df.iloc[test_size+val_size:]
test.to_csv("test.csv")
val.to_csv("val.csv")
train.to_csv("train.csv")

