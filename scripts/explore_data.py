import pandas as pd
from datasets import load_dataset

dataset = pd.DataFrame(load_dataset("juancavallotti/multilingual-gec", split="train"))

for idx, row in dataset.head(10).iterrows():
    print(row.to_dict())


preds = pd.read_csv("../data/predicted_validation.csv")


print(preds.lang.value_counts())
