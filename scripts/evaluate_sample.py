import pandas as pd

from distill.config import (
    BASE_PATH,
)

if __name__ == "__main__":
    data_path = BASE_PATH / "data"

    in_path = data_path / "lora_predicted_test.csv"

    data = pd.read_csv(in_path)

    data = data[data.lang == "en"]

    data["answer_cleaned"] = data["answer"].apply(lambda x: x.split("\n")[0].strip())

    score = (data["answer_cleaned"] == data["sentence"]).mean()

    print(f"{score=}")

    # score = (data["text"] == data["sentence"]).mean()

    # print(f"{score=}")

    for idx, row in data.sample(100).iterrows():
        if row["sentence"] != row["answer_cleaned"]:
            print(row["sentence"])
            print(row["text"])
            print(row["answer_cleaned"])
