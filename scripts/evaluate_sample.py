import pandas as pd

from distill.config import (
    BASE_PATH,
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--file", type=str, default="lora_predicted_test.csv")
    args = parser.parse_args()
    file_name = args.file

    data_path = BASE_PATH / "data"

    in_path = data_path / file_name

    data = pd.read_csv(in_path)

    # data = data[data.lang == "en"]

    data["answer_cleaned"] = data["answer"].apply(lambda x: x.split("\n")[0].strip())

    score = (data["answer_cleaned"] == data["sentence"]).mean()

    print(f"{file_name=} {score=}")

    # score = (data["text"] == data["sentence"]).mean()

    # print(f"{score=}")

    # for idx, row in data.sample(5).iterrows():
    #     if row["sentence"] != row["answer_cleaned"]:
    #         print(row["sentence"])
    #         print(row["text"])
    #         print(row["answer_cleaned"])
