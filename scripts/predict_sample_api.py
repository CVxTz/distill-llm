from concurrent.futures import ThreadPoolExecutor
from string import Template

import pandas as pd
import tqdm
from datasets import load_dataset
from openai import OpenAI

from distill.config import (
    API_KEY,
    BASE_MODEL,
    BASE_PATH,
    BASE_URL,
)
from distill.llm import call_llm
from distill.utils import exception_handler

BASE_CLIENT = OpenAI(base_url=BASE_URL, api_key=API_KEY)

with open(BASE_PATH / "data" / "gec.template", "r") as f:
    TEMPLATE = Template(f.read().strip())


def process_call(prompt):
    with exception_handler():
        return call_llm(
            client=BASE_CLIENT,
            prompt=prompt,
            model_name=BASE_MODEL,
        ).strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    split = args.split

    data_path = BASE_PATH / "data"

    out_name = "api_predicted_test" if split == "test" else "api_predicted_distill"
    size = 1000 if split == "test" else 5000

    out_path = data_path / f"{out_name}.csv"

    data = pd.DataFrame(
        load_dataset("juancavallotti/multilingual-gec", split=split)
    ).head(size)

    data["text"] = data["modified"].str.replace("fix grammar: ", "")

    data["prompt"] = data.apply(lambda x: TEMPLATE.substitute(text=x["text"]), axis=1)

    print(data["prompt"].values[0])

    with ThreadPoolExecutor(max_workers=20) as executor:
        data["answer"] = list(
            tqdm.tqdm(
                executor.map(process_call, data.prompt.values),
                total=data.shape[0],
            )
        )

    data.to_csv(out_path, index=False)
