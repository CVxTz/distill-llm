from string import Template

import pandas as pd
import torch
import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from distill.config import (
    BASE_PATH,
)

with open(BASE_PATH / "data" / "gec.template", "r") as f:
    TEMPLATE = Template(f.read().strip())


if __name__ == "__main__":
    data_path = BASE_PATH / "data"
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    split = "test"  # train test
    out_name = "local_predicted_test" if split == "test" else "local_predicted_distill"
    size = 1000 if split == "test" else 5000

    llama_tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True
    )
    llama_tokenizer.padding_side = "right"  # Fix for fp16

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    # Model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        device_map={"": 0},
    )

    text_gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=llama_tokenizer,
        max_new_tokens=256,
        do_sample=False,
        return_full_text=False,
    )

    out_path = data_path / f"{out_name}.csv"

    data = pd.DataFrame(
        load_dataset("juancavallotti/multilingual-gec", split=split)
    ).head(size)

    data["text"] = data["modified"].str.replace("fix grammar: ", "")

    data["prompt"] = data.apply(lambda x: TEMPLATE.substitute(text=x["text"]), axis=1)

    print(data["prompt"].values[0])

    data["answer"] = [
        a[0]["generated_text"].strip()
        for a in tqdm.tqdm(text_gen(data["prompt"].tolist()))
    ]

    data.to_csv(out_path, index=False)
