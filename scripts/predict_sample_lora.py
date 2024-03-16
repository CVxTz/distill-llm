from pathlib import Path

import torch
import tqdm
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    pipeline,
)

from distill.dataset_utils import build_data

if __name__ == "__main__":
    BASE_PATH = Path(__file__).parents[1] / "outputs"
    DATA_PATH = Path(__file__).parents[1] / "data"

    refined_model = str(BASE_PATH / "TinyLlama-1.1B-refined")
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    out_name = "lora_predicted_test"  # local_predicted_validation local_predicted_test

    out_path = DATA_PATH / f"{out_name}.csv"

    llama_tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True
    )
    # llama_tokenizer.pad_token = llama_tokenizer.eos_token
    # Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # Model
    base_model = LlamaForCausalLM.from_pretrained(
        base_model_name,
        # quantization_config=quant_config,
        device_map={"": 0},
    )

    model = AutoPeftModelForCausalLM.from_pretrained(
        refined_model, device_map={"": 0}
    ).merge_and_unload()
    model.eval()

    text_gen = pipeline(
        task="text-generation",
        model=base_model,
        tokenizer=llama_tokenizer,
        max_new_tokens=256,
        do_sample=False,
        return_full_text=False,
    )
    text_gen.model = model

    test_data, df = build_data(llama_tokenizer, split="test")

    prompts = [test_data.get_sample(i)["prompt"] for i in range(len(test_data))]

    df["text"] = df["modified"].str.replace("fix grammar: ", "")

    df["answer"] = [
        text_gen(prompt)[0]["generated_text"].strip() for prompt in tqdm.tqdm(prompts)
    ]

    df.to_csv(out_path, index=False)
