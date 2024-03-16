from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from distill.dataset_utils import CustomDataCollator, build_distill_data

if __name__ == "__main__":
    IN_DATA_PATH = Path(__file__).parents[1] / "data" / "api_predicted_distill.csv"
    BASE_PATH = Path(__file__).parents[1] / "outputs"

    EPOCHS = 16

    BASE_PATH.mkdir(exist_ok=True)

    # Model and tokenizer names
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    refined_model = str(BASE_PATH / "TinyLlama-1.1B-refined")
    # Tokenizer
    llama_tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True
    )

    training_data, _ = build_distill_data(tokenizer=llama_tokenizer, path=IN_DATA_PATH)
    collator = CustomDataCollator(
        pad_token=llama_tokenizer.pad_token_id, ignore_index=-100
    )

    print(f"{len(training_data)=}")

    # Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # Model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map={"": 0},
        quantization_config=quant_config,
        trust_remote_code=True,
    )

    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    # target_modules = [
    #     "q_proj",
    #     "k_proj",
    #     "v_proj",
    #     "o_proj",
    #     "gate_proj",
    #     "down_proj",
    #     "up_proj",
    #     "lm_head",
    # ]
    # LoRA Config
    peft_parameters = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules=target_modules,
    )

    base_model = prepare_model_for_kbit_training(base_model)
    base_model = get_peft_model(base_model, peft_parameters)

    # Training Params
    train_params = TrainingArguments(
        output_dir=str(BASE_PATH / "results_modified"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=len(training_data) // 10,
        logging_steps=len(training_data) // 100,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.05,
        fp16=True,
        max_steps=-1,
        group_by_length=False,
        max_grad_norm=0.3,
    )
    # Trainer
    fine_tuning = SFTTrainer(
        model=base_model,
        train_dataset=training_data,
        data_collator=collator,
        peft_config=peft_parameters,
        dataset_text_field="Why is this mandatory ? lol",
        tokenizer=llama_tokenizer,
        args=train_params,
        max_seq_length=llama_tokenizer.model_max_length,
    )

    print(fine_tuning.model.print_trainable_parameters())
    # Training
    fine_tuning.train()
    # Save Model
    fine_tuning.model.save_pretrained(refined_model)
