
BASE_PATH="$(dirname "${BASH_SOURCE[0]}")/.."

export PYTHONPATH="${PYTHONPATH}:$BASE_PATH"

# Predict using the remote LLM
python $BASE_PATH/scripts/predict_sample_api.py --split "train"
python $BASE_PATH/scripts/predict_sample_api.py --split "test"

# Predict using the local LLM
$python $BASE_PATH/scripts/predict_sample_local.py --split "test"

# Finetune the local LLM using LoRa
$python $BASE_PATH/scripts/finetune.py

# Predict on the fine-tuned model
python $BASE_PATH/scripts/predict_sample_lora.py --split "test"

# Evaluate all
python $BASE_PATH/scripts/evaluate_sample.py --file "api_predicted_test.csv"
python $BASE_PATH/scripts/evaluate_sample.py --file "local_predicted_test.csv"
python $BASE_PATH/scripts/evaluate_sample.py --file "lora_predicted_test.csv"