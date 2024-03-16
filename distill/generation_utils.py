import torch
from transformers import (
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteriaList,
)


class StopAfterStopIsGenerated(LogitsProcessor):
    """Logits processor (to use with HuggingFace `generate()` method :
    https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/
    text_generation#transformers.generation_utils.GenerationMixin).

    This logit processor simply ensure that we generate at least one letter
    other than space, and that we don't generate anything after generating
    a space (in order to generate single word).

    Args:
        stop_token_ids (list[int]): ID of the space token.
        eos_token_id (int): ID of the EOS token.
    """

    def __init__(self, stop_token_ids: list[int], eos_token_id: int):
        super().__init__()

        self.stop_token_ids = stop_token_ids
        self.eos_token_id = eos_token_id

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        forced_eos = torch.full(
            (scores.size(1),), -float("inf"), device=input_ids.device
        )
        forced_eos[self.eos_token_id] = 0

        # Force generation of EOS after a space
        for stop_token_id in self.stop_token_ids:
            scores[input_ids[:, -1] == stop_token_id] = forced_eos
        return scores


def get_logit_criteria(tokenizer, stop_token=(".", "▁.")):
    logit_criteria = LogitsProcessorList(
        [
            StopAfterStopIsGenerated(
                stop_token_ids=[
                    tokenizer.vocab[a] for a in stop_token if a in tokenizer.vocab
                ],
                eos_token_id=tokenizer.eos_token_id,
            )
        ]
    )

    return logit_criteria


def get_stopping_criteria(tokenizer):
    def custom_stopping_criteria(
        input_ids: torch.LongTensor,
        score: torch.FloatTensor,
        stop_token=(".", "▁."),
        **kwargs,
    ) -> bool:
        return any(
            input_ids[-1] == tokenizer.vocab[a]
            for a in stop_token
            if a in tokenizer.vocab
        )

    stopping_criteria = StoppingCriteriaList([custom_stopping_criteria])

    return stopping_criteria


if __name__ == "__main__":
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    llama_tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True
    )

    print(llama_tokenizer.eos_token_id)
    print(llama_tokenizer.vocab["."])
    print(llama_tokenizer._convert_id_to_token(29901))
