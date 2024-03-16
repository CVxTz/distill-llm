from openai import OpenAI

from distill.config import (
    BASE_MODEL,
    TEMPERATURE,
)
from distill.logger import logger


def call_llm(
    client: OpenAI,
    prompt: str,
    model_name: str = BASE_MODEL,
    temperature: float = TEMPERATURE,
    stop: str = "\n",
):
    logger.debug(f"{prompt=}")

    completion = client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=100,
        temperature=temperature,
        stop=stop,
    )
    result = completion.model_dump()

    return result["choices"][0]["text"].strip()


if __name__ == "__main__":
    from distill.config import API_KEY, BASE_URL

    BASE_CLIENT = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    _prompt = """<s>[INST]
Your role is correct all grammatical errors in the following text.

Text: Il est très importante de parler une langue étrangère.
[/INST]
Output: Il est très important de parler une langue étrangère.</s>
[INST]
Text: Nadie dise ezo.
[/INST]
Output: Nadie dice eso.</s>
[INST]
Text: What is your favorite part of being a member of SWE RMS?
[/INST]
Output: What is your favorite part of being a member of SWE RMS?
[INST]
Text: Je être malade.
[/INST]
Output:"""

    answer = call_llm(
        client=BASE_CLIENT,
        prompt=_prompt,
        model_name=BASE_MODEL,
    )

    logger.info(f"{answer=}")
