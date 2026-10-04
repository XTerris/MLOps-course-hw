"""Шаблон инференса совпадает с apply_chat_template из ДЗ 4."""


def build_prompt(tokenizer, system: str, user: str, enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
    )
