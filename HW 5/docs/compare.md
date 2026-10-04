# Базовая модель против адаптера

Пять фиксированных GitHub Issues из validation ДЗ 4. Это демонстрация подключения адаптера; по пяти примерам нельзя оценить качество на всей задаче.

Адаптер: `models/adapter_all_layers`. Генерация жадная, до 16 токенов.

Системная инструкция: Classify the GitHub issue into exactly one category: bug, documentation, feature, or question. Reply with the category only.

## 1. Unclear licensing for contrib/sparsity

Источник: `github:NVIDIA/apex#1789`. Метка датасета: **bug**.

```text
Title: Unclear licensing for contrib/sparsity

Body:
**Describe the Bug**
The https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity folder describes itself as All Rights Reserved.

Is this covered by the overall BSD-3-Clause license of apex, or is it proprietary NVIDIA and not intended for others to use?
```

**База:**

> bug

**Адаптер:**

> bug

## 2. The documentation for CSS variables should clearly state the scope of variables

Источник: `github:Textualize/textual#3951`. Метка датасета: **documentation**.

```text
Title: The documentation for CSS variables should clearly state the scope of variables

Body:
The [documentation for CSS variables](https://textual.textualize.io/guide/CSS/#css-variables) doesn't appear to say what the scope of those variables is. For example, it would seem that if you're using `CSS_PATH`, you can't define a series of variables in one file earlier in the path, and then use those variables in files later on in the path. If this is working as intended we should document this.
```

**База:**

> feature

**Адаптер:**

> documentation

## 3. pls try qwen3(think version) and deepseek-r1-0528

Источник: `github:TauricResearch/TradingAgents#12`. Метка датасета: **feature**.

```text
Title: pls try qwen3(think version) and deepseek-r1-0528

Body:
As two excellent and inexpensive models, it would be nice to see the results of testing these two models, perhaps to make this project more practicable!
```

**База:**

> feature

**Адаптер:**

> feature

## 4. pytorch-ada

Источник: `github:Lightning-AI/pytorch-lightning#19612`. Метка датасета: **question**.

```text
Title: pytorch-ada

Body:
### Description & Motivation

Hi,
Do you officially support pytorch-ada https://pytorch-ada.readthedocs.io/en/latest/getting_started.html ?
Best

### Pitch

_No response_

### Alternatives

_No response_

### Additional context

_No response_

cc @borda
```

**База:**

> question

**Адаптер:**

> question

## 5. Not Able to install apex.

Источник: `github:NVIDIA/apex#1800`. Метка датасета: **bug**.

```text
Title: Not Able to install apex.

Body:
from pyramid.session import UnencryptedCookieSessionFactoryConfig
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ImportError: cannot import name 'UnencryptedCookieSessionFactoryConfig' from 'pyramid.session' (unknown location)


plese help . need your suggestion .. Thanks in Advance .
```

**База:**

> question

**Адаптер:**

> question

## Наблюдение

Дословное совпадение с меткой датасета: база 3/5, адаптер 4/5. Метки получены по labels GitHub и могут содержать ошибки. Полная оценка на независимом наборе относится к следующей работе.
