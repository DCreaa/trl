# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
from collections.abc import Callable

from ..import_utils import is_jsonschema_available


if is_jsonschema_available():
    import jsonschema


def think_format_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
    r"""
    Reward function that checks if the reasoning process is enclosed within `"<think>"` and `"</think>"` tags. The
    function returns a reward of 1.0 if the format is correct, otherwise 0.0.

    Args:
        completions (`list[list[dict[str, str]]]`):
            List of completions to be evaluated. Each completion must be a list of one message, i.e. a dictionary
            containing the key `"content"` with the value being the text of the completion.
        **kwargs:
            Additional keyword arguments. This function does not use them, but they are required in the function
            signature to ensure compatibility with trainers like [`GRPOTrainer`].

    Returns:
        `list[float]`:
            A list of rewards, where each reward is 1.0 if the completion matches the expected format, otherwise 0.0.

    Example:
    ```python
    >>> from trl.rewards import think_format_reward

    >>> completions = [
    ...     [{"content": "<think>\nThis is my reasoning.\n</think>\nThis is my answer."}],
    ...     [{"content": "<think>\nThis is my reasoning.\nThis is my answer."}],
    ... ]
    >>> think_format_reward(completions)
    [1.0, 0.0]
    ```
    """
    pattern = r"^<think>(?!.*<think>)(.*?)</think>.*$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def json_schema_reward(
    completions: list[list[dict[str, str]]],
    schema: list[str | dict | None],
    log_extra: Callable[[str, list], None] | None = None,
    **kwargs,
) -> list[float | None]:
    r"""
    Reward function that checks if the completion is a JSON that validates against the JSON Schema of its example. The
    function returns 1.0 if the JSON is valid, otherwise 0.0: a single wrong field makes the whole completion wrong.

    The completion must contain only the JSON, optionally wrapped in a Markdown code block. If the schema of an example
    is `None`, the reward is `None` and the example is skipped, which is useful when only some prompts of the dataset
    ask for a JSON output.

    Args:
        completions (`list[list[dict[str, str]]]`):
            List of completions to be evaluated. Each completion must be a list of one message, i.e. a dictionary
            containing the key `"content"` with the value being the text of the completion.
        schema (`list[str | dict | None]`):
            List of JSON Schemas, one per completion, as JSON strings or dictionaries. When training, it is filled
            automatically from the `"schema"` column of the dataset.
        log_extra (`callable`, *optional*):
            Callable to log extra columns to the completions table, provided automatically by the trainer. It is used
            to log why each completion is not valid. Defaults to `None` to allow calling the function directly outside
            of a trainer (e.g., for testing).
        **kwargs:
            Additional keyword arguments. This function does not use them, but they are required in the function
            signature to ensure compatibility with trainers like [`GRPOTrainer`].

    Returns:
        `list[float | None]`:
            A list of rewards: 1.0 if the completion validates against its schema, 0.0 otherwise, and `None` if the
            schema is `None`.

    Example:
    ```python
    >>> from trl.rewards import json_schema_reward

    >>> schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    >>> completions = [
    ...     [{"content": '{"name": "Ada"}'}],
    ...     [{"content": '{"age": 36}'}],
    ...     [{"content": "Ada"}],
    ... ]
    >>> json_schema_reward(completions, schema=[schema, schema, schema])
    [1.0, 0.0, 0.0]
    ```
    """
    if not is_jsonschema_available():
        raise ImportError("Please install the `jsonschema` package to use json_schema_reward")

    rewards = []
    errors = []
    for completion, example_schema in zip(completions, schema, strict=True):
        if example_schema is None:
            # This example does not ask for a JSON output: we assign `None` to skip it
            rewards.append(None)
            errors.append("[no schema]")
            continue
        if isinstance(example_schema, str):
            example_schema = json.loads(example_schema)

        content = completion[0]["content"].strip()
        # Many models wrap the JSON in a Markdown code block: keep only what is inside it
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            content = match.group(1)

        try:
            jsonschema.validate(json.loads(content), example_schema)
            rewards.append(1.0)
            errors.append("")
        except json.JSONDecodeError as e:
            rewards.append(0.0)
            errors.append(f"invalid JSON: {e.msg}")
        except jsonschema.ValidationError as e:
            rewards.append(0.0)
            errors.append(e.message)

    if log_extra is not None:
        log_extra("json_schema_error", errors)

    return rewards
