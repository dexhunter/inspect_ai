"""Deterministic CPU benchmarks for the intake hot-path targets.

One target per invocation, so each INTAKE.json ``bench_cmd`` measures a
single function. Prints ``median_ms`` over repeated calls after warmup.

Usage:
    uv run --no-sync python osspr-intake/harness/bench.py --target exceeds_max_depth
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

from inspect_ai._util.constants import get_deserializing_context
from inspect_ai._util.json import exceeds_max_depth
from inspect_ai.dataset._util import data_to_samples, record_to_sample_fn
from inspect_ai.event._model import ModelEvent
from inspect_ai.event._tool import ToolEvent
from inspect_ai.log._condense import (
    MAX_SAMPLE_DUMP_DEPTH,
    attachment_refs_from_object,
    condense_sample,
)
from inspect_ai.log._log import EvalSample
from inspect_ai.model._chat_message import (
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.model._model_output import ModelOutput
from inspect_ai.scorer._metrics.std import _bootstrap_means
from inspect_ai.tool._tool_call import ToolCall
from inspect_ai.tool._tool_info import ToolInfo

WARMUP = 5
REPEATS = 49

_PARAGRAPH = (
    "Inspect the repository and report the function that dominates CPU time. "
    "Include the file path, how often it is called, and whether tests cover it. "
)


def _paragraph(index: int, repeats: int) -> str:
    return (_PARAGRAPH * repeats) + f" turn={index}"


def _median_ms(fn: Callable[[], None]) -> float:
    for _ in range(WARMUP):
        fn()
    samples: list[float] = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def _agent_sample(sample_id: int, turns: int) -> EvalSample:
    """One agent sample whose model events carry a growing message history."""
    tools = [
        ToolInfo(name="bash", description="Run a shell command and return stdout."),
        ToolInfo(name="read_file", description="Read a workspace file."),
    ]
    messages: list[ChatMessageUser | ChatMessageAssistant | ChatMessageTool] = [
        ChatMessageUser(content=_paragraph(0, 6), id=f"s{sample_id}-u0", source="input")
    ]
    events: list[ModelEvent | ToolEvent] = []
    for turn in range(turns):
        call_id = f"s{sample_id}-c{turn}"
        assistant = ChatMessageAssistant(
            id=f"s{sample_id}-a{turn}",
            content=_paragraph(turn, 3),
            source="generate",
            model="mock/cpu",
            tool_calls=[
                ToolCall(
                    id=call_id,
                    function="bash",
                    arguments={"cmd": f"rg -n target src | head -n {turn % 20}"},
                )
            ],
        )
        tool_message = ChatMessageTool(
            id=f"s{sample_id}-t{turn}",
            content=_paragraph(turn, 8),
            tool_call_id=call_id,
            function="bash",
        )
        events.append(
            ModelEvent(
                model="mock/cpu",
                input=list(messages),
                tools=tools,
                tool_choice="auto",
                config={},
                output=ModelOutput.from_content("mock/cpu", assistant.content),
            )
        )
        events.append(
            ToolEvent(
                id=call_id,
                function="bash",
                arguments={"cmd": f"rg -n target src | head -n {turn % 20}"},
                result=tool_message.content,
            )
        )
        messages.extend(
            [
                assistant,
                tool_message,
                ChatMessageUser(
                    id=f"s{sample_id}-u{turn + 1}",
                    content="Continue. " + _paragraph(turn, 2),
                    source="input",
                ),
            ]
        )
    return EvalSample(
        id=sample_id,
        epoch=1,
        input=messages[0].content,
        target="hotpath",
        messages=messages,
        output=ModelOutput.from_content("mock/cpu", str(messages[-2].content)),
        events=events,
        metadata={"cluster": sample_id % 8},
        store={"notes": _paragraph(sample_id, 2)},
    )


def _dumped_samples() -> list[object]:
    """Python dumps of condensed samples, the graph both walks actually see."""
    dumps: list[object] = []
    for sample_id in range(4):
        condensed = condense_sample(_agent_sample(sample_id, 36))
        dumps.append(condensed.model_dump(mode="python", exclude={"attachments"}))
    return dumps


def bench_exceeds_max_depth() -> float:
    """Walk the same per-sample dumps ``condense_sample`` depth-checks."""
    dumps = _dumped_samples()

    def run() -> None:
        for dumped in dumps:
            exceeds_max_depth(dumped, MAX_SAMPLE_DUMP_DEPTH)

    return _median_ms(run)


def bench_attachment_refs_from_object() -> float:
    """Collect attachment refs from those same condensed-sample dumps."""
    dumps = _dumped_samples()

    def run() -> None:
        for dumped in dumps:
            attachment_refs_from_object(dumped)

    return _median_ms(run)


def _message_payloads() -> list[dict[str, object]]:
    """Repeated conversation prefixes, as a log read re-parses model inputs.

    Every payload is a user message so the measured call is
    ``ChatMessageBase._wrap`` on ``ChatMessageUser``, not union dispatch.
    """
    messages: list[dict[str, object]] = []
    for index in range(80):
        messages.append(
            {
                "role": "user",
                "content": _paragraph(index, 4),
                "id": f"m{index}",
                "source": "input",
                "metadata": {"turn": index, "tool": "bash"},
            }
        )
    payloads: list[dict[str, object]] = []
    for turn in range(80):
        payloads.extend(messages[: turn + 1])
    return payloads


def bench_chat_message_wrap() -> float:
    """Re-validate one sample's message stream with the deserializer cache."""
    payloads = _message_payloads()

    def run() -> None:
        context = get_deserializing_context()
        for payload in payloads:
            ChatMessageUser.model_validate(payload, context=context)

    return _median_ms(run)


def _dataset_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in range(8000):
        records.append(
            {
                "input": f"Question {index}: " + _paragraph(index % 50, 2),
                "target": str(index % 11),
                "id": index,
                "metadata": {"cluster": index % 8, "topic": f"topic-{index % 40}"},
                "choices": [f"option-{choice}-{index % 5}" for choice in range(4)],
            }
        )
    return records


def bench_record_to_sample() -> float:
    """Map already-parsed JSON rows through the default dataset field spec."""
    records = _dataset_records()
    convert = record_to_sample_fn(None)

    def run() -> None:
        data_to_samples(records, convert, False)

    return _median_ms(run)


def bench_bootstrap_means() -> float:
    """Default 1000-resample bootstrap of the mean used by ``ci(method='bootstrap')``."""
    import numpy as np

    values = [((index * 17) % 100) / 100.0 for index in range(4000)]
    np.random.seed(0)

    def run() -> None:
        _bootstrap_means(values, None, 1000)

    return _median_ms(run)


_TARGETS: dict[str, Callable[[], float]] = {
    "exceeds_max_depth": bench_exceeds_max_depth,
    "attachment_refs_from_object": bench_attachment_refs_from_object,
    "ChatMessageBase._wrap": bench_chat_message_wrap,
    "record_to_sample": bench_record_to_sample,
    "_bootstrap_means": bench_bootstrap_means,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        required=True,
        choices=sorted(_TARGETS),
        help="Function to measure.",
    )
    args = parser.parse_args()
    median_ms = _TARGETS[args.target]()
    print(
        f"target={args.target} median_ms={median_ms:.4f} "
        f"repeats={REPEATS} warmup={WARMUP}"
    )


if __name__ == "__main__":
    main()
