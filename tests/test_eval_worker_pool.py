# SPDX-License-Identifier: Apache-2.0
"""Scheduling tests for BaseBenchmark.run: batch_size is a worker-pool width."""

import asyncio
import contextlib
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from omlx.eval.base import BaseBenchmark
from omlx.eval.humaneval import HumanEvalBenchmark
from omlx.eval.livecodebench import LiveCodeBenchBenchmark
from omlx.eval.mbpp import MBPPBenchmark

WAIT = 1.0

CODE_BENCHMARKS = [HumanEvalBenchmark, MBPPBenchmark, LiveCodeBenchBenchmark]


class _GatedEngine:
    """Engine whose chat() calls finish only when the test releases them.

    Questions are keyed by the first message's content (the item id).
    """

    is_external_api = False
    model_type = None

    def __init__(self, reply: Callable[[str, bool], str] | None = None):
        self.reply = reply or (lambda qid, thinking: "A")
        self.calls: list[dict[str, Any]] = []
        self.in_flight: set[str] = set()
        self.max_in_flight = 0
        self.cancelled: list[str] = []
        self._started: dict[str, asyncio.Event] = {}
        self._release: dict[str, asyncio.Event] = {}

    @staticmethod
    def _event(table: dict[str, asyncio.Event], qid: str) -> asyncio.Event:
        if qid not in table:
            table[qid] = asyncio.Event()
        return table[qid]

    def release(self, *qids: str) -> None:
        for qid in qids:
            self._event(self._release, qid).set()

    def started(self, qid: str) -> bool:
        return self._event(self._started, qid).is_set()

    async def wait_started(self, *qids: str) -> None:
        for qid in qids:
            await asyncio.wait_for(self._event(self._started, qid).wait(), WAIT)

    async def chat(self, messages, **kwargs):
        qid = messages[0]["content"]
        thinking = kwargs["chat_template_kwargs"]["enable_thinking"]
        self.calls.append({"id": qid, "thinking": thinking})
        self.in_flight.add(qid)
        self.max_in_flight = max(self.max_in_flight, len(self.in_flight))
        self._event(self._started, qid).set()
        try:
            await self._event(self._release, qid).wait()
        except asyncio.CancelledError:
            self.cancelled.append(qid)
            raise
        finally:
            self.in_flight.discard(qid)
        return SimpleNamespace(text=self.reply(qid, thinking))


class _EchoBenchmark(BaseBenchmark):
    name = "echo"

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        return []

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        return [{"role": "user", "content": item["id"]}]

    def extract_answer(self, response: str, item: dict) -> str:
        return response.strip()

    def check_answer(self, predicted: str, item: dict) -> bool:
        return predicted.startswith(item["answer"])


def _items(n: int) -> list[dict]:
    return [{"id": f"q{i}", "answer": "A"} for i in range(n)]


def _code_benchmark(cls, monkeypatch) -> BaseBenchmark:
    bench = cls()
    monkeypatch.setattr(
        bench, "format_prompt", lambda item: [{"role": "user", "content": item["id"]}]
    )
    monkeypatch.setattr(bench, "extract_answer", lambda response, item: response.strip())
    monkeypatch.setattr(bench, "check_answer", lambda predicted, item: predicted.startswith("A"))
    return bench


@pytest.fixture(params=["base", "humaneval", "mbpp", "livecodebench"])
def benchmark(request, monkeypatch) -> BaseBenchmark:
    if request.param == "base":
        return _EchoBenchmark()
    cls = {
        "humaneval": HumanEvalBenchmark,
        "mbpp": MBPPBenchmark,
        "livecodebench": LiveCodeBenchBenchmark,
    }[request.param]
    return _code_benchmark(cls, monkeypatch)


@contextlib.asynccontextmanager
async def _running(coro):
    task = asyncio.create_task(coro)
    try:
        yield task
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


async def test_next_question_starts_as_soon_as_a_slot_frees(benchmark):
    engine = _GatedEngine()
    async with _running(benchmark.run(engine, _items(4), batch_size=2)) as run:
        await engine.wait_started("q0", "q1")
        assert not engine.started("q2")

        engine.release("q1")
        await engine.wait_started("q2")
        assert engine.in_flight == {"q0", "q2"}

        engine.release("q2")
        await engine.wait_started("q3")
        engine.release("q0", "q3")
        result = await asyncio.wait_for(run, WAIT)

    assert engine.max_in_flight == 2
    assert [c["id"] for c in engine.calls] == ["q0", "q1", "q2", "q3"]
    assert [r.question_id for r in result.question_results] == ["q0", "q1", "q2", "q3"]
    assert result.correct_count == 4


async def test_each_question_reports_its_own_latency():
    engine = _GatedEngine()
    async with _running(_EchoBenchmark().run(engine, _items(3), batch_size=3)) as run:
        await engine.wait_started("q0", "q1", "q2")
        engine.release("q2")
        await asyncio.sleep(0.05)
        engine.release("q1")
        await asyncio.sleep(0.05)
        engine.release("q0")
        result = await asyncio.wait_for(run, WAIT)

    times = {r.question_id: r.time_seconds for r in result.question_results}
    assert times["q0"] > times["q1"] > times["q2"]


async def test_progress_is_reported_after_every_question():
    engine = _GatedEngine()
    engine.release("q0", "q1", "q2", "q3")
    progress: list[tuple[int, int]] = []

    async def on_progress(current: int, total: int) -> None:
        progress.append((current, total))

    await asyncio.wait_for(
        _EchoBenchmark().run(engine, _items(4), on_progress, batch_size=4), WAIT
    )

    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]


async def test_think_tags_in_probe_rerun_every_non_thinking_result():
    def reply(qid: str, thinking: bool) -> str:
        if qid == "q0" and not thinking:
            return "<think>hmm</think>A"
        return "A"

    engine = _GatedEngine(reply)
    async with _running(_EchoBenchmark().run(engine, _items(4), batch_size=2)) as run:
        await engine.wait_started("q0", "q1")
        engine.release("q1")  # probe result without think tags, recorded
        await engine.wait_started("q2")
        engine.release("q2")  # non-probe result recorded before the switch
        await engine.wait_started("q3")  # in flight, started without thinking
        engine.release("q0")  # probe result with think tags: switch
        engine.release("q3")
        result = await asyncio.wait_for(run, WAIT)

    assert result.thinking_used is True
    assert result.correct_count == 4
    assert [r.question_id for r in result.question_results] == ["q0", "q1", "q2", "q3"]
    modes: dict[str, list[bool]] = {}
    for call in engine.calls:
        modes.setdefault(call["id"], []).append(call["thinking"])
    assert modes == {qid: [False, True] for qid in ("q0", "q1", "q2", "q3")}


async def test_think_tags_outside_the_probe_do_not_switch_mode():
    engine = _GatedEngine(
        lambda qid, thinking: "<think>hmm</think>A" if qid == "q2" else "A"
    )
    engine.release("q0", "q1", "q2", "q3")

    result = await asyncio.wait_for(
        _EchoBenchmark().run(engine, _items(4), batch_size=2), WAIT
    )

    assert result.thinking_used is False
    assert [c["thinking"] for c in engine.calls] == [False] * 4


async def test_cancelling_progress_callback_stops_in_flight_questions():
    engine = _GatedEngine()

    async def on_progress(current: int, total: int) -> None:
        raise asyncio.CancelledError()

    async with _running(
        _EchoBenchmark().run(engine, _items(4), on_progress, batch_size=2)
    ) as run:
        await engine.wait_started("q0", "q1")
        engine.release("q0")
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run, WAIT)

    # q0's worker may already have picked up q2 before the callback ran.
    assert "q1" in engine.cancelled
    assert set(engine.cancelled) <= {"q1", "q2"}
    assert engine.in_flight == set()
    assert not engine.started("q3")


async def test_hard_cancel_leaves_no_question_in_flight():
    engine = _GatedEngine()
    run = asyncio.create_task(_EchoBenchmark().run(engine, _items(4), batch_size=2))
    await engine.wait_started("q0", "q1")

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run, WAIT)

    assert sorted(engine.cancelled) == ["q0", "q1"]
    assert engine.in_flight == set()
    assert not engine.started("q2")


@pytest.mark.parametrize(
    ("cls", "expected_label"),
    [
        (HumanEvalBenchmark, "(unit tests)"),
        (MBPPBenchmark, "(test cases)"),
        (LiveCodeBenchBenchmark, "(test cases)"),
    ],
)
async def test_code_benchmarks_keep_test_case_result_format(
    cls, expected_label, monkeypatch
):
    bench = _code_benchmark(cls, monkeypatch)
    engine = _GatedEngine(lambda qid, thinking: "A" * 300)
    engine.release("q0")

    result = await asyncio.wait_for(bench.run(engine, _items(1), batch_size=1), WAIT)

    question = result.question_results[0]
    assert question.correct is True
    assert question.expected == expected_label
    assert question.predicted == "A" * 200 + "..."


@pytest.mark.parametrize("cls", CODE_BENCHMARKS)
async def test_code_benchmarks_do_not_auto_switch_thinking(cls, monkeypatch):
    bench = _code_benchmark(cls, monkeypatch)
    engine = _GatedEngine(lambda qid, thinking: "<think>x</think>A")
    engine.release("q0")

    result = await asyncio.wait_for(bench.run(engine, _items(1), batch_size=1), WAIT)

    assert result.thinking_used is False
    assert [c["thinking"] for c in engine.calls] == [False]

