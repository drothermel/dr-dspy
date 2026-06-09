import random
from collections import deque
from types import SimpleNamespace

from dspy.clients.finetune import FinetuneAssistantMessage, FinetuneChatMessage, GRPOChatData
from dspy.clients.lm import LM
from dspy.teleprompt.grpo.batch_dispatch import dispatch_training_step
from dspy.teleprompt.grpo.session import GRPOCompileSession


def _sample_group(reward: float = 1.0) -> list[GRPOChatData]:
    return [
        GRPOChatData(
            messages=[FinetuneChatMessage(role="user", content="hi")],
            completion=FinetuneAssistantMessage(content="ok"),
            reward=reward,
        )
    ]


class _FakeJob:
    def __init__(self, pending_batch_ids: list[int]) -> None:
        self._pending_batch_ids = pending_batch_ids
        self.steps: list = []

    def get_status(self):
        return SimpleNamespace(pending_batch_ids=self._pending_batch_ids)

    def step(self, *, train_data, train_data_format):
        self.steps.append((train_data, train_data_format))


def test_dispatch_training_step_matches_pending_batch_ids():
    session = GRPOCompileSession()
    lm = LM("openai/gpt-4o-mini")
    job_key = (lm, None)
    job = _FakeJob(pending_batch_ids=[1, 2])
    group = _sample_group()
    train_data = [group, group]

    submitted = dispatch_training_step(
        session,
        job_key=job_key,
        job=job,
        train_data=train_data,
        train_batch_per_predictor=[train_data],
        num_rollouts_per_grpo_step=1,
        rng=random.Random(0),
    )

    assert submitted is True
    assert len(job.steps) == 1
    final_train_data, _fmt = job.steps[0]
    assert {item.batch_id for item in final_train_data} == {1, 2}
    assert set(session.fulfilled_batch_ids) == {1, 2}


def test_dispatch_training_step_reuses_group_queue():
    session = GRPOCompileSession()
    lm = LM("openai/gpt-4o-mini")
    job_key = (lm, None)
    session.fulfilled_batch_ids = [1]
    session.group_queues[job_key] = deque([_sample_group(0.5)])
    job = _FakeJob(pending_batch_ids=[2])
    train_data = [_sample_group(1.0)]

    dispatch_training_step(
        session,
        job_key=job_key,
        job=job,
        train_data=train_data,
        train_batch_per_predictor=[train_data],
        num_rollouts_per_grpo_step=1,
        rng=random.Random(0),
    )

    assert job.steps[0][0][0].group[0].reward == 0.5
