import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from unittest import mock
from unittest.mock import AsyncMock

import pytest

try:
    from litellm import Choices, Message, ModelResponse
except ImportError:
    pytest.skip("litellm is not installed", allow_module_level=True)  # ty: ignore[too-many-positional-arguments]

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.clients.lm import LM
from dspy.dsp.utils.settings import settings
from dspy.predict.parallel import Parallel
from dspy.predict.predict import Predict
from dspy.primitives.module import Module


def test_basic_dspy_settings():
    settings.configure(lm=LM("openai/gpt-4o"), adapter=JSONAdapter(), callbacks=[lambda x: x])
    assert settings.lm.model == "openai/gpt-4o"
    assert isinstance(settings.adapter, JSONAdapter)
    assert len(settings.callbacks) == 1


def test_forbid_configure_call_in_child_thread():
    settings.configure(lm=LM("openai/gpt-4o"), adapter=JSONAdapter(), callbacks=[lambda x: x])

    def worker():
        with pytest.raises(RuntimeError, match="settings can only be changed"):
            settings.configure(lm=LM("openai/gpt-4o-mini"), callbacks=[])

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(worker)


def test_dspy_context():
    settings.configure(lm=LM("openai/gpt-4o"), adapter=JSONAdapter(), callbacks=[lambda x: x])
    with settings.context(lm=LM("openai/gpt-4o-mini"), callbacks=[]):
        assert settings.lm.model == "openai/gpt-4o-mini"
        assert len(settings.callbacks) == 0

    assert settings.lm.model == "openai/gpt-4o"
    assert len(settings.callbacks) == 1


def test_dspy_context_parallel():
    settings.configure(lm=LM("openai/gpt-4o"), adapter=JSONAdapter(), callbacks=[lambda x: x])

    def worker(i):
        with settings.context(lm=LM("openai/gpt-4o-mini"), trace=[i], callbacks=[]):
            assert settings.lm.model == "openai/gpt-4o-mini"
            assert settings.trace == [i]
            assert len(settings.callbacks) == 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(worker, range(3))

    assert settings.lm.model == "openai/gpt-4o"
    assert len(settings.callbacks) == 1


def test_dspy_context_with_dspy_parallel():
    settings.configure(lm=LM("openai/gpt-4o", cache=False), adapter=ChatAdapter())

    class MyModule(Module):
        def __init__(self):
            self.predict = Predict("question -> answer")

        async def aforward(self, question: str) -> str:
            lm = LM("openai/gpt-4o-mini", cache=False) if "France" in question else settings.lm
            with settings.context(lm=lm):
                await asyncio.sleep(1)
                assert settings.lm.model == lm.model
                return await self.predict(question=question)

    with mock.patch(
        "dspy.clients.lm.alitellm_completion",
        new_callable=AsyncMock,
        return_value=ModelResponse(
            choices=[Choices(message=Message(content="[[ ## answer ## ]]\nParis"))],
            model="openai/gpt-4o-mini",
        ),
    ) as mock_completion:
        module = MyModule()
        parallelizer = Parallel()
        input_pairs = [
            (module, {"question": "What is the capital of France?"}),
            (module, {"question": "What is the capital of Germany?"}),
        ]
        asyncio.run(parallelizer(input_pairs))

        # Verify mock was called correctly
        assert mock_completion.call_count == 2
        for call_args in mock_completion.call_args_list:
            request = call_args.kwargs["request"]
            if "France" in request["messages"][-1]["content"]:
                # France question uses gpt-4o-mini
                assert request["model"] == "openai/gpt-4o-mini"
            else:
                # Germany question uses gpt-4o
                assert request["model"] == "openai/gpt-4o"

        # The main thread is not affected by the context
        assert settings.lm.model == "openai/gpt-4o"


@pytest.mark.asyncio
async def test_dspy_context_with_async_task_group():
    class MyModule(Module):
        def __init__(self):
            self.predict = Predict("question -> answer")

        async def aforward(self, question: str) -> str:
            lm = LM("openai/gpt-4o-mini", cache=False) if "France" in question else LM("openai/gpt-4o", cache=False)
            with settings.context(lm=lm, trace=[]):
                await asyncio.sleep(1)
                assert settings.lm.model == lm.model
                result = await self.predict.acall(question=question)
                assert len(settings.trace) == 1
                return result

    module = MyModule()

    with settings.context(lm=LM("openai/gpt-4.1", cache=False), adapter=ChatAdapter()):
        with mock.patch("litellm.acompletion") as mock_completion:
            mock_completion.return_value = ModelResponse(
                choices=[Choices(message=Message(content="[[ ## answer ## ]]\nParis"))],
                model="openai/gpt-4o-mini",
            )

            # Define the coroutines to be run
            coroutines = [
                module.acall(question="What is the capital of France?"),
                module.acall(question="What is the capital of France?"),
                module.acall(question="What is the capital of Germany?"),
                module.acall(question="What is the capital of Germany?"),
            ]

            # Run them concurrently and gather results
            results = await asyncio.gather(*coroutines)

        assert results[0].answer == "Paris"
        assert results[1].answer == "Paris"
        assert results[2].answer == "Paris"
        assert results[3].answer == "Paris"

        # Verify mock was called correctly
        assert mock_completion.call_count == 4
        # France question uses gpt-4o-mini
        assert mock_completion.call_args_list[0].kwargs["model"] == "openai/gpt-4o-mini"
        assert mock_completion.call_args_list[1].kwargs["model"] == "openai/gpt-4o-mini"
        # Germany question uses gpt-4o
        assert mock_completion.call_args_list[2].kwargs["model"] == "openai/gpt-4o"
        assert mock_completion.call_args_list[3].kwargs["model"] == "openai/gpt-4o"

        # The main thread is not affected by the context
        assert settings.lm.model == "openai/gpt-4.1"
        assert settings.trace == []


@pytest.mark.asyncio
async def test_dspy_configure_allowance_async():
    def bar1():
        # `settings.configure` is disallowed in different async tasks from the initial one.
        # In this case, foo1 (async) calls bar1 (sync), and bar1 uses the async task from foo1.
        with pytest.raises(RuntimeError) as e:
            settings.configure(lm=LM("openai/gpt-4o"))
        assert "settings.configure(...) can only be called from the same async" in str(e.value)

    async def foo1():
        bar1()
        await asyncio.sleep(0.1)

    async def foo2():
        # `settings.configure` is disallowed in different async tasks from the initial one.
        with pytest.raises(RuntimeError) as e:
            settings.configure(lm=LM("openai/gpt-4o"))
        assert "settings.configure(...) can only be called from the same async" in str(e.value)
        await asyncio.sleep(0.1)

    async def foo3():
        # `settings.context` is allowed in different async tasks from the initial one.
        with settings.context(lm=LM("openai/gpt-4o")):
            await asyncio.sleep(0.1)

    async def foo4():
        # foo4 is directly invoked by the entry task, so it has the same async task as the entry task.
        settings.configure(lm=LM("openai/gpt-4o"))
        await asyncio.sleep(0.1)

    # `settings.configure` is allowed to be called multiple times in the same async task.
    settings.configure(lm=LM("openai/gpt-4o-mini"))
    settings.configure(lm=LM("openai/gpt-4o"))
    settings.configure(adapter=JSONAdapter())

    await asyncio.gather(foo1(), foo2(), foo3())

    foo4()  # ty:ignore[unused-awaitable]


def test_dspy_settings_save_load(tmp_path):
    settings.configure(lm=LM("openai/gpt-4o"), adapter=JSONAdapter(), callbacks=[lambda x: x])

    settings.save(tmp_path / "settings.pkl")
    settings.configure(lm=None, adapter=None, callbacks=None)

    loaded_settings = settings.load(tmp_path / "settings.pkl", allow_pickle=True)
    settings.configure(**loaded_settings)
    assert settings.lm.model == "openai/gpt-4o"
    assert isinstance(settings.adapter, JSONAdapter)
    assert len(settings.callbacks) == 1


def test_dspy_settings_save_exclude_keys(tmp_path):
    settings.configure(lm=LM("openai/gpt-4o"), adapter=JSONAdapter(), track_usage=True)

    settings.save(tmp_path / "settings.pkl", exclude_keys=["adapter", "track_usage"])
    settings.configure(lm=None, adapter=None, track_usage=False)

    loaded_settings = settings.load(tmp_path / "settings.pkl", allow_pickle=True)
    settings.configure(**loaded_settings)
    assert settings.lm.model == "openai/gpt-4o"
    assert settings.adapter is None
    assert not settings.track_usage


def test_settings_save_with_extra_modules(tmp_path):
    # Create a temporary Python file with our custom module
    custom_module_path = tmp_path / "custom_module.py"
    with open(custom_module_path, "w") as f:
        f.write(
            """
def callback(x):
    return x + 1
"""
        )

    # Add the tmp_path to Python path so we can import the module
    sys.path.insert(0, str(tmp_path))
    try:
        import custom_module  # ty:ignore[unresolved-import]

        settings.configure(callbacks=[custom_module.callback])

        settings_path = tmp_path / "settings.pkl"
        sys.path.insert(0, str(tmp_path))

        settings.configure(callbacks=[custom_module.callback])
        settings.save(settings_path, modules_to_serialize=[custom_module])

        # Remove the custom module again to simulate it not being available at load time
        sys.modules.pop("custom_module", None)
        sys.path.remove(str(tmp_path))
        del custom_module

        settings.configure(callbacks=None)

        # Loading should now succeed and preserve the adapter instance
        loaded_settings = settings.load(settings_path, allow_pickle=True)
        settings.configure(**loaded_settings)

        assert settings.callbacks[0](3) == 4

    finally:
        # Only need to clean up sys.path
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))
