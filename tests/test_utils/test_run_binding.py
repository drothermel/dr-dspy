from pathlib import Path

from tests.test_utils.run_binding import find_run_binding_violations


def test_allows_run_fixture_parameter():
    source = """
def test_foo(run):
    asyncio.run(predict(question="x", run=run))
"""
    assert find_run_binding_violations(source) == []


def test_allows_local_run_assignment():
    source = """
def test_foo(make_run):
    lm = DummyLM([{}])
    run = make_run(lm=lm)
    asyncio.run(predict(question="x", run=run))
"""
    assert find_run_binding_violations(source) == []


def test_allows_tuple_unpacking_assignment():
    source = """
def test_foo(make_run):
    predictor, lm, run = setup_predictor("q -> a", {"a": "x"}, make_run)
    asyncio.run(predictor(question="x", run=run))
"""
    assert find_run_binding_violations(source) == []


def test_flags_unbound_run_kwarg():
    source = """
def test_foo(make_run):
    asyncio.run(predict(question="x", run=run))
"""
    violations = find_run_binding_violations(source, path=Path("bad_test.py"))
    assert len(violations) == 1
    assert violations[0].func_name == "test_foo"
    assert violations[0].line == 3
