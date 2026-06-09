from dspy.teleprompt.compilation import CompileResult, CompileStats, ProgramCandidate


class _DummyProgram:
    def __init__(self) -> None:
        self._compiled = False


def test_compile_result_with_compiled_program_sets_flag():
    program = _DummyProgram()
    assert program._compiled is False
    result = CompileResult.with_compiled_program(program)
    assert result.program._compiled is True
    assert result.candidates == []
    assert result.stats.metric_calls == 0


def test_program_candidate_accepts_optional_fields():
    program = _DummyProgram()
    candidate = ProgramCandidate(score=0.9, program=program, label="p -> w", subscores=[1.0, 0.8], full_eval=True)
    assert candidate.score == 0.9
    assert candidate.label == "p -> w"


def test_compile_stats_defaults():
    stats = CompileStats()
    assert stats.metric_calls == 0
    assert stats.error_occurred is False
    assert stats.trial_logs == {}
