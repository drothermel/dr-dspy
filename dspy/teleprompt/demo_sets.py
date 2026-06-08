import random

from dspy.dsp.utils.settings import settings
from dspy.teleprompt.bootstrap import BootstrapFewShot, LabeledFewShot


async def create_n_fewshot_demo_sets(
    student,
    num_candidate_sets,
    trainset,
    max_labeled_demos,
    max_bootstrapped_demos,
    metric,
    teacher_settings,
    max_errors=None,
    max_rounds=1,
    labeled_sample=True,
    min_num_samples=1,
    metric_threshold=None,
    teacher=None,
    include_non_bootstrapped=True,
    seed=0,
    rng=None,
):
    max_errors = settings.max_errors if max_errors is None else max_errors
    demo_candidates = {}
    num_candidate_sets -= 3
    for i, _ in enumerate(student.predictors()):
        demo_candidates[i] = []
    rng = rng or random.Random(seed)
    for seed in range(-3, num_candidate_sets):
        trainset_copy = list(trainset)
        if seed == -3 and include_non_bootstrapped:
            program2 = student.reset_copy()
        elif seed == -2 and max_labeled_demos > 0 and include_non_bootstrapped:
            teleprompter = LabeledFewShot(k=max_labeled_demos)
            program2 = await teleprompter.compile(student, trainset=trainset_copy, sample=labeled_sample)
        elif seed == -1:
            program = BootstrapFewShot(
                metric=metric,
                max_errors=max_errors,
                metric_threshold=metric_threshold,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                teacher_settings=teacher_settings,
                max_rounds=max_rounds,
            )
            program2 = await program.compile(student, teacher=teacher, trainset=trainset_copy)
        else:
            rng.shuffle(trainset_copy)
            size = rng.randint(min_num_samples, max_bootstrapped_demos)
            teleprompter = BootstrapFewShot(
                metric=metric,
                max_errors=max_errors,
                metric_threshold=metric_threshold,
                max_bootstrapped_demos=size,
                max_labeled_demos=max_labeled_demos,
                teacher_settings=teacher_settings,
                max_rounds=max_rounds,
            )
            program2 = await teleprompter.compile(student, teacher=teacher, trainset=trainset_copy)
        for i, _ in enumerate(student.predictors()):
            demo_candidates[i].append(program2.predictors()[i].demos)
    return demo_candidates
