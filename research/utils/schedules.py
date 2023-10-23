"""
This file contains schedule functions that can be used as learning rate schedules

All learning rate schedulers use the pytorch LambdaLR function and any additional kwargs.
"""


def linear_decay(total_steps: int, start_step: int = 1, offset: int = 0):
    def fn(step):
        return 1.0 - max(0, step + offset - start_step) / (total_steps - start_step)

    return fn


def linear_warmup(total_steps: int, multiplier: float = 1.0):
    def fn(step):
        return multiplier * min(1.0, step / total_steps)

    return fn
