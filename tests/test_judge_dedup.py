"""C3：重复评委输入只调用一次并保持结果顺序与缺失值。"""
from collections import Counter
import threading

from antler.eval.judge import Judge


def test_duplicate_prompts_share_one_call_and_score():
    judge = object.__new__(Judge)
    judge.max_workers = 8
    calls = Counter()
    lock = threading.Lock()

    def call(prompt):
        with lock:
            calls[prompt] += 1
            count = calls[prompt]
        return str(50 + count) if prompt == '重复题' else 'REFUSAL'

    judge._call = call
    assert judge.score_many(['重复题', '拒绝题', '重复题', '重复题']) == [51., None, 51., 51.]
    assert calls == Counter({'重复题': 1, '拒绝题': 1})
    assert judge.score_many([]) == []
