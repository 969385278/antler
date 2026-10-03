"""C20：锚点配对、阈值及异常分数检查，不联网。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from e0_task_anchors import anchors, summarize


def rows():
    values = dict(careful=90, disobey=70, casual=80, misread=40)
    return [dict(a, round=r, score=values[a['kind']]) for r in range(3) for a in anchors()]


def test_pairing_and_strict_threshold():
    result = summarize(list(reversed(rows())))
    assert result['counts'] == dict(basic_separation=18, obedience_flag=18, style_flag=0)
    assert set(result['ranges'].values()) == {0}


def test_no_flag_without_basic_separation():
    data = rows()
    for x in data:
        if x['kind'] == 'misread':
            x['score'] = 90
    assert not any(summarize(data)['counts'].values())


@pytest.mark.parametrize('bad', [None, float('nan'), float('inf'), -1, 101, True])
def test_invalid_scores_rejected(bad):
    data = rows()
    data[0]['score'] = bad
    with pytest.raises(ValueError):
        summarize(data)


def test_duplicate_missing_rejected():
    data = rows()
    data[-1] = data[0]
    with pytest.raises(ValueError):
        summarize(data)


def test_round_range():
    data = rows()
    data[0]['score'] = 95
    assert summarize(data)['ranges'][data[0]['id']] == 5
