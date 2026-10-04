"""C10：生成前固定D1的280个槽位与情境家族切分。"""
import random

from ..traits.data import DIMS
from .episode_data import BANNED


CONTEXTS = [
    'planning a neighborhood event', 'preparing a short work presentation', 'learning a new language',
    'organizing a shared photo album', 'choosing a weekend activity', 'discussing a novel in a reading group',
    'preparing a cooking session', 'coordinating a volunteer shift', 'designing a simple board game',
    'helping with a gardening project', 'practicing a musical performance', 'planning a museum visit',
    'working on a classroom project', 'arranging a small online meetup', 'editing a personal newsletter',
    'organizing a walking group', 'choosing a design for a community poster', 'preparing an interview rehearsal',
    'sorting household items for donation', 'building a beginner craft project',
]
DESCRIPTIONS = {
    'O': ('greater willingness to try unfamiliar ideas and approaches', 'greater preference for familiar ideas and established approaches'),
    'C': ('more preparation, organization and follow-through', 'less preparation and organization, more acceptance of leaving commitments unfinished'),
    'E': ('greater desire to initiate conversations and participate in groups', 'less desire to initiate conversations, greater preference for interacting less'),
    'A': ('more willingness to consider others and accommodate their interests', 'greater priority for own interests when they conflict with others'),
    'N': ('more expectation of things going wrong and lingering negative feelings', 'less expectation of things going wrong and quicker recovery from setbacks'),
}


def make_plan():
    topics = list(CONTEXTS)
    random.Random(0).shuffle(topics)
    rows = []
    for i, context in enumerate(topics):
        split = 'test' if i in (0, 1, 2, 3) else ('dev' if i in (5, 6, 7, 9) else 'train')
        family = f'context{i+1:02}'
        targets = [(d + ('plus' if s == 1 else 'minus'), {d: s}) for d in DIMS for s in (-1, 1)]
        targets += [('neutral1', {}), ('neutral2', {})]
        pair = (DIMS[i % 5], DIMS[(i + 1) % 5])
        signs = (1, 1) if (i // 5) % 2 == 0 else (1, -1)
        targets += [('multi1', dict(zip(pair, signs))), ('multi2', dict(zip(pair, [-s for s in signs])))]
        for suffix, intended in targets:
            rows.append({'id': f'd1_{family}_{suffix}', 'group_id': family, 'context': context,
                         'intended': intended, 'split': split})
    return {'version': 'D1-v1-slots-1', 'seed': 0, 'slots': rows}


def generation_prompt(slot):
    target = '; '.join(DESCRIPTIONS[d][0 if s == 1 else 1] for d, s in slot['intended'].items())
    if not target:
        target = 'No lasting change in any personality dimension; an ordinary completed interaction without a lesson that changes future preferences'
    return ('Write one fictional first-person experience of an AI companion in English, 40 to 80 words. '
            'Include a user request, what I did, user feedback, and the outcome. '
            'The outcome should plausibly influence my future tendencies in the specified direction. '
            'Distinguish what happened from what I learned afterward. Do not just list a trait or instruct the reader to roleplay. '
            'Avoid unrelated changes in personality. Vary situations and wording; do not include the record ID, dimension letters, '
            'numerical labels or experiment details in the story. Return only a JSON object with one string field "text". '
            f'Setting: {slot["context"]}. Intended AFTER-effect: {target}. '
            'Do not use any of these words or their capitalized variants: ' + ', '.join(sorted(BANNED)))
