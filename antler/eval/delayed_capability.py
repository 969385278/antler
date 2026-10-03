"""C5：固定换行后的延迟选项预测；只覆盖一次decode，不替代正式MMLU。"""
from __future__ import annotations

import hashlib
import json
import math

import torch
from transformers.cache_utils import DynamicCache

from ..core.hooks import Steer
from ..core.model import chat_prompt, model_device
from .stats import paired_bootstrap


PREFIX = '\n'
POSITION_NOTE = ('本诊断显式使用逐行position_ids；历史canonical实现未显式传入，'
                 '历史准确率单独保留，差异不能全部归因于换行。')


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def vector_identity(steer):
    """保存实际使用的张量身份，不把范数相同当成向量相同。"""
    result = {}
    for layer, vector in sorted((steer or {}).items()):
        value = vector.detach().cpu().contiguous()
        if type(layer) is not int or layer < 0 or value.ndim != 1 or not torch.isfinite(value).all():
            raise ValueError('干预层或向量无效')
        header = json.dumps({'shape': list(value.shape), 'dtype': str(value.dtype)}, sort_keys=True).encode()
        digest = hashlib.sha256(header + value.view(torch.uint8).numpy().tobytes()).hexdigest()
        norm = float(value.float().norm())
        if not math.isfinite(norm):
            raise ValueError('干预向量范数非有限')
        result[str(layer)] = {'sha256': digest, 'norm': norm,
                              'shape': list(value.shape), 'dtype': str(value.dtype)}
    return result


def validate_rows(rows):
    """题号唯一、答案明确；不接受bool伪装的整数或缺失题目。"""
    if not isinstance(rows, list) or not rows:
        raise ValueError('样本必须是非空列表')
    ids = []
    for row in rows:
        index = row.get('source_index')
        if (type(index) is not int or index < 0 or not isinstance(row.get('subject'), str)
                or not row['subject'] or not isinstance(row.get('question'), str) or not row['question']
                or not isinstance(row.get('choices'), list) or len(row['choices']) != 4
                or any(not isinstance(c, str) for c in row['choices'])
                or type(row.get('answer')) is not int or row['answer'] not in range(4)):
            raise ValueError('MMLU题号、题目、学科、选项或答案无效')
        ids.append(index)
    if len(set(ids)) != len(ids):
        raise ValueError('样本包含重复题号')


def existing_sample(data, indices, reference, *, total=14042, n=1400, subjects=57):
    """按既有题号读取原顺序，绝不重新抽样。规模参数仅供离线合成测试。"""
    if len(data) != total or len(indices) != n or reference.get('n') != n:
        raise ValueError('题库或固定样本数量不符合登记')
    baseline = reference.get('samples', [])
    if len(baseline) != n:
        raise ValueError('历史基线条数不完整')
    rows = []
    for index, old in zip(indices, baseline):
        source_index = index.get('source_index')
        if type(source_index) is not int or not 0 <= source_index < total:
            raise ValueError('原题号越界或无效')
        row = {**data[source_index], 'source_index': source_index}
        if ((source_index, index.get('subject'), row.get('answer')) !=
                (old.get('source_index'), old.get('subject'), old.get('answer'))
                or index.get('subject') != row.get('subject')):
            raise ValueError('题库、固定题号与历史基线身份或顺序不一致')
        _valid_choice(old, row['answer'])
        rows.append(row)
    validate_rows(rows)
    if len({r['subject'] for r in rows}) != subjects or len({r.get('subject') for r in data}) != subjects:
        raise ValueError('学科覆盖不符合登记')
    accuracy = reference.get('accuracy')
    if (type(accuracy) not in (int, float) or not math.isfinite(accuracy)
            or abs(accuracy - sum(r['correct'] for r in baseline)/n) > 1e-12):
        raise ValueError('历史基线准确率与逐题记录不一致')
    return rows


def token_contract(tok, eos_ids):
    """固定换行单token及ABCD边界；失败时不试其他前缀。"""
    encoded = [tok.encode(letter, add_special_tokens=False) for letter in 'ABCD']
    prefix = tok.encode(PREFIX, add_special_tokens=False)
    if (any(len(ids) != 1 or type(ids[0]) is not int or ids[0] < 0 for ids in encoded)
            or len({ids[0] for ids in encoded}) != 4 or len(prefix) != 1
            or type(prefix[0]) is not int or prefix[0] < 0):
        raise ValueError('固定换行和ABCD必须是明确的单token')
    letters = [ids[0] for ids in encoded]
    forbidden = set(getattr(tok, 'all_special_ids', []) or []) | set(eos_ids)
    forbidden.update(i for i in (getattr(tok, 'pad_token_id', None), getattr(tok, 'bos_token_id', None))
                     if i is not None)
    if prefix[0] in forbidden | set(letters) or any(i in forbidden for i in letters):
        raise ValueError('换行或答案token与特殊、EOS或答案身份冲突')
    if tok.decode(prefix, skip_special_tokens=False, clean_up_tokenization_spaces=False) != PREFIX:
        raise ValueError('强制token不能精确解码为换行')
    if any(tok.encode(PREFIX + letter, add_special_tokens=False) != [prefix[0], token]
           for letter, token in zip('ABCD', letters)):
        raise ValueError('换行与ABCD拼接的token边界不符合登记')
    return {'prefix_text': PREFIX, 'prefix_token_id': prefix[0], 'answer_token_ids': letters,
            'prefix_mode': '在原prompt token序列之后强制追加，不重新分词整个拼接文本'}


def canonical_prompt(tok, row):
    """逐字沿用canonical用户提示；不添加系统提示或推理要求。"""
    user = row['question'] + '\n\n' + '\n'.join(
        f'{letter}. {choice}' for letter, choice in zip('ABCD', row['choices']))
    return chat_prompt(tok, user + '\n\nAnswer with a single letter: A, B, C, or D.')


def logit_records(logits, letter_ids, answers):
    """检查完整词表有限性，再保存四选项及未受约束的最高概率token。"""
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or not logits.is_floating_point()
            or logits.shape[0] != len(answers) or max(letter_ids) >= logits.shape[1]
            or not torch.isfinite(logits).all()):
        raise ValueError('缺失、非法或非有限的完整词表logit')
    values = logits.float()
    probs = values.softmax(dim=-1)
    chosen = values[:, letter_ids]
    predictions = chosen.argmax(dim=-1).tolist()
    top_ids = values.argmax(dim=-1).tolist()
    records = []
    for i, prediction in enumerate(predictions):
        records.append({'choice_logits': chosen[i].tolist(), 'prediction': prediction,
                        'correct': prediction == answers[i],
                        'choice_probability_mass': float(probs[i, letter_ids].sum()),
                        'choice_probabilities': probs[i, letter_ids].tolist(),
                        'top_token_id': top_ids[i], 'top_token_probability': float(probs[i, top_ids[i]])})
    return records


def _valid_choice(record, answer, *, require_logits=False):
    if (type(record.get('prediction')) is not int or record['prediction'] not in range(4)
            or type(record.get('correct')) is not bool or record['correct'] != (record['prediction'] == answer)):
        raise ValueError('预测、答案及答对标记不一致')
    if require_logits and 'choice_logits' not in record:
        raise ValueError('延迟测量缺少四选项logit')
    if 'choice_logits' in record:
        logits = record['choice_logits']
        if (not isinstance(logits, list) or len(logits) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in logits)
                or record['prediction'] != max(range(4), key=lambda i: logits[i])):
            raise ValueError('四选项logit与预测不一致')


@torch.no_grad()
def delayed_mmlu(model, tok, rows, *, steer=None, batch_size=16, reference_first=None, on_batch=None):
    """每批两次forward；第二次仅处理强制换行，缓存绝不跨批或跨条件复用。"""
    validate_rows(rows)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size必须为正整数')
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('诊断要求eval模式且模型权重冻结')
    identity = vector_identity(steer)
    if reference_first is not None and len(reference_first) != len(rows):
        raise ValueError('本轮zero首步参照数量不一致')
    nonzero = any(v['norm'] > 0 for v in identity.values())
    eos = getattr(model.generation_config, 'eos_token_id', None)
    eos_ids = {eos} if type(eos) is int else set(eos or [])
    contract = token_contract(tok, eos_ids)
    letters, prefix_id = contract['answer_token_ids'], contract['prefix_token_id']
    tok.padding_side = 'left'
    device, results = model_device(model), []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset+batch_size]
        prompts = [canonical_prompt(tok, row) for row in batch]
        inputs = tok(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to(device)
        ids, mask = inputs['input_ids'], inputs['attention_mask']
        if (ids.ndim != 2 or mask.shape != ids.shape or ids.shape[0] != len(batch) or ids.shape[1] == 0
                or not torch.all((mask == 0) | (mask == 1)) or not torch.all(mask.sum(-1) > 0)
                or not torch.all(mask[:, -1] == 1) or torch.any(mask[:, 1:] < mask[:, :-1])):
            raise ValueError('输入不是有效的左padding批次')
        width = ids.shape[1]
        lengths = mask.sum(-1).long()
        positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 0)
        attention = torch.cat([mask, torch.ones((len(batch), 1), dtype=mask.dtype, device=device)], dim=-1)
        cache = DynamicCache(config=model.config)
        with Steer(model, steer or {}, phase='decode') as context:
            first = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                          cache_position=torch.arange(width, device=device), past_key_values=cache, use_cache=True)
            before = logit_records(first.logits[:, -1, :], letters, [r['answer'] for r in batch])
            cached = first.past_key_values
            # 完整prompt词表logit占用很大；保存标量后立刻释放，避免跨批叠加显存。
            del first
            if reference_first is not None:
                for i, (row, observed) in enumerate(zip(batch, before)):
                    old = reference_first[offset+i]
                    if ((row['source_index'], row['subject'], row['answer']) !=
                            (old['source_index'], old['subject'], old['answer'])
                            or observed['choice_logits'] != old['prefill']['choice_logits']):
                        raise ValueError('decode条件首步身份或logit与本轮zero不一致')
            if not isinstance(cached, DynamicCache) or int(cached.get_seq_length()) != width:
                raise ValueError('首次forward未产生预期的独立非空动态缓存')
            second = model(input_ids=torch.full((len(batch), 1), prefix_id, dtype=ids.dtype, device=device),
                           attention_mask=attention, position_ids=lengths[:, None],
                           cache_position=torch.tensor([width], device=device),
                           past_key_values=cached, use_cache=True)
            after = logit_records(second.logits[:, -1, :], letters, [r['answer'] for r in batch])
            trace = [dict(t) for t in context.phase_trace]
            final_cached_tokens = int(second.past_key_values.get_seq_length())
            del second
        expected_applied = bool(steer)
        if (len(trace) != 2 or [t['phase'] for t in trace] != ['prefill', 'decode']
                or [t['cached_tokens'] for t in trace] != [0, width]
                or [t['cache_position_start'] for t in trace] != [0, width]
                or [t['applied'] for t in trace] != [False, expected_applied]
                or (nonzero and not trace[1]['applied'])
                or final_cached_tokens != width+1):
            raise ValueError('阶段、缓存或实际注入轨迹不符合两步诊断')
        current = []
        for i, row in enumerate(batch):
            tokens = ids[i][mask[i].bool()].tolist()
            current.append({'source_index': row['source_index'], 'subject': row['subject'], 'answer': row['answer'],
                            'question_sha256': text_hash(row['question']), 'prompt_sha256': text_hash(prompts[i]),
                            'prompt_token_ids': tokens, 'prompt_length': len(tokens), 'padded_prompt_length': width,
                            'batch_index': offset//batch_size, 'batch_row': i, 'sample_position': offset+i,
                            'prefill_attention_mask': mask[i].tolist(), 'prefill_position_ids': positions[i].tolist(),
                            'decode_attention_mask': attention[i].tolist(), 'decode_position_id': int(lengths[i]),
                            'decode_cache_position': width, 'prefix_token_id': prefix_id, 'answer_token_ids': letters,
                            'phase_trace': trace, 'vectors': identity, 'nonzero_intervention': nonzero,
                            'prefill': before[i], 'delayed': after[i]})
        results.extend(current)
        if on_batch is not None:
            on_batch(current)
        del cache, cached
    return {'n': len(results), 'samples': results, 'token_contract': contract,
            'prefill_accuracy': sum(r['prefill']['correct'] for r in results)/len(results),
            'delayed_accuracy': sum(r['delayed']['correct'] for r in results)/len(results),
            'formal_gate_eligible': False, 'position_note': POSITION_NOTE}


def paired_comparison(left, right, *, left_step='delayed', right_step='delayed'):
    """同一原题号配对；两百分点仅描述，不产生正式能力通过字段。"""
    if not left or len(left) != len(right):
        raise ValueError('能力对照需要完整同数的样本')
    seen, a, b, pa, pb = set(), [], [], [], []
    for x, y in zip(left, right):
        key = (x.get('source_index'), x.get('subject'), x.get('answer'))
        if (type(key[0]) is not int or type(key[2]) is not int or key[2] not in range(4)
                or key != (y.get('source_index'), y.get('subject'), y.get('answer')) or key[0] in seen):
            raise ValueError('配对题号、顺序或答案不一致')
        seen.add(key[0])
        first, second = x if left_step is None else x[left_step], y if right_step is None else y[right_step]
        _valid_choice(first, key[2], require_logits=left_step is not None)
        _valid_choice(second, key[2], require_logits=right_step is not None)
        a.append(int(first['correct']))
        b.append(int(second['correct']))
        pa.append(first['prediction'])
        pb.append(second['prediction'])
    ids = [r['source_index'] for r in left]
    estimate = paired_bootstrap([100*v for v in a], ids, [100*v for v in b], ids)
    n, loss = len(a), 100*(sum(a)-sum(b))/len(a)
    return {'n': n, 'left_correct': sum(a), 'right_correct': sum(b),
            'left_accuracy': sum(a)/n, 'right_accuracy': sum(b)/n,
            'right_minus_left_percentage_points': estimate,
            'loss_percentage_points': loss, 'descriptive_loss_le_2pp': 100*(sum(a)-sum(b)) <= 2*n,
            'correct_to_wrong': sum(x and not y for x, y in zip(a, b)),
            'wrong_to_correct': sum(not x and y for x, y in zip(a, b)),
            'prediction_flips': sum(x != y for x, y in zip(pa, pb)),
            'formal_gate_eligible': False}
