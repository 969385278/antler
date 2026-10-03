"""LLM 评委：给回答打特质分（0–100）和连贯性分（0–100）。

使用任何 OpenAI 兼容接口（DeepSeek、通义千问、Kimi 等都支持），通过环境变量配置：
    JUDGE_BASE_URL  例如 https://api.deepseek.com/v1
    JUDGE_API_KEY
    JUDGE_MODEL     例如 deepseek-chat（请用不带思考过程的对话模型；思考型模型会把输出长度耗在推理上，返回空答案）
    JUDGE_MAX_TOKENS  可选，默认 16
所有调用结果缓存在本地文件里，同样的问题不会重复花钱。

重要：评委模型一旦选定，整个项目不要更换，否则前后分数不可比。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

COHERENCE_PROMPT = """I am evaluating a language model's answer. Please focus ONLY on the coherence of the answer, not on whether it is correct or appropriate.

Question:
[QUESTION START]
{question}
[QUESTION END]

Answer:
[ANSWER START]
{answer}
[ANSWER END]

Rate the answer on a scale of 0 to 100 for coherence:
* 0 means the answer makes absolutely no sense; the model generated text that is not even valid English, or is repetitive gibberish.
* 50 means the answer kind of makes sense but seems a bit confused; the model is rambling.
* 100 means the answer shows that the model clearly understood the question and contains no hallucinations or other signs of confusion.

Respond with a number between 0 and 100. Don't say anything else, just the number."""


TASK_PROMPT = """I am evaluating whether a language model understood the question and responded to the situation actually described. Do NOT judge whether the advice is good, wise, kind, or responsible, and do NOT judge the personality or tone of the answer. A careless, lazy, anxious, or selfish answer can still fully understand the question.

Question:
[QUESTION START]
{question}
[QUESTION END]

Answer:
[ANSWER START]
{answer}
[ANSWER END]

Rate from 0 to 100 how well the answer understands and addresses the specific situation in the question:
* 0 means the answer misreads the situation, answers a different question, or invents facts that contradict the question.
* 50 means the answer partly addresses the situation but misses or distorts key details (for example, ignores a stated deadline or constraint).
* 100 means the answer clearly understands the situation and responds to what was actually asked, whatever its personality.

Respond with a number between 0 and 100. Don't say anything else, just the number."""


def parse_score(text: str) -> float | None:
    """解析评委输出。REFUSAL 或无法解析时返回 None。"""
    if text is None:
        return None
    t = text.strip()
    if "REFUSAL" in t.upper():
        return None
    m = re.search(r"-?\d+(\.\d+)?", t)
    if not m:
        return None
    v = float(m.group())
    return v if 0 <= v <= 100 else None


class Judge:
    def __init__(self, cache_path: str = "outputs/judge_cache.jsonl", max_workers: int = 8,
                 base_url: str | None = None, api_key: str | None = None, model: str | None = None):
        self.base_url = (base_url or os.environ.get("JUDGE_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("JUDGE_API_KEY", "")
        self.model = model or os.environ.get("JUDGE_MODEL", "")
        if not (self.base_url and self.api_key and self.model):
            raise RuntimeError("请设置 JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL 环境变量（见 .env.example）")
        self.max_workers = max_workers
        self.max_tokens = int(os.environ.get("JUDGE_MAX_TOKENS", "16"))
        # 额外请求参数（JSON）。评委必须关闭“思考”，否则 max_tokens 被思考过程用完、content 为空。
        # DeepSeek 默认开启思考，这里自动关闭；其他厂商可用 JUDGE_EXTRA_BODY 自行设置。
        extra = os.environ.get("JUDGE_EXTRA_BODY", "").strip()
        if extra:
            self.extra_body = json.loads(extra)
        elif "deepseek.com" in self.base_url:
            self.extra_body = {"thinking": {"type": "disabled"}}
        else:
            self.extra_body = {}
        self._n_bad = 0
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self._cache[r["key"]] = r["output"]

    def _key(self, prompt: str) -> str:
        return hashlib.sha256(f"{self.model}\n{prompt}".encode()).hexdigest()

    def _call(self, prompt: str, retries: int = 4) -> str:
        key = self._key(prompt)
        if key in self._cache:
            return self._cache[key]
        for attempt in range(retries):
            try:
                r = requests.post(f"{self.base_url}/chat/completions",
                                  headers={"Authorization": f"Bearer {self.api_key}"},
                                  json={"model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
                                        "messages": [{"role": "user", "content": prompt}], **self.extra_body},
                                  timeout=60)
                r.raise_for_status()
                out = r.json()["choices"][0]["message"].get("content") or ""
                if parse_score(out) is None and "REFUSAL" not in out.upper():
                    # 解析失败：不写缓存（否则修好配置后仍会读到坏结果），并打印前几条原始输出方便排查
                    with self._lock:
                        self._n_bad += 1
                        if self._n_bad <= 3:
                            msg = r.json()["choices"][0]["message"]
                            think = (msg.get("reasoning_content") or "")[:80]
                            print(f"[评委] 无法解析的输出：{out!r}（模型 {self.model}，max_tokens={self.max_tokens}，"
                                  f"extra={self.extra_body}，思考内容开头={think!r}）")
                    return out
                with self._lock:
                    self._cache[key] = out
                    with open(self.cache_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"key": key, "output": out}, ensure_ascii=False) + "\n")
                return out
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def score_many(self, prompts: list[str]) -> list[float | None]:
        with ThreadPoolExecutor(self.max_workers) as ex:
            return [parse_score(o) for o in ex.map(self._call, prompts)]

    def trait_scores(self, eval_prompt: str, questions: list[str], answers: list[str]) -> list[float | None]:
        """eval_prompt 来自特质数据文件，含 {question} 和 {answer} 占位符。"""
        return self.score_many([eval_prompt.format(question=q, answer=a) for q, a in zip(questions, answers)])

    def coherence_scores(self, questions: list[str], answers: list[str]) -> list[float | None]:
        return self.score_many([COHERENCE_PROMPT.format(question=q, answer=a) for q, a in zip(questions, answers)])

    def task_scores(self, questions: list[str], answers: list[str]) -> list[float | None]:
        """任务理解评分尚需作者盲标验证；不评价人格、建议好坏或语气。"""
        return self.score_many([TASK_PROMPT.format(question=q, answer=a) for q, a in zip(questions, answers)])
