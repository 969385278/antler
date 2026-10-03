"""一键执行当前 P0 诊断：C17 提取回答补分，再做 C6 的 C/N 低极测量。"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.eval.judge import Judge
from antler.traits.data import DIMS, KIT
from e0_extract_coherence import DEFAULT_POLES, _redacted, load_inputs

STEPS = {
    "c17": ["extract_coherence.json", "extract_coherence_samples.jsonl", "run_meta.json"],
    "c6_low": ["calibration_judge.csv", "calibration_judge_samples.jsonl", "run_meta.json"],
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    """原子更新进度；断电时保留上一次完整状态。"""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str,
                                    allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_arguments(ap):
    ap.add_argument("--fresh", action="store_true", help="另建一轮诊断；默认恢复最近一轮相同输入的诊断")


def build_commands(config):
    common = [sys.executable, "-u"]
    return {
        "c17": common + ["experiments/e0_extract_coherence.py", "--config", str(config),
                         "--poles", *DEFAULT_POLES],
        "c6_low": common + ["experiments/e0_calibrate.py", "--config", str(config),
                            "--mode", "judge", "--layers", "16", "--dims", "C", "N",
                            "--coefs", "-0.2", "0", "--pole", "low", "--n-samples", "3",
                            "--save-samples", "--reuse-baseline"],
    }


def build_plan(args, cfg):
    """调用评委前检查两步输入；只在 CPU 读取方向，不加载生成模型。"""
    if not cfg.get("use_judge", True):
        raise ValueError("一键诊断需要 use_judge: true")
    if cfg.get("calib_questions", 10) != 10:
        raise ValueError("当前一键诊断固定使用前 10 道 calib 题，请设置 calib_questions: 10")
    if cfg.get("dtype", "bfloat16") not in ("bfloat16", "float16", "float32"):
        raise ValueError("dtype 必须为 bfloat16、float16 或 float32")
    tokens = cfg.get("max_new_tokens", 200)
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
        raise ValueError("max_new_tokens 必须为正整数")
    inputs = load_inputs(cfg["out_path"], DEFAULT_POLES)
    if any(not source["samples"] for source in inputs.values()):
        raise ValueError("提取样本不能为空，请先完成 E0 提取")
    direction_path = cfg["out_path"] / "directions.pt"
    directions = torch.load(direction_path, map_location="cpu", weights_only=True)
    U, s = directions["U"], directions["s"]
    if (not isinstance(U, torch.Tensor) or U.ndim != 3 or U.shape[0] <= 16
            or U.shape[1] != 5 or not isinstance(s, torch.Tensor) or s.ndim != 1
            or len(s) != U.shape[0] or not torch.isfinite(U).all()
            or not torch.isfinite(s).all() or s[16] <= 0
            or directions.get("dims") != DIMS):
        raise ValueError("directions.pt 必须含第 16 层有效的 U、s 和 O/C/E/A/N 维度顺序")
    files = [Path(args.config).resolve(), direction_path]
    files += [Path(source["source_file"]) for source in inputs.values()]
    for pole in ("careless", "calm"):
        path = KIT / "trait_eval" / f"{pole}.json"
        data = read_json(path)
        if len(data["questions"]) < 10 or not data["eval_prompt"]:
            raise ValueError(f"{pole} 缺少前 10 道 calib 题或评分提示")
        files.append(path)
    # 代码变动会另建一轮；不受 README 修改或提交号变化影响。
    files += sorted((ROOT / "antler").rglob("*.py"))
    files += sorted((ROOT / "experiments").glob("*.py"))
    model_path = ROOT / cfg["model"]
    model_files = {}
    if model_path.is_dir():
        model_config = read_json(model_path / "config.json")
        if model_config.get("num_hidden_layers") != U.shape[0] or model_config.get("hidden_size") != U.shape[2]:
            raise ValueError("模型 config.json 与 directions.pt 的层数或隐藏维度不一致")
        files += sorted(model_path.glob("*.json"))
        weights = sorted(model_path.glob("*.safetensors")) + sorted(model_path.glob("pytorch_model*.bin"))
        if not weights or any(p.stat().st_size == 0 for p in weights):
            raise FileNotFoundError("模型目录没有权重文件，请先完成下载")
        for index in (model_path / "model.safetensors.index.json", model_path / "pytorch_model.bin.index.json"):
            if index.exists():
                for name in set(read_json(index)["weight_map"].values()):
                    shard = model_path / name
                    if not shard.is_file() or shard.stat().st_size == 0:
                        raise FileNotFoundError(f"模型分片缺失或为空：{shard}")
        # 分词器与模板也先检查，避免 C17 花费 API 后才发现本地模型未下载完整。
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        tokenizer.apply_chat_template([{"role": "user", "content": "检查"}], tokenize=False,
                                      add_generation_prompt=True)
        # 大权重只记录文件大小和修改时间，避免每次扫描数 GB 文件。
        model_files = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in weights}
    elif str(cfg["model"]).replace("\\", "/").startswith(("models/", "./", "../", "/")):
        raise FileNotFoundError(f"找不到本地模型目录：{model_path}")
    judge = Judge(cache_path=str(cfg["out_path"] / "judge_cache.jsonl"))
    provenance = {
        "config": _redacted(cfg), "commands": build_commands(Path(args.config).resolve()),
        "file_hashes": {str(p.resolve()): sha256(p) for p in files},
        "model_files": model_files,
        "judge": _redacted({k: getattr(judge, k) for k in ("model", "base_url", "max_tokens", "extra_body")}),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "scope": "C17 四极点；C6 第16层 C/N，系数 -0.2/0，低极量表，每题3次采样",
    }
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True, default=str).encode()).hexdigest()
    return {"fingerprint": fingerprint, "provenance": provenance}


@contextmanager
def output_lock(base):
    """防止两个一键入口同时覆盖输出；进程退出后系统自动释放锁。"""
    base.mkdir(parents=True, exist_ok=True)
    with (base / ".lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("已有一键诊断正在使用此输出目录，请勿重复启动") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def run_step(command, log_path):
    """逐行显示并保存日志；失败码不会被日志管道吞掉。"""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    with Path(log_path).open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()
    if code:
        raise RuntimeError(f"子步骤退出码 {code}，日志：{log_path}")


def validate_step(step, out_path, cfg):
    """退出码为零还不够：必须有本步完整的结果和完成元数据。"""
    for name in STEPS[step]:
        if not (out_path / name).is_file():
            raise RuntimeError(f"{step} 缺少结果 {name}")
    meta = read_json(out_path / "run_meta.json")
    if meta.get("status") != "completed" or not meta.get("end_time"):
        raise RuntimeError(f"{step} 元数据未记录完成")
    if step == "c17":
        summary = read_json(out_path / "extract_coherence.json")
        records = [json.loads(line) for line in (out_path / STEPS[step][1]).read_text(encoding="utf-8").splitlines()]
        expected = load_inputs(out_path, DEFAULT_POLES)
        if meta.get("task") != "C17" or set(summary["poles"]) != set(DEFAULT_POLES):
            raise RuntimeError("C17 元数据或极点集合不匹配")
        identities = [(r["pole"], r["source_index"], r["source_sha256"]) for r in records]
        wanted = [(pole, i, source["source_sha256"]) for pole, source in expected.items()
                  for i in range(len(source["samples"]))]
        if identities != wanted:
            raise RuntimeError("C17 逐条结果与原始样本不匹配")
    else:
        if any(meta.get(k) != v for k, v in {"layers": [16], "dims": ["C", "N"], "coefs": [-0.2, 0],
                                           "pole": "low", "n_samples": 3}.items()):
            raise RuntimeError("C6 元数据与本次低极测量参数不匹配")
        with (out_path / STEPS[step][0]).open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        records = [json.loads(line) for line in (out_path / STEPS[step][1]).read_text(encoding="utf-8").splitlines()]
        wanted = {(16, dim, coef, "low") for dim in ("C", "N") for coef in (-0.2, 0)}
        keys = lambda rs: {(int(r["layer"]), r["dim"], float(r["coef"]), r["pole"]) for r in rs}
        if len(rows) != 4 or len(records) != 4 or keys(rows) != wanted or keys(records) != wanted:
            raise RuntimeError("C6 结果点数或测量参数不完整")
        if any(len(r["result"]["samples"]) != 30 for r in records):
            raise RuntimeError("C6 每个测量点必须有 10 题 × 3 次采样")


def completed(step, state, run_dir):
    record = state["steps"].get(step, {})
    hashes = record.get("artifact_hashes", {})
    return (record.get("status") == "completed" and set(hashes) == set(STEPS[step])
            and all((run_dir / step / name).is_file() and sha256(run_dir / step / name) == digest
                    for name, digest in hashes.items()))


def finish_report(run_dir):
    """汇总原始测量，保留缺失分信息，不把诊断结果当成 E0 验收。"""
    summary = read_json(run_dir / "c17" / "extract_coherence.json")
    lines = ["# C17 + C6 诊断结果", "", "C17 给已有提取回答补分；C6 使用 calib 题，仅用于诊断。每题采样 3 次不是 3 个独立种子实验。", "",
             "## C17：提取回答的连贯性", "", "| 极点 | 组 | 有效/总数 | 平均连贯性 | 低于50比例 |", "|---|---|---|---|---|"]
    fmt = lambda x: "缺失" if x is None else f"{float(x):.3f}"
    for pole, groups in summary["poles"].items():
        for side in ("pos", "neg"):
            g = groups[side]
            lines.append(f"| {pole} | {side} | {g['n_valid']}/{g['n_total']} | {fmt(g['mean_coherence'])} | {fmt(g['below_threshold_fraction'])} |")
    lines += ["", "## C6：第16层低极测量", "", "| 维度 | 低极 | 系数 | 特质分 | 连贯性 |", "|---|---|---|---|---|"]
    with (run_dir / "c6_low" / "calibration_judge.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            lines.append(f"| {row['dim']} | {'careless' if row['dim'] == 'C' else 'calm'} | {row['coef']} | {row['trait'] or '缺失'} | {row['coherence'] or '缺失'} |")
    lines += ["", "逐条回答及各步骤元数据见 c17/、c6_low/；日志见 attempts/。原文件备份见 backup/。", ""]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    # 只打包本轮结果和日志，不包含旧结果、评委缓存、密钥或模型。
    with zipfile.ZipFile(run_dir / "results.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        paths = [run_dir / "summary.md", run_dir / "run_meta.json"]
        paths += [run_dir / step / name for step, names in STEPS.items() for name in names]
        paths += sorted((run_dir / "attempts").glob("*/log.txt"))
        for path in paths:
            archive.write(path, path.relative_to(run_dir))


def run_diagnostics(args, cfg):
    out = Path(cfg["out_path"])
    base = out / "diagnostics"
    with output_lock(base):
        plan = build_plan(args, cfg)
        pointer = base / "latest.json"
        latest = read_json(pointer) if pointer.exists() else {}
        run_id = latest.get("run_id", "")
        if (not args.fresh and latest.get("fingerprint") == plan["fingerprint"]
                and run_id and Path(run_id).name == run_id):
            run_dir = base / run_id
            state = read_json(run_dir / "run_meta.json")
            if state["fingerprint"] != plan["fingerprint"]:
                raise RuntimeError("恢复记录与输入不一致，请使用 --fresh 新建诊断")
        else:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
            run_dir = base / run_id
            run_dir.mkdir()
            backup = run_dir / "backup"
            backup.mkdir()
            for name in sorted({name for names in STEPS.values() for name in names}):
                if (out / name).exists():
                    shutil.copy2(out / name, backup / name)
            state = {"task": "C17+C6", "experiment": "e0_diagnostics", **plan,
                     "start_time": now(), "end_time": None, "status": "pending", "steps": {}}
            write_json(run_dir / "run_meta.json", state)
            write_json(pointer, {"run_id": run_id, "fingerprint": plan["fingerprint"]})
        print(f"[诊断目录] {run_dir}", flush=True)
        commands = build_commands(Path(args.config).resolve())
        try:
            state.update(status="running", end_time=None)
            state.pop("error_type", None)
            write_json(run_dir / "run_meta.json", state)
            for step, names in STEPS.items():
                if completed(step, state, run_dir):
                    print(f"[跳过] {step} 已完成，归档校验通过", flush=True)
                    continue
                record = state["steps"].setdefault(step, {})
                attempt = record.get("attempts", 0) + 1
                previous = run_dir / "attempts" / f"{step}_{attempt}_{uuid.uuid4().hex[:8]}" / "previous"
                previous.mkdir(parents=True)
                record.update(status="running", attempts=attempt, start_time=now(), artifact_hashes={})
                write_json(run_dir / "run_meta.json", state)
                # 移走本步旧文件，防止异常退出时把遗留结果误认作本轮成功。
                for name in names:
                    if (out / name).exists():
                        shutil.move(str(out / name), str(previous / name))
                print(f"[运行] {step}（第 {attempt} 次尝试）", flush=True)
                run_step(commands[step], previous.parent / "log.txt")
                validate_step(step, out, cfg)
                archive = run_dir / step
                archive.mkdir(exist_ok=True)
                for name in names:
                    if name == "run_meta.json":
                        write_json(archive / name, _redacted(read_json(out / name)))
                    else:
                        shutil.copy2(out / name, archive / name)
                record.update(status="completed", end_time=now(),
                              artifact_hashes={name: sha256(archive / name) for name in names})
                write_json(run_dir / "run_meta.json", state)
            state.update(status="completed", end_time=now())
            write_json(run_dir / "run_meta.json", state)
            finish_report(run_dir)
        except BaseException as exc:
            for record in state["steps"].values():
                if record.get("status") == "running":
                    record.update(status="failed", end_time=now())
            state.update(status="failed", end_time=now(), error_type=type(exc).__name__)
            write_json(run_dir / "run_meta.json", state)
            print(f"[中断] 再次执行同一命令可恢复。日志和已完成结果：{run_dir}", flush=True)
            raise
        print(f"[全部完成] 汇总：{run_dir / 'summary.md'}\n[结果包] {run_dir / 'results.zip'}", flush=True)
        return run_dir


def main():
    args = parse_args(add_arguments)
    cfg, _, _, _ = setup(args, need_judge=False, need_model=False)
    run_diagnostics(args, cfg)


if __name__ == "__main__":
    main()
