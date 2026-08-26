#!/usr/bin/env python3
"""run_evals_wave.py — 串行分批跑 8 组 1000 局评估（默认每批 4 个，避免 16GB 内存被打满）。
上一轮 24 个进程并发（三个重复 driver）触发了 OOM，全部被杀，故限并发重跑。
用法: python3 run_evals_wave.py [concurrency]
"""
import os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.abspath(os.path.join(ROOT, "..", "rlvenv", "bin", "python"))
CODE = {"v0_repro": "code_v0", "v3a": "code_v3a", "v3b": "code_v3b", "v3ab": "code_v3ab"}
JOBS = [(g, s) for g in ["v0_repro", "v3a", "v3b", "v3ab"] for s in [7, 123]]
CONC = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def ts():
    return time.strftime("%H:%M:%S")


def launch(g, s):
    out_json = os.path.join(ROOT, "results", f"eval_{g}_s{s}.json")
    log = open(os.path.join(ROOT, "logs", f"eval_{g}_s{s}.log"), "w")
    cmd = [PY, os.path.join(ROOT, "scripts", "eval_balance2.py"),
           os.path.join(ROOT, CODE[g]), os.path.join(ROOT, "models", f"model_{g}_s{s}.zip"),
           out_json, "1000"]
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                         env=dict(os.environ, OMP_NUM_THREADS="6"))
    return (f"{g}_s{s}", p, log, out_json)


done, fail = [], []
for i in range(0, len(JOBS), CONC):
    wave = JOBS[i:i + CONC]
    print(f"[{ts()}] 第 {i // CONC + 1} 批启动: {[f'{g}_s{s}' for g, s in wave]}", flush=True)
    running = [launch(g, s) for g, s in wave]
    for name, p, log, out_json in running:
        rc = p.wait()
        log.close()
        ok = rc == 0 and os.path.exists(out_json) and os.path.getsize(out_json) > 100
        print(f"[{ts()}] {name} rc={rc} ok={ok}", flush=True)
        (done if ok else fail).append(name)
    print(f"[{ts()}] 累计完成 {len(done)}/{len(JOBS)}，失败 {len(fail)}", flush=True)

print(f"[{ts()}] 全部结束。成功 {len(done)}，失败 {fail}", flush=True)
print("[ALL_DONE]", flush=True)
