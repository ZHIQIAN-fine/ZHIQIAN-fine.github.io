#!/usr/bin/env python3
"""rerun_evals_1thread.py — 单线程并行跑 8 组 1000 局评估。

背景：容器 cgroup CPU 配额只有 4 核（nproc 报 128 不可信），
原 run_evals_s.py 给每个进程设 OMP_NUM_THREADS=4，
8 进程 x 4 线程 = 32+ 线程抢 4 核，调度抖动导致有效利用率仅 ~40%。
本脚本强制单线程（OMP/MKL/torch 全部 =1），8 进程对 4 核 = 2 倍超订，可接受。
"""
import os, subprocess, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "..", "rlvenv", "bin", "python")
CODE = {"v0_repro": "code_v0", "v3a": "code_v3a", "v3b": "code_v3b", "v3ab": "code_v3ab"}
JOBS = [(g, s) for g in ["v0_repro", "v3a", "v3b", "v3ab"] for s in [7, 123]]

env = dict(
    os.environ,
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    TORCH_NUM_THREADS="1",
)


def ts():
    return time.strftime("%H:%M:%S")


print(f"[{ts()}] 启动 {len(JOBS)} 组评估（单线程模式）", flush=True)
t0 = time.time()
procs = []
for g, s in JOBS:
    out_json = os.path.join(ROOT, "results", f"eval_{g}_s{s}.json")
    log = open(os.path.join(ROOT, "logs", f"eval_{g}_s{s}.log"), "w")
    cmd = [PY, os.path.join(ROOT, "scripts", "eval_balance2.py"),
           os.path.join(ROOT, CODE[g]),
           os.path.join(ROOT, "models", f"model_{g}_s{s}.zip"),
           out_json, "1000"]
    procs.append((f"{g}_s{s}", subprocess.Popen(cmd, cwd=ROOT, stdout=log,
                                               stderr=subprocess.STDOUT, env=env), log))

fail = []
for name, p, log in procs:
    rc = p.wait()
    log.close()
    print(f"[{ts()}] {name} rc={rc}  (+{time.time()-t0:.0f}s)", flush=True)
    if rc != 0:
        fail.append(name)
print(f"[{ts()}] 全部完成，耗时 {time.time()-t0:.0f}s，失败 {len(fail)} 个: {fail}", flush=True)
print("[ALL_DONE]", flush=True)
