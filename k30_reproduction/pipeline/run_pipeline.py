#!/usr/bin/env python3
"""SASAFL 논문(K=30) 재현 실험 전체를 순서대로 실행하고, 실험마다 논문 형식으로 정리한다.

- 재개 가능: 이미 완료된 실행(result.json outcome=completed, 요청 라운드 도달)은 건너뛴다.
- 실행 하나가 끝날 때마다 해당 실험의 그림/표/SUMMARY.md를 다시 만든다.
- 실패한 실행은 1회 재시도한 뒤 기록을 남기고 다음으로 넘어간다.
- 진행 기록: k30_reproduction/PROGRESS.md

사용법:
  python run_pipeline.py                 # 전체 (E1 → E6)
  python run_pipeline.py --only E2 E3    # 일부
  python run_pipeline.py --report-only   # 실행 없이 정리만 다시
"""

import argparse
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
K30 = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(K30))

import analysis as A  # noqa: E402
import paper_figures as PF  # noqa: E402
import paper_values as P  # noqa: E402
from run_k30 import Run  # noqa: E402

EXP_ROOT = K30 / "experiments"
PAPER_FIGS = K30 / "paper_figures"
PROGRESS = K30 / "PROGRESS.md"

BASE = {"mode": "single", "cats": "1,2,3,4,8", "K": 30, "batch": 32, "epochs": 5, "target": 1.0, "tau": 9}
SETUP_TEXT = ("K=30, wise-175 단일 노드, CPU 카테고리 1·2·3·4·8코어 × 6명(논문 K=30 구성), "
              "batch 32, local epoch 5, τ=9, ζ=0.9, non-IID(gaussian), 설정당 반복 10회 목표(논문: average of ten repetitions)")
RUN_MAX_WALL_MIN = 1500   # 실행 하나의 시간 상한 (MNIST SASAFL C=0.5가 561분 걸려 600분 상한에 걸린 적 있음)
MAX_ROUNDS = 500   # 500라운드를 넘는 실험(CIFAR-10, 논문 3000)은 500으로 고정 (2026-10-10 결정)
# 원저자 마지막 커밋(2023-12-14)의 SASAFL.py: last_global_loss를 quota 계산 직전에 덮어써서 quota가 C·K로 고정된다
ERR_SASAFL_SRC = K30 / "original_code" / "SASAFL_c0b819b.py"
REPS = 10   # 논문: "All the experiments were performed with an average of ten repetitions per setting."


def rep_name(name, r):
    """반복 r회차 실행 이름. 1회차는 기존 이름 그대로 (LT_tau*_rep1은 _repN으로 바꾼다)."""
    if r == 1:
        return name
    if name.endswith("_rep1"):
        return f"{name[:-5]}_rep{r}"
    return f"{name}_rep{r}"


def add_reps(specs, skip=lambda spec: False):
    """1회차 실행 정의 → 1~REPS회차 정의 (회차 순서). spec['setting'] = 1회차 이름 (같은 설정끼리 묶는 키)."""
    out = [{**sp, "rep": 1, "setting": sp["name"]} for sp in specs]
    for r in range(2, REPS + 1):
        out += [{**sp, "name": rep_name(sp["name"], r), "rep": r, "setting": sp["name"]} for sp in specs if not skip(sp)]
    return out


METHOD_TAG = {"SemiAsync": "BSA", "SASAFL": "SASAFL", "FedAvg": "FedAvg", "FedAsync": "FedAsync"}


def table2_runs(prefix, dataset, lr, rounds, max_wall_min):
    runs = []
    for method in ("SemiAsync", "SASAFL"):
        for c in (0.1, 0.3, 0.5):
            runs.append({**BASE, "name": f"{prefix}_{METHOD_TAG[method]}_C{c}" + ("_single_cats12348" if prefix == "T2_MNIST" else ""),
                         "method": method, "C": c, "dataset": dataset, "lr": lr, "rounds": rounds,
                         "max_wall_min": max_wall_min})
    return runs


EXPERIMENTS = [
    {"id": "E1", "dir": "E1_MNIST_table2", "title": "MNIST — Table 2(MNIST 열), Fig. 7(a), Table 4(MNIST)",
     "runs_root": K30 / "runs",   # 2026-10-10 시작한 Table 2 실행을 그대로 사용
     "runs": add_reps(table2_runs("T2_MNIST", "MNIST", 0.01, 500, RUN_MAX_WALL_MIN))},
    {"id": "E2", "dir": "E2_FashionMNIST_table2",
     "title": "FashionMNIST — Table 2(FashionMNIST 열), Fig. 7(c), Fig. 11(K=30), Table 4(FashionMNIST)",
     "runs": add_reps(table2_runs("FMNIST", "FashionMNIST", 0.001, 500, RUN_MAX_WALL_MIN))},
    {"id": "E4", "dir": "E4_CIFAR10_table2",
     "title": "CIFAR-10 — Table 2(CIFAR-10 열), Fig. 8(a), Fig. 9, Table 4(CIFAR-10)",
     "runs": add_reps(table2_runs("CIFAR10", "CIFAR10", 0.001, MAX_ROUNDS, RUN_MAX_WALL_MIN))},
    {"id": "E5", "dir": "E5_Fig5_lag_tolerance", "title": "Lag tolerance τ=1–10 — Fig. 5 (MNIST, SASAFL, C=0.3, 설정당 10회)",
     # τ=9는 E1의 SASAFL C=0.3과 설정이 같아 그 반복을 재사용한다 (1회차는 심볼릭 링크, 2회차부터는 E1 실행을 직접 읽음)
     "runs": add_reps([{**BASE, "name": f"LT_tau{t}_rep1", "method": "SASAFL", "C": 0.3, "dataset": "MNIST", "lr": 0.01,
                        "rounds": 500, "tau": t, "max_wall_min": RUN_MAX_WALL_MIN} for t in range(1, 11)],
                      skip=lambda sp: sp["tau"] == 9)},
    {"id": "E7", "dir": "E7_adaptive_off_2023code",
     "title": "적응형 quota가 꺼진 원저자 코드(c0b819b) — Fig. 7(a)(c), Fig. 9 재실행 (SASAFL만, BSA는 E1·E2·E4 재사용)",
     "runs": add_reps([{**BASE, "name": f"ERR_{tag}_SASAFL_C{c}", "method": "SASAFL", "C": c, "dataset": ds, "lr": lr,
                        "rounds": MAX_ROUNDS, "max_wall_min": RUN_MAX_WALL_MIN, "sasafl_src": str(ERR_SASAFL_SRC)}
                       for tag, ds, lr in (("MNIST", "MNIST", 0.01), ("FMNIST", "FashionMNIST", 0.001), ("CIFAR10", "CIFAR10", 0.001))
                       for c in (0.1, 0.3, 0.5)])},
    {"id": "E8", "dir": "E8_K10",
     "title": "K=10 — 논문의 K=100 칸 대신 (Fig. 7(b)(d), Fig. 8(b), Fig. 11, Table 4 K=10 행). "
              "BSA · SASAFL(현재 코드) · SASAFL(원저자 코드, quota 고정)",
     "runs": add_reps([{**BASE, "K": 10, "name": f"K10_{tag}_{variant}_C{c}", "method": method, "C": c, "dataset": ds,
                        "lr": lr, "rounds": MAX_ROUNDS, "max_wall_min": RUN_MAX_WALL_MIN, "variant": variant,
                        **({"sasafl_src": str(ERR_SASAFL_SRC)} if variant == "SASAFLorig" else {})}
                       for tag, ds, lr in (("MNIST", "MNIST", 0.01), ("FMNIST", "FashionMNIST", 0.001), ("CIFAR10", "CIFAR10", 0.001))
                       for c in (0.1, 0.3, 0.5)
                       for variant, method in (("BSA", "SemiAsync"), ("SASAFL", "SASAFL"), ("SASAFLorig", "SASAFL"))])},
    {"id": "E6", "dir": "E6_final_summary", "title": "최종 종합 — Table 2(3개 데이터셋), Table 4(K=30), 결론 수치 점검",
     "runs": []},
]


# ------------------------------------------------------------------ 공통
QUIET = False   # --quiet: PROGRESS.md에 쓰지 않는다 (주기적으로 정리만 다시 할 때)


def log_progress(msg):
    line = f"- {datetime.now().strftime('%Y-%m-%d %H:%M')} {msg}"
    print(line, flush=True)
    if QUIET:
        return
    if not PROGRESS.exists():
        PROGRESS.write_text("# K=30 재현 파이프라인 진행 기록\n\n")
    with PROGRESS.open("a") as f:
        f.write(line + "\n")


def runs_root(exp):
    return exp.get("runs_root", EXP_ROOT / exp["dir"] / "runs")


def load_exp_runs(exp):
    root = runs_root(exp)
    return {spec["name"]: A.load_run(root / spec["name"]) for spec in exp["runs"]}


def grouped(exp, loaded, key):
    """실행들을 key(spec)별로 모아 반복 평균 가상 실행으로 만든다 (완료된 반복만)."""
    groups = {}
    for spec in exp["runs"]:
        groups.setdefault(key(spec), []).append(loaded.get(spec["name"]))
    return {k: A.combine_runs(v) for k, v in groups.items()}


def by_method_c(exp, loaded):
    return grouped(exp, loaded, lambda sp: (METHOD_TAG[sp["method"]], sp["C"]))


def status_table(exp, loaded):
    """설정별 반복 진행 상황 (완료 n/10, 최고 정확도 평균 ± 표준편차)."""
    lines = ["| 설정 (1회차 이름) | 방법 | C | 완료 반복 | 최고 정확도 평균 ± 표준편차 | 소요 평균(분) |",
             "|---|---|---:|---:|---:|---:|"]
    settings = {}
    for spec in exp["runs"]:
        settings.setdefault(spec["setting"], []).append(spec)
    for setting, specs in settings.items():
        sp = specs[0]
        runs = [loaded.get(s["name"]) for s in specs]
        done = [r for r in runs if r is not None and r["complete"]]
        target = len(specs) if not (exp["id"] == "E5" and sp.get("tau") == 9) else REPS
        accs = [A.metrics(r, sp["dataset"])[0] for r in done]
        accs = [a for a in accs if a is not None]
        walls = [r["result"].get("wall_min") for r in done if r["result"].get("wall_min")]
        acc_s = f"{np.mean(accs):.2f} ± {np.std(accs):.2f}" if accs else "-"
        wall_s = f"{np.mean(walls):.0f}" if walls else "-"
        tau = f" (τ={sp['tau']})" if exp["id"] == "E5" else ""
        note = " (τ=9 반복은 E1 SASAFL C=0.3 재사용)" if exp["id"] == "E5" and sp.get("tau") == 9 else ""
        lines.append(f"| `{setting}` | {sp.get('variant', METHOD_TAG[sp['method']])}{tau} | {sp['C']} | "
                     f"{len(done)}/{target}{note} | {acc_s} | {wall_s} |")
    return "\n".join(lines)


def partial_notes(exp, loaded):
    """아직 10회를 채우지 못한 설정 (표의 값은 완료된 반복만 평균)."""
    notes = []
    settings = {}
    for spec in exp["runs"]:
        settings.setdefault(spec["setting"], []).append(spec)
    for setting, specs in settings.items():
        done = sum(1 for s in specs if loaded.get(s["name"]) is not None and loaded[s["name"]]["complete"])
        if done < len(specs):
            notes.append(f"{setting}: 완료 반복 {done}/{len(specs)} (완료된 반복만 평균)")
    return notes


def write_summary(exp, sections):
    d = EXP_ROOT / exp["dir"]
    body = [f"# {exp['id']}. {exp['title']}", "",
            f"마지막 갱신: {datetime.now().strftime('%Y-%m-%d %H:%M')}", "",
            f"**재현 설정**: {SETUP_TEXT if exp['id'] != 'E8' else SETUP_TEXT.replace('K=30', 'K=10').replace('× 6명', '× 2명')}", ""] + sections
    (d / "SUMMARY.md").write_text("\n".join(body) + "\n")


def compare(exp_dir, paper_name, ours_png, out_name, title):
    paper = PAPER_FIGS / paper_name
    if paper.exists() and ours_png.exists():
        A.compare_with_paper(paper, ours_png, exp_dir / out_name, title)
        return f"![{title}]({out_name})"
    return ""


# ------------------------------------------------------------------ Table 2 계열 공통 정리
def table2_section(exp, d, dataset, loaded, mc):
    ours = {c: {m: A.metrics(mc.get((m, c)), dataset) for m in ("BSA", "SASAFL")} for c in (0.1, 0.3, 0.5)}
    rows = A.table2_rows(dataset, P.TABLE2[dataset], ours)
    notes = partial_notes(exp, loaded)
    md = A.table2_markdown(dataset, rows, notes)
    (d / "table2_comparison.md").write_text(md + "\n")
    obs = []
    for c in (0.1, 0.3, 0.5):
        p_b, p_s = P.TABLE2[dataset][c]["BSA"][0], P.TABLE2[dataset][c]["SASAFL"][0]
        o_b, o_s = ours[c]["BSA"][0], ours[c]["SASAFL"][0]
        if o_b is None or o_s is None:
            continue
        paper_dir = "SASAFL" if p_s > p_b else "BSA"
        our_dir = "SASAFL" if o_s > o_b else "BSA"
        obs.append(f"- C={c}: 정확도 차이(SASAFL−BSA) 논문 {p_s - p_b:+.2f}%p, 재현 {o_s - o_b:+.2f}%p → "
                   f"{'논문과 같은 방향' if paper_dir == our_dir else '**논문과 반대 방향**'}")
        rb, rs = ours[c]["BSA"][1], ours[c]["SASAFL"][1]
        if rb and rs:
            pb, ps = P.TABLE2[dataset][c]["BSA"][1], P.TABLE2[dataset][c]["SASAFL"][1]
            paper_ratio = f"{pb / ps:.2f}배" if pb and ps else "BSA 미도달"
            obs.append(f"  - round-to-accuracy BSA/SASAFL = {rb / rs:.2f}배 (논문 {paper_ratio}) "
                       f"→ 1보다 크면 SASAFL이 더 빨리 도달")
    sec = ["## Table 2 비교", "", md, "", 
           "- Accuracy = 최고 test 정확도, Round-to-accuracy = 목표 첫 도달 라운드, "
           "Time-to-accuracy = 서버 setup 시작부터 목표 도달까지(분). `-` = 도달 못 함/미실행",
           "- Time-to-accuracy는 단일 노드 과다 할당(thread 109개 / 논리 CPU 32개) 환경이라 논문과 절대값 비교가 어렵다.",
           ""]
    if obs:
        sec += ["### 관찰", ""] + obs + [""]
    return sec, ours


def table4_section(d, dataset, mc):
    paper_rounds = P.TABLE4_K30[dataset]["rounds"]
    rounds = min(paper_rounds, MAX_ROUNDS)
    header = ["", "C=0.1", "C=0.3", "C=0.5"]
    paper_row = [f"논문 K=30 ({paper_rounds}R)"] + [P.TABLE4_K30[dataset][c] for c in (0.1, 0.3, 0.5)]
    ours_row = [f"재현 K=30 ({rounds}R)"]
    for c in (0.1, 0.3, 0.5):
        w = A.wall_at_round(mc.get(("SASAFL", c)), rounds)
        ours_row.append("-" if w is None else f"{w:.0f}")
    md = ["| " + " | ".join(header) + " |", "|---|---:|---:|---:|",
          "| " + " | ".join(map(str, paper_row)) + " |", "| " + " | ".join(map(str, ours_row)) + " |"]
    note = ("" if rounds == paper_rounds else
            f" **주의: 논문은 {paper_rounds}라운드 기준, 재현은 {rounds}라운드 기준이라 직접 비교 불가 "
            f"(라운드당 시간 비교용: 논문 {paper_rounds}R 값을 {rounds}/{paper_rounds}배 하면 대략적인 기준이 된다).**")
    return ["## Table 4 (K=30 행)", "", f"SASAFL로 {rounds}라운드를 마치는 데 걸린 시간(분).{note}", ""] + md + \
           ["", "<!-- /table4 -->", ""]


# ------------------------------------------------------------------ 실험별 정리
def report_E1(exp, loaded):
    d = EXP_ROOT / exp["dir"]
    mc = by_method_c(exp, loaded)
    sec, _ = table2_section(exp, d, "MNIST", loaded, mc)
    return sec + table4_section(d, "MNIST", mc)


def report_E2(exp, loaded):
    d = EXP_ROOT / exp["dir"]
    mc = by_method_c(exp, loaded)
    sec, _ = table2_section(exp, d, "FashionMNIST", loaded, mc)
    return sec + table4_section(d, "FashionMNIST", mc)


def report_E3(exp, loaded):
    d = EXP_ROOT / exp["dir"]
    e1 = next(e for e in EXPERIMENTS if e["id"] == "E1")
    semi = A.load_run(runs_root(e1) / "T2_MNIST_BSA_C0.1_single_cats12348")
    sync, asyn = loaded["SYNC_FedAvg_C0.1"], loaded["ASYNC_FedAsync"]
    A.fig2(sync, asyn, semi, d / "fig2_sync_async_semi.png")

    def describe(run):
        if run is None or len(run["acc"]) == 0:
            return ("-", "-", "-")
        idx90 = [i for i, a in enumerate(run["acc"]) if a >= 90 and run["rounds"][i] >= 1]
        t90 = f"{run['wall'][idx90[0]]:.1f}" if idx90 else "-"
        return (f"{run['acc'][-1]:.2f}", f"{run['wall'][-1]:.1f}", t90)

    rows = [("Synchronous (FedAvg, C=0.1)", sync), ("Asynchronous (FedAsync)", asyn),
            ("Semi-asynchronous (BSA, C=0.1, E1 재사용)", semi)]
    md = ["| 방식 | 최종 정확도(%) | 500라운드 소요(분) | 90% 도달(분) |", "|---|---:|---:|---:|"]
    md += [f"| {name} | " + " | ".join(describe(r)) + " |" for name, r in rows]
    md += ["", f"논문 본문: semi 최종 {P.FIG2_TEXT['semi_final_acc']}%, sync는 그보다 약 6%p 높음, "
               f"90% 도달 시간 semi 약 {P.FIG2_TEXT['time_to_90_semi_min']}분 / sync 약 {P.FIG2_TEXT['time_to_90_sync_min']}분, "
               "500라운드 wall clock은 sync가 semi·async의 약 3배."]
    return ["## Fig. 2 — 동기·비동기·반비동기 (MNIST, 500라운드)", "",
            "논문은 Fig.2의 K와 C를 밝히지 않았다. K=30, C=0.1로 가정했다. 반비동기는 E1의 BSA C=0.1 실행을 재사용한다.", "",
            "![Fig2](fig2_sync_async_semi.png)", "",
            compare(d, "paper_fig2.jpg", d / "fig2_sync_async_semi.png", "compare_fig2.png", "Fig. 2 비교"), "",
            ] + md + [""]


def report_E4(exp, loaded):
    d = EXP_ROOT / exp["dir"]
    mc = by_method_c(exp, loaded)
    sec, _ = table2_section(exp, d, "CIFAR10", loaded, mc)
    sec += ["**라운드 축소**: 논문은 CIFAR-10을 3000라운드 돌렸지만 여기서는 500라운드로 고정했다. "
            "그래서 논문 그림과는 0~500라운드 구간만 비교할 수 있다.", ""]
    return sec + table4_section(d, "CIFAR10", mc)


def report_E5(exp, loaded):
    e1 = next(e for e in EXPERIMENTS if e["id"] == "E1")
    acc_by_tau = e5_acc_by_tau({"E5": loaded, "E1": load_exp_runs(e1)})
    md = ["| τ | 실행 수 | 평균 | 최소 | 최대 |", "|---:|---:|---:|---:|---:|"]
    for t in range(1, 11):
        v = acc_by_tau.get(t, [])
        md.append(f"| {t} | {len(v)} | " + (" | ".join(f"{f(v):.2f}" for f in (lambda x: sum(x) / len(x), min, max))
                                           if v else "- | - | -") + " |")
    best = max(acc_by_tau, key=lambda t: sum(acc_by_tau[t]) / len(acc_by_tau[t])) if acc_by_tau else None
    return ["## Fig. 5 — lag tolerance별 최고 정확도", "",
            "논문은 Fig.5의 데이터셋·C를 밝히지 않았고 설정당 10회 반복했다. "
            "여기서는 MNIST, C=0.3으로 가정하고 τ마다 10회 반복한다(완료된 반복만 집계). "
            "τ=9는 E1의 SASAFL C=0.3과 같은 설정이라 그 반복을 그대로 쓴다.", "",
            ] + md + ["", f"평균이 가장 높은 τ: **{best}** (논문은 τ=9 선택)" if best else "", ""]


def report_E6(exp, loaded_all):
    d = EXP_ROOT / exp["dir"]
    d.mkdir(parents=True, exist_ok=True)
    exps = {e["id"]: e for e in EXPERIMENTS}
    mcs = {eid: by_method_c(exps[eid], loaded_all.get(eid, {})) for eid in ("E1", "E2", "E4")}
    ours = {ds: {c: {m: A.metrics(mcs[eid].get((m, c)), ds) for m in ("BSA", "SASAFL")} for c in (0.1, 0.3, 0.5)}
            for eid, ds in (("E1", "MNIST"), ("E2", "FashionMNIST"), ("E4", "CIFAR10"))}
    sec = ["## Table 2 전체 (K=30)", ""]
    for eid, dataset in (("E1", "MNIST"), ("E2", "FashionMNIST"), ("E4", "CIFAR10")):
        md = EXP_ROOT / next(e for e in EXPERIMENTS if e["id"] == eid)["dir"] / "table2_comparison.md"
        sec += [md.read_text() if md.exists() else f"{dataset}: 아직 결과 없음", ""]
    sec += ["## Table 4 (K=30 행)", ""]
    for eid in ("E1", "E2", "E4"):
        sm = EXP_ROOT / next(e for e in EXPERIMENTS if e["id"] == eid)["dir"] / "SUMMARY.md"
        if sm.exists():
            text = sm.read_text()
            if "## Table 4" in text:
                sec += [text.split("## Table 4 (K=30 행)")[1].split("<!-- /table4 -->")[0].strip(), ""]
    sec += ["## 결론 수치 점검", "",
            "| 결론의 주장 | 논문 출처 | 이 장비에서 재현 |", "|---|---|---|"]
    for claim, src in P.CONCLUSION_CLAIMS:
        sec.append(f"| {claim} | {src} | 불가 (K=100) |")
    sec += ["", "K=30에서 같은 지표(재현값, 논문 K=30 값):", ""]
    for eid, dataset in (("E1", "MNIST"), ("E2", "FashionMNIST"), ("E4", "CIFAR10")):
        exp_i = next(e for e in EXPERIMENTS if e["id"] == eid)
        mc = by_method_c(exp_i, loaded_all.get(eid, {}))
        for c in (0.1, 0.3, 0.5):
            o_b, o_s = A.metrics(mc.get(("BSA", c)), dataset), A.metrics(mc.get(("SASAFL", c)), dataset)
            p_b, p_s = P.TABLE2[dataset][c]["BSA"], P.TABLE2[dataset][c]["SASAFL"]
            gap = f"{o_s[0] - o_b[0]:+.2f}" if o_b[0] is not None and o_s[0] is not None else "-"
            rr = f"{o_b[1] / o_s[1]:.2f}" if o_b[1] and o_s[1] else "-"
            tr = f"{o_b[2] / o_s[2]:.2f}" if o_b[2] and o_s[2] else "-"
            prr = f"{p_b[1] / p_s[1]:.2f}" if p_b[1] and p_s[1] else "-"
            ptr = f"{p_b[2] / p_s[2]:.2f}" if p_b[2] and p_s[2] else "-"
            sec.append(f"- {dataset} C={c}: 정확도 차이 재현 {gap}%p (논문 {p_s[0] - p_b[0]:+.2f}%p), "
                       f"RTA 배수 재현 {rr} (논문 {prr}), TTA 배수 재현 {tr} (논문 {ptr})")
    sec += ["", "## 이 장비로 재현하지 못한 항목", "",
            "- K=100: Table 3, Fig. 6, Table 1, Fig. 7(b)(d), Fig. 8(b), Fig. 10, Table 4의 K≥100 행",
            "- 이유: 논문 K=100 카테고리(1·2·3·5·8코어 × 20명)는 노드당 190 thread가 필요하다. "
            "두 노드(각 논리 32)로 시도한 실행은 실패했다(../../crash_study, ../README.md 참고).", ""]
    return sec




def build_figures(loaded_all):
    """논문 원본(왼쪽) / 재현(오른쪽) 비교 그림을 모든 실험 폴더에 다시 만든다.
    두 노드 결과 동기화·실행 종료 정리가 동시에 그림을 쓰면 반쯤 쓰인 PNG를 읽게 되므로 파일 잠금으로 한 번에 하나만 그린다."""
    with open(EXP_ROOT / ".figures.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _build_figures(loaded_all)


def _build_figures(loaded_all):
    exps = {e["id"]: e for e in EXPERIMENTS}
    mcs = {"MNIST": by_method_c(exps["E1"], loaded_all.get("E1", {})),
           "FashionMNIST": by_method_c(exps["E2"], loaded_all.get("E2", {})),
           "CIFAR10": by_method_c(exps["E4"], loaded_all.get("E4", {}))}
    acc_by_tau = e5_acc_by_tau(loaded_all)
    dirs = {eid: EXP_ROOT / exps[eid]["dir"] for eid in ("E1", "E2", "E4", "E5", "E6", "E7", "E8")}
    made = PF.build_all(mcs, acc_by_tau, dirs, MAX_ROUNDS, k10_mc(loaded_all, "SASAFL"))
    for eid, names in PF.build_err(mcs, err_sasafl(loaded_all), dirs, MAX_ROUNDS,
                                   k10_mc(loaded_all, "SASAFLorig")).items():
        made.setdefault(eid, []).extend(names)
    return made


ERR_DATASETS = (("MNIST", "MNIST", "E1"), ("FMNIST", "FashionMNIST", "E2"), ("CIFAR10", "CIFAR10", "E4"))


def k10_mc(loaded_all, variant):
    """E8 실행 → {dataset: {(BSA|SASAFL, C): 반복 평균 run}}. variant: 'SASAFL'(현재 코드) | 'SASAFLorig'(원저자 코드)"""
    exp = next(e for e in EXPERIMENTS if e["id"] == "E8")
    g = grouped(exp, loaded_all.get("E8", {}), lambda sp: (sp["dataset"], sp["variant"], sp["C"]))
    return {ds: {**{("BSA", c): g.get((ds, "BSA", c)) for c in (0.1, 0.3, 0.5)},
                 **{("SASAFL", c): g.get((ds, variant, c)) for c in (0.1, 0.3, 0.5)}}
            for _, ds, _ in ERR_DATASETS}


def report_E8(exp, loaded):
    k10_now, k10_old = k10_mc({"E8": loaded}, "SASAFL"), k10_mc({"E8": loaded}, "SASAFLorig")
    sec = ["## 무엇을 돌렸나", "",
           "- 논문은 K=30과 K=100을 비교했지만 이 장비에서 K=100은 돌릴 수 없어 K=10으로 대신한다. "
           "그림·표에서 논문의 K=100 칸 자리에 K=10을 넣었다(왼쪽 논문 원본은 K=100 그대로).",
           "- K=10 클라이언트는 K=30과 같은 CPU 카테고리(1·2·3·4·8코어)를 2명씩 쓴다. 나머지 설정은 K=30과 같다.",
           "- SASAFL은 현재 코드(quota 적응 동작)와 원저자 코드(c0b819b, quota 고정) 둘 다 돌렸다.", "",
           "## 실행 상태", "", status_table(exp, loaded), "",
           "## 최고 정확도 (500라운드)", "",
           "| 데이터셋 | C | BSA | SASAFL 현재 코드 | SASAFL 원저자 코드 | 원저자 코드 quota: 범위 / 서로 다른 값 수 / int(C·K) 비율 |",
           "|---|---:|---:|---:|---:|---|"]
    for _, ds, _ in ERR_DATASETS:
        for c in (0.1, 0.3, 0.5):
            fmt = lambda r: "-" if A.metrics(r, ds)[0] is None else f"{A.metrics(r, ds)[0]:.2f}"
            old = k10_old[ds][("SASAFL", c)]
            q = quota_summary(old, c, k=10)
            sec.append(f"| {ds} | {c} | {fmt(k10_now[ds][('BSA', c)])} | {fmt(k10_now[ds][('SASAFL', c)])} | "
                       f"{fmt(old)} | {' / '.join(q)} |")
    return sec + [""]


def err_sasafl(loaded_all):
    """E7 실행 → {dataset: {C: 반복 평균 run}}"""
    exp = next(e for e in EXPERIMENTS if e["id"] == "E7")
    g = grouped(exp, loaded_all.get("E7", {}), lambda sp: (sp["dataset"], sp["C"]))
    return {ds: {c: g.get((ds, c)) for c in (0.1, 0.3, 0.5)} for _, ds, _ in ERR_DATASETS}


def e5_acc_by_tau(loaded_all):
    """τ별 반복의 최고 정확도 목록 (Fig.5 박스플롯). τ=9는 LT_tau9_rep1(=E1 1회차 링크) + E1 SASAFL C=0.3 2~10회차."""
    exps = {e["id"]: e for e in EXPERIMENTS}
    out = {}
    for spec in exps["E5"]["runs"]:
        run = loaded_all.get("E5", {}).get(spec["name"])
        if run is not None and run["complete"]:
            out.setdefault(spec["tau"], []).append(A.metrics(run, "MNIST")[0])
    for spec in exps["E1"]["runs"]:
        if spec["method"] == "SASAFL" and spec["C"] == 0.3 and spec["rep"] >= 2:
            run = loaded_all.get("E1", {}).get(spec["name"])
            if run is not None and run["complete"]:
                out.setdefault(9, []).append(A.metrics(run, "MNIST")[0])
    return out


def write_repeat_files(exp, loaded):
    """반복 실험 값을 실험 폴더의 repeats/에 파일로 나눠 저장한다.
    - runs_index.csv: 실행(반복) 하나당 한 줄 — 설정, 회차, 노드, 최고 정확도, 도달 라운드/시간, 소요 시간, 폴더
    - summary_by_setting.csv: 설정별 완료 반복 수와 평균·표준편차·최소·최대
    - curves/<설정>.csv: 라운드별 정확도 (회차별 열 + 평균·표준편차) — 그림이 쓰는 반복 평균 곡선"""
    import csv
    d = EXP_ROOT / exp["dir"] / "repeats"
    (d / "curves").mkdir(parents=True, exist_ok=True)
    root = runs_root(exp)
    settings = {}
    with (d / "runs_index.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "rep", "run_name", "method", "variant", "dataset", "K", "C", "tau", "host", "outcome",
                    "rounds_done", "best_acc", "round_to_acc", "time_to_acc_min", "wall_min", "run_dir"])
        for spec in exp["runs"]:
            run = loaded.get(spec["name"])
            settings.setdefault(spec["setting"], []).append((spec, run))
            if run is None:
                w.writerow([spec["setting"], spec["rep"], spec["name"], METHOD_TAG[spec["method"]], spec.get("variant", ""),
                            spec["dataset"], spec["K"], spec["C"], spec.get("tau", ""), "", "not_run"] + [""] * 6)
                continue
            m = A.metrics(run, spec["dataset"])
            w.writerow([spec["setting"], spec["rep"], spec["name"], METHOD_TAG[spec["method"]], spec.get("variant", ""),
                        spec["dataset"], spec["K"], spec["C"], spec.get("tau", ""), run["cfg"].get("host", "wise-175"),
                        run["result"].get("outcome", "running"), run["rounds_done"],
                        "" if m[0] is None else f"{m[0]:.2f}", "" if m[1] is None else m[1],
                        "" if m[2] is None else f"{m[2]:.2f}", run["result"].get("wall_min", ""),
                        os.path.relpath(root / spec["name"], EXP_ROOT / exp["dir"])])
    with (d / "summary_by_setting.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "method", "variant", "dataset", "K", "C", "tau", "reps_done", "reps_target",
                    "best_acc_mean", "best_acc_std", "best_acc_min", "best_acc_max",
                    "round_to_acc_mean", "time_to_acc_mean_min", "reps_reached_target", "wall_min_mean"])
        for setting, items in settings.items():
            sp = items[0][0]
            done = [r for _, r in items if r is not None and r["complete"]]
            ms = [A.metrics(r, sp["dataset"]) for r in done]
            accs = [m[0] for m in ms if m[0] is not None]
            hit = [m for m in ms if m[1] is not None]
            walls = [r["result"].get("wall_min") for r in done if r["result"].get("wall_min")]
            fmt = lambda v: "" if v is None else f"{v:.2f}"
            w.writerow([setting, METHOD_TAG[sp["method"]], sp.get("variant", ""), sp["dataset"], sp["K"], sp["C"],
                        sp.get("tau", ""), len(done), len(items),
                        fmt(np.mean(accs) if accs else None), fmt(np.std(accs) if accs else None),
                        fmt(min(accs) if accs else None), fmt(max(accs) if accs else None),
                        fmt(np.mean([m[1] for m in hit]) if hit else None), fmt(np.mean([m[2] for m in hit]) if hit else None),
                        len(hit), fmt(np.mean(walls) if walls else None)])
            comb = A.combine_runs([r for _, r in items])
            if comb is None:
                continue
            with (d / "curves" / f"{setting}.csv").open("w", newline="") as cf:
                cw = csv.writer(cf)
                reps = comb["reps"]
                cw.writerow(["round"] + [f"rep_{r['name']}" for r in reps] + ["mean", "std"])
                for i, rnd in enumerate(comb["rounds"]):
                    vals = [r["acc"][np.searchsorted(r["rounds"], rnd)] for r in reps]
                    cw.writerow([int(rnd)] + [f"{v:.2f}" for v in vals] + [f"{comb['acc'][i]:.3f}", f"{comb['acc_std'][i]:.3f}"])


def queue_order():
    """두 노드가 나눠 돌릴 순서: 남은 1회차(E5) → 2회차 전체 → 3회차 전체 … (회차 단위로 한 벌씩 완성).
    한 회차 안에서는 Table 2 핵심(E1·E2·E4) → E7(원저자 코드) → E8(K=10) → E5(τ) 순."""
    order = []
    exps = {e["id"]: e for e in EXPERIMENTS}
    for r in range(1, REPS + 1):
        for eid in ("E1", "E2", "E4", "E7", "E8", "E5"):
            for spec in exps[eid]["runs"]:
                if spec["rep"] == r:
                    order.append((eid, spec["name"], os.path.relpath(runs_root(exps[eid]) / spec["name"], K30)))
    return order


def quota_summary(run, c, k=30):
    """라운드별 quota(다음 라운드에 쓸 로컬 모델 수) 요약 문자열."""
    if run is None or len(run["quota"]) == 0:
        return "-", "-", "-"
    q = run["quota"][~np.isnan(run["quota"])].astype(int)
    if len(q) == 0:
        return "-", "-", "-"
    fixed = int(k * c)
    return f"{q.min()}–{q.max()}", str(len(set(q.tolist()))), f"{100 * (q == fixed).mean():.0f}%"


def report_E7(exp, loaded):
    exps = {e["id"]: e for e in EXPERIMENTS}
    sha = hashlib.sha256(ERR_SASAFL_SRC.read_bytes()).hexdigest()
    sec = ["## 무엇을 돌렸나", "",
           f"- SASAFL.py만 원저자 마지막 커밋 `c0b819b`(2023-12-14)의 파일로 바꿔 실행했다: "
           f"`original_code/SASAFL_c0b819b.py` (sha256 `{sha[:16]}…`). 나머지 코드는 현재 저장소와 동일하다.",
           "- 이 파일은 집계 직후 `Server.last_global_loss = global_loss`로 직전 loss를 덮어쓴 뒤 "
           "`get_next_round_minimum_local_model(global_loss, ...)`를 부른다. 그래서 gl_r / gl_(r-1)이 항상 1이 되어 "
           "quota가 처음 값 int(C·K)에 고정된다.",
           "- 현재 저장소 코드(E1·E2·E4)는 `73f5e0b`(2026-10-08)에서 이 줄을 루프 밖으로 옮겨 quota가 실제로 변한다.",
           "- BSA는 quota 적응을 쓰지 않으므로 E1·E2·E4의 BSA 실행을 그대로 쓴다.", "",
           "## 실행 상태", "", status_table(exp, loaded), "",
           "## quota가 실제로 고정됐는지 (rounds.csv의 라운드별 quota)", "",
           "| 데이터셋 | C | int(C·K) | 현재 코드: 범위 / 서로 다른 값 수 / 고정값 비율 | 원저자 코드: 범위 / 서로 다른 값 수 / 고정값 비율 | 원저자 코드 실행 노드 |",
           "|---|---:|---:|---|---|---|"]
    acc = ["## 최고 정확도 비교 (500라운드)", "",
           "| 데이터셋 | C | BSA | SASAFL 현재 코드 | SASAFL 원저자 코드 | 논문 BSA | 논문 SASAFL |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    err = err_sasafl({"E7": loaded})
    for tag, ds, eid in ERR_DATASETS:
        mc = by_method_c(exps[eid], load_exp_runs(exps[eid]))
        for c in (0.1, 0.3, 0.5):
            now, old = mc.get(("SASAFL", c)), err[ds][c]
            sec.append(f"| {ds} | {c} | {int(30 * c)} | " + " / ".join(quota_summary(now, c)) + " | "
                       + " / ".join(quota_summary(old, c)) + " | "
                       + ("-" if old is None else f"{old['cfg'].get('host', 'wise-175')} {old['cfg'].get('cpu_list', '')}".strip()) + " |")
            fmt = lambda r: "-" if A.metrics(r, ds)[0] is None else f"{A.metrics(r, ds)[0]:.2f}"
            p = P.TABLE2[ds][c]
            acc.append(f"| {ds} | {c} | {fmt(mc.get(('BSA', c)))} | {fmt(now)} | {fmt(old)} | {p['BSA'][0]} | {p['SASAFL'][0]} |")
    sec += ["", "원저자 코드 열의 고정값 비율이 100%면 quota 적응이 한 번도 일어나지 않았다는 뜻이다.",
            "E7은 wise-175와 wise-176에 나눠 돌렸다. wise-176은 결함 코어 24(CPU 12,13)를 빼고 30개 논리 CPU만 쓴다 "
            "(175는 32개). 그래서 실행 시간은 노드끼리 직접 비교하지 않는다.", ""]
    hist = ["## 논문 실험 당시 코드가 이 버전이었는지 (git 기록)", "",
            "| 커밋 | 날짜 | 작성자 | quota 적응 |", "|---|---|---|---|",
            "| `e7304af`~`6546e9d` | 2023-11-24~26 | miri | 동작 (last_global_loss 초기값 1.0, 서버 안에서만 갱신) |",
            "| `a45de5d` \"PM1\" | 2023-11-27 | miri | **꺼짐** — 루프 안 덮어쓰기 줄 추가, 초기값 None |",
            "| `d36ab53`·`a5cf02b`·`96dc297` | 2023-11-28 | miri | 꺼짐 (마지막 SASAFL 수정) |",
            "| `c0b819b` | 2023-12-14 | miri | 꺼짐 (원저자 마지막 커밋, SAFA 수정) |",
            "| 논문 최초 투고 | 2023-12-19 | — | (논문 첫 페이지 \"Received 19 December 2023\") |",
            "| `73f5e0b` | 2026-10-08 | CheongJaeyoon | 동작 (덮어쓰기 줄을 루프 밖으로 이동) |", "",
            "- lag tolerance 인자(Fig.5)는 11-26~27 커밋에서 처음 생겼다. Fig.5를 포함한 투고 실험은 11-27 이후 코드, "
            "즉 quota가 고정된 버전으로 돌렸을 가능성이 높다.",
            "- 한계: 2024-05 수정본(revised) 때 다시 돌린 실험이 있다면 그 코드는 이 저장소에 없다. "
            "그래서 \"투고본 실험은 고정 버전일 가능성이 높다\"까지가 git 기록으로 말할 수 있는 범위다.", ""]
    return hist + sec + acc + [""]


REPORTS = {"E1": report_E1, "E2": report_E2, "E4": report_E4, "E5": report_E5, "E7": report_E7, "E8": report_E8}


def build_report(exp, loaded_all):
    d = EXP_ROOT / exp["dir"]
    d.mkdir(parents=True, exist_ok=True)
    try:
        made = build_figures(loaded_all)
        figs = ["## 비교 그림 (왼쪽: 논문 원본, 오른쪽: 재현)", ""]
        figs += [f"![{name}]({name})\n" for name in made.get(exp["id"], [])]
        if exp["id"] == "E6":
            sections = figs + report_E6(exp, loaded_all)
        else:
            loaded = loaded_all[exp["id"]]
            write_repeat_files(exp, loaded)
            sections = figs + ["## 반복 실행 상태 (그림은 완료된 반복의 평균 곡선, 표는 반복별 값의 평균)", "",
                               status_table(exp, loaded), "",
                               "반복 값 파일: `repeats/runs_index.csv`(실행별), `repeats/summary_by_setting.csv`(설정별 평균·표준편차), "
                               "`repeats/curves/<설정>.csv`(라운드별 회차 값과 평균)", ""] + REPORTS[exp["id"]](exp, loaded)
        write_summary(exp, sections)
    except Exception:
        (d / "report_error.log").write_text(traceback.format_exc())
        log_progress(f"{exp['id']} 정리 중 오류 → {d / 'report_error.log'}")


# ------------------------------------------------------------------ 실행
def claim(name):
    """두 노드가 같은 실행 목록을 나눠 가질 때: 실행 이름을 먼저 차지한 노드만 돌린다.
    K30_CLAIM_DIR(wise-175의 폴더)에 mkdir로 원자적으로 차지한다. K30_CLAIM_SSH가 있으면 그 호스트에서 한다.
    반환: 차지한 노드 이름 (설정이 없으면 이 노드, 연결 실패면 None)"""
    claim_dir = os.environ.get("K30_CLAIM_DIR")
    me = socket.gethostname()
    if not claim_dir:
        return me
    d = f"{claim_dir}/{name}"
    cmd = f"mkdir -p {claim_dir} && {{ mkdir {d} 2>/dev/null && echo {me} > {d}/host; cat {d}/host; }}"
    ssh = os.environ.get("K30_CLAIM_SSH")
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", ssh, cmd] if ssh else ["bash", "-c", cmd]
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=60).stdout.strip()
    except subprocess.TimeoutExpired:
        return None
    return out or None


def run_experiment(exp, port_counter, stall_min, attempts, names=None):
    root = runs_root(exp)
    root.mkdir(parents=True, exist_ok=True)
    if exp["id"] == "E1":
        link = EXP_ROOT / exp["dir"] / "runs"
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(Path("../../runs"))
    for spec in exp["runs"]:
        if names and spec["name"] not in names:
            continue
        existing = A.load_run(root / spec["name"])
        if existing is not None and existing["complete"]:
            continue
        owner = claim(spec["name"])
        while owner is None:   # claim 폴더(wise-175)에 닿지 않음 → 건너뛰지 말고 다시 닿을 때까지 기다린다
            log_progress(f"{exp['id']} claim 확인 실패: {spec['name']} — 2분 뒤 다시 시도")
            time.sleep(120)
            owner = claim(spec["name"])
        if owner != socket.gethostname():
            log_progress(f"{exp['id']} 건너뜀: {spec['name']} (차지한 노드: {owner or '확인 실패'})")
            continue
        for attempt in range(1, attempts + 1):
            log_progress(f"{exp['id']} 실행 시작: {spec['name']} (시도 {attempt}/{attempts})")
            port_counter[0] += 1
            port = 29700 + (port_counter[0] % 250)
            result = Run(dict(spec), root, stall_min * 60, spec.get("max_wall_min", 600) * 60, port).run()
            done = result["outcome"] == "completed" and result["rounds_done"] >= spec["rounds"]
            log_progress(f"{exp['id']} 실행 종료: {spec['name']} → {result['outcome']}, "
                         f"{result['rounds_done']}/{spec['rounds']}라운드, 최고 {result['best_acc']}%, "
                         f"{result['wall_min']}분")
            time.sleep(15)
            if done:
                break
        yield spec["name"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--stall-min", type=float, default=20)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--names", nargs="*", help="이 실행 이름들만 돌린다 (두 노드에 나눠 돌릴 때)")
    parser.add_argument("--quiet", action="store_true", help="PROGRESS.md에 기록하지 않음")
    parser.add_argument("--print-queue", action="store_true", help="두 노드 실행 순서 출력: 실험ID 실행이름 실행폴더(k30_reproduction 기준)")
    args = parser.parse_args()
    if args.print_queue:
        for eid, name, rel in queue_order():
            print(eid, name, rel)
        return
    global QUIET
    QUIET = args.quiet
    exps = [e for e in EXPERIMENTS if not args.only or e["id"] in args.only]
    EXP_ROOT.mkdir(exist_ok=True)
    port_counter = [int(time.time()) % 100]
    log_progress(f"파이프라인 시작 ({', '.join(e['id'] for e in exps)}, report_only={args.report_only})")
    loaded_all = {}
    for exp in EXPERIMENTS:   # E6 정리를 위해 전체 로딩
        if exp["runs"]:
            loaded_all[exp["id"]] = load_exp_runs(exp)
    for exp in exps:
        if not args.report_only and exp["runs"]:
            log_progress(f"=== {exp['id']} 시작: {exp['title']}")
            for _ in run_experiment(exp, port_counter, args.stall_min, args.attempts, args.names):
                loaded_all[exp["id"]] = load_exp_runs(exp)
                build_report(exp, loaded_all)
        if exp["runs"]:
            loaded_all[exp["id"]] = load_exp_runs(exp)
        build_report(exp, loaded_all)
        log_progress(f"=== {exp['id']} 정리 완료 → experiments/{exp['dir']}/SUMMARY.md")
    log_progress("파이프라인 종료")


if __name__ == "__main__":
    main()
