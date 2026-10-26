"""실행 결과 로딩, 지표 계산, 논문 형식의 그래프와 표 생성."""

import csv
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import warnings  # noqa: E402
warnings.filterwarnings("ignore", "Mean of empty slice")
from matplotlib import font_manager  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from paper_values import TARGET_ACC  # noqa: E402

FONT_R = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
FONT_B = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
for f in (FONT_R, FONT_B):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": ["Liberation Serif", "NanumGothic"], "axes.unicode_minus": False, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white", "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
})

# 논문 그림의 색과 선 스타일
GREEN, RED = "#00a650", "#ff0000"
C_STYLE = {0.1: "-", 0.3: ":", 0.5: "--"}
TS_RE = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\] SERVER >>")


# ------------------------------------------------------------------ 로딩과 지표
def load_run(run_dir):
    """run 폴더 -> dict. 실행 기록이 없으면 None."""
    run_dir = Path(run_dir)
    if not (run_dir / "rounds.csv").exists():
        return None
    cfg = json.loads((run_dir / "config.json").read_text())
    res = json.loads((run_dir / "result.json").read_text()) if (run_dir / "result.json").exists() else {}
    start = None
    with (run_dir / "run.log").open(errors="replace") as f:
        for line in f:
            m = TS_RE.search(line)
            if m:
                start = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
                break
    rounds, acc, loss, wall, quota = [], [], [], [], []
    for r in csv.DictReader((run_dir / "rounds.csv").open()):
        if not r["acc"]:
            continue
        rounds.append(int(r["round"]))
        acc.append(float(r["acc"]))
        loss.append(float(r["loss"]) if r.get("loss") else np.nan)
        ts = datetime.strptime(r["ts"], "%Y-%m-%d %H:%M:%S.%f")
        wall.append((ts - start).total_seconds() / 60 if start else np.nan)
        quota.append(int(r["next_quota"]) if r.get("next_quota") else np.nan)
    return {
        "dir": run_dir, "name": run_dir.name, "cfg": cfg, "result": res,
        "rounds": np.array(rounds), "acc": np.array(acc), "loss": np.array(loss),
        "wall": np.array(wall), "quota": np.array(quota),
        "rounds_done": int(max(rounds)) if rounds else 0,
        "requested": cfg.get("rounds"),
        "complete": res.get("outcome") == "completed" and rounds and max(rounds) >= cfg.get("rounds", 0),
    }


def combine_runs(runs):
    """같은 설정의 반복 실행들(완료된 것만) → 반복 평균 '가상 실행'.
    그림용 곡선은 라운드별 평균, 표용 지표는 반복마다 구한 뒤 평균한다(metrics/wall_at_round가 reps를 본다).
    반복이 없으면 None."""
    runs = [r for r in runs if r is not None and r["complete"]]
    if not runs:
        return None
    rounds = runs[0]["rounds"]
    for r in runs[1:]:
        rounds = np.intersect1d(rounds, r["rounds"])

    def stack(key):
        return np.vstack([r[key][np.searchsorted(r["rounds"], rounds)] for r in runs])
    return {
        "dir": runs[0]["dir"], "name": runs[0]["name"], "cfg": runs[0]["cfg"], "result": runs[0]["result"],
        "rounds": rounds, "acc": stack("acc").mean(axis=0), "acc_std": stack("acc").std(axis=0),
        "loss": np.nanmean(stack("loss"), axis=0), "wall": np.nanmean(stack("wall"), axis=0),
        "quota": np.concatenate([r["quota"] for r in runs]),
        "rounds_done": int(rounds.max()) if len(rounds) else 0, "requested": runs[0]["requested"],
        "complete": True, "reps": runs,
    }


def metrics(run, dataset):
    """논문 정의: 최고 정확도, 목표 첫 도달 라운드, 그때까지 경과 시간(분).
    반복 평균 실행(reps)이면 반복마다 구해 평균한다. 도달 라운드·시간은 반복의 절반 이상이 도달했을 때만
    도달한 반복들의 평균을 쓰고, 아니면 '-'(None)."""
    if run is not None and "reps" in run:
        each = [metrics(r, dataset) for r in run["reps"]]
        accs = [m[0] for m in each if m[0] is not None]
        hit = [m for m in each if m[1] is not None]
        reached = len(hit) * 2 >= len(each)
        return (float(np.mean(accs)) if accs else None,
                float(np.mean([m[1] for m in hit])) if hit and reached else None,
                float(np.mean([m[2] for m in hit])) if hit and reached else None)
    if run is None or len(run["acc"]) == 0:
        return (None, None, None)
    mask = run["rounds"] >= 1
    r, a, w = run["rounds"][mask], run["acc"][mask], run["wall"][mask]
    if len(a) == 0:   # 첫 집계 전에 끝난 실행 (0라운드 평가만 있음)
        return (None, None, None)
    target = TARGET_ACC[dataset]
    hit = np.nonzero(a >= target)[0]
    if len(hit) == 0:
        return (float(a.max()), None, None)
    i = hit[0]
    return (float(a.max()), int(r[i]), float(w[i]))


def wall_at_round(run, rnd):
    if run is not None and "reps" in run:
        v = [wall_at_round(r, rnd) for r in run["reps"]]
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else None
    if run is None:
        return None
    idx = np.nonzero(run["rounds"] >= rnd)[0]
    return float(run["wall"][idx[0]]) if len(idx) else None


def status_line(run):
    if run is None:
        return "미실행"
    tag = "완료" if run["complete"] else f"부분 결과 ({run['result'].get('outcome', 'running')})"
    return f"{tag}: {run['rounds_done']}/{run['requested']}라운드"


# ------------------------------------------------------------------ 그림 공통
def _subsample(run, every):
    keep = (run["rounds"] % every == 0)
    return run["rounds"][keep], run["acc"][keep]


def compare_with_paper(paper_img, our_img, out_path, title):
    """왼쪽 논문 원본, 오른쪽 재현 결과를 한 장으로 붙인다."""
    left, right = Image.open(paper_img).convert("RGB"), Image.open(our_img).convert("RGB")
    h = 900
    left = left.resize((int(left.width * h / left.height), h))
    right = right.resize((int(right.width * h / right.height), h))
    head = 110
    canvas = Image.new("RGB", (left.width + right.width + 60, h + head + 20), "white")
    canvas.paste(left, (20, head))
    canvas.paste(right, (left.width + 40, head))
    d = ImageDraw.Draw(canvas)
    d.text((20, 15), title, fill="black", font=ImageFont.truetype(FONT_B, 38))
    f = ImageFont.truetype(FONT_R, 30)
    d.text((20, 68), "논문 원본 (Yu et al., 2024)", fill="#444444", font=f)
    d.text((left.width + 40, 68), "재현 (K=30, wise-175 단일 노드, 1회)", fill="#444444", font=f)
    d.line([(left.width + 30, head), (left.width + 30, head + h)], fill="#bbbbbb", width=3)
    canvas.save(out_path)


def _legend_top(ax, ncol=3):
    # 논문처럼 가로 방향(행 우선)으로 읽히게: matplotlib은 열 우선으로 채우므로 순서를 바꾼다
    handles, labels = ax.get_legend_handles_labels()
    rows = -(-len(handles) // ncol)
    order = [r * ncol + c for c in range(ncol) for r in range(rows) if r * ncol + c < len(handles)]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, frameon=False,
              handlelength=3)


# ------------------------------------------------------------------ Fig. 7 / Fig. 8 형식
def draw_bsa_vs_sasafl(ax, runs, every, ylim, xmax, inset=None):
    """논문 Fig.7/8 한 패널: BSA 초록, SASAFL 빨강, C별 선 스타일."""
    for method, color in (("BSA", GREEN), ("SASAFL", RED)):
        for c in (0.1, 0.3, 0.5):
            run = runs.get((method, c))
            if run is None or len(run["acc"]) == 0:
                ax.plot([], [], color=color, ls=C_STYLE[c], label=f"{method}, C={c}")
                continue
            x, y = _subsample(run, every)
            ax.plot(x, y, color=color, ls=C_STYLE[c], lw=1.4, label=f"{method}, C={c}")
    ax.set_xlim(0, xmax)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Rounds (#)")
    ax.set_ylabel("Global model accuracy (%)")
    ax.grid(axis="y", color="#dddddd")
    _legend_top(ax)
    if inset:
        (x0, x1), (y0, y1) = inset
        ins = ax.inset_axes([0.55, 0.12, 0.4, 0.38])
        for (method, c), run in runs.items():
            if run is None or len(run["acc"]) == 0:
                continue
            m = (run["rounds"] >= x0) & (run["rounds"] <= x1)
            ins.plot(run["rounds"][m], run["acc"][m], color=GREEN if method == "BSA" else RED,
                     ls=C_STYLE[c], lw=1.1)
        ins.set_xlim(x0, x1)
        ins.set_ylim(y0, y1)
        ins.tick_params(labelsize=8)
        ax.indicate_inset_zoom(ins, edgecolor="black")


def draw_unavailable(ax, text):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#bbbbbb")


def fig_bsa_vs_sasafl(runs, out_path, every, ylim, xmax, inset=None, title=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    draw_bsa_vs_sasafl(ax, runs, every, ylim, xmax, inset)
    if title:
        ax.set_title(title, fontsize=12, pad=46)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig7_grid(mnist_runs, fmnist_runs, out_path, mnist_k10=None, fmnist_k10=None):
    """논문 Fig.7과 같은 2×2 배치. 논문의 (K=30, K=100) 두 열에 재현의 (K=10, K=30)을 대응시킨다 (작은 K가 왼쪽)."""
    with plt.rc_context({"font.size": 15, "legend.fontsize": 12}):
        _fig7_grid(mnist_runs, fmnist_runs, out_path, mnist_k10 or {}, fmnist_k10 or {})


def _fig7_grid(mnist_runs, fmnist_runs, out_path, mnist_k10, fmnist_k10):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.6))
    draw_bsa_vs_sasafl(axes[0][0], mnist_k10, 5, (0, 100), 500, inset=((450, 500), (80, 100)))
    draw_bsa_vs_sasafl(axes[0][1], mnist_runs, 5, (0, 100), 500, inset=((450, 500), (80, 100)))
    draw_bsa_vs_sasafl(axes[1][0], fmnist_k10, 5, (0, 100), 500)
    draw_bsa_vs_sasafl(axes[1][1], fmnist_runs, 5, (0, 100), 500)
    caps = ["(a) MNIST with K=10", "(b) MNIST with K=30", "(c) FashionMNIST with K=10", "(d) FashionMNIST with K=30"]
    for ax, cap in zip(axes.flat, caps):
        ax.text(0.5, -0.3, cap, transform=ax.transAxes, ha="center", fontsize=18)
    fig.tight_layout(h_pad=4)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig8_grid(cifar_runs, out_path, xmax, cifar_k10=None):
    with plt.rc_context({"font.size": 15, "legend.fontsize": 12}):
        _fig8_grid(cifar_runs, out_path, xmax, cifar_k10 or {})


def _fig8_grid(cifar_runs, out_path, xmax, cifar_k10):
    """논문 Fig.8 배치. 논문의 (a) K=30, (b) K=100 자리에 재현의 (a) K=10, (b) K=30을 대응시킨다."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    draw_bsa_vs_sasafl(axes[0], cifar_k10, 10, (0, 60), xmax)
    draw_bsa_vs_sasafl(axes[1], cifar_runs, 10, (0, 60), xmax)
    for ax, cap in zip(axes, ["(a) K=10", "(b) K=30"]):
        ax.text(0.5, -0.32, cap, transform=ax.transAxes, ha="center", fontsize=18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 9 형식
def fig_sasafl_c(runs, out_path, ylim=(30, 60), xmax=3000):
    colors = {0.1: "#ffc000", 0.3: "#00b0f0", 0.5: "#b07fd6"}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for c in (0.1, 0.3, 0.5):
        run = runs.get(("SASAFL", c))
        if run is None or len(run["acc"]) == 0:
            ax.plot([], [], color=colors[c], label=f"SASAFL, C={c}")
            continue
        ax.plot(run["rounds"], run["acc"], color=colors[c], lw=0.8, label=f"SASAFL, C={c}")
    ax.set_xlim(0, xmax)
    vals = [v for c in (0.1, 0.3, 0.5) for v in (runs.get(("SASAFL", c)) or {"acc": []})["acc"]]
    lo = min(ylim[0], 10 * np.floor(min(vals) / 10)) if len(vals) else ylim[0]
    ax.set_ylim(lo, ylim[1])
    ax.set_xlabel("Rounds (#)")
    ax.set_ylabel("Global model accuracy (%)")
    ax.grid(axis="y", color="#dddddd")
    _legend_top(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 3 형식
def fig3(runs, out_path):
    """BSA, 30 clients, picked 10/30/50%: (a) 정확도 0–500, (b) loss 400–500."""
    styles = {0.1: (":", "picked 10%"), 0.3: ("--", "picked 30%"), 0.5: ("-", "picked 50%")}
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    for c, (ls, label) in styles.items():
        run = runs.get(("BSA", c))
        if run is None or len(run["acc"]) == 0:
            a.plot([], [], color="black", ls=ls, label=label + " (미실행)")
            b.plot([], [], color="black", ls=ls, label=label + " (미실행)")
            continue
        a.plot(run["rounds"], run["acc"], color="black", ls=ls, lw=1.2, label=label)
        m = (run["rounds"] >= 400) & (run["rounds"] <= 500)
        b.plot(run["rounds"][m], run["loss"][m], color="black", ls=ls, lw=1.2, label=label)
    a.set(xlim=(0, 500), ylim=(0, 100), xlabel="Rounds (#)", ylabel="Global model accuracy (%)")
    b.set(xlim=(400, 500), ylim=(0, 0.8), xlabel="Rounds (#)", ylabel="Global model loss")
    for ax, cap in ((a, "(a) Global model accuracy trace"), (b, "(b) Global model loss trace")):
        _legend_top(ax)
        ax.text(0.5, -0.27, cap, transform=ax.transAxes, ha="center", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 2 형식
def fig2(sync, asyn, semi, out_path):
    spec = [(sync, "#0000ff", "Synchronous"), (asyn, "black", "Asynchronous"), (semi, RED, "Semi-asynchronous")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    for run, color, label in spec:
        if run is None or len(run["acc"]) == 0:
            a.plot([], [], color=color, label=label + " (미실행)")
            b.plot([], [], color=color, label=label + " (미실행)")
            continue
        a.plot(run["rounds"], run["acc"], color=color, lw=1.2, label=label)
        b.plot(run["rounds"], run["wall"], color=color, lw=1.4, label=label)
    a.set(xlim=(0, 500), ylim=(0, 100), xlabel="Rounds (#)", ylabel="Global model accuracy (%)")
    b.set(xlim=(0, 500), ylim=(0, None), xlabel="Rounds (#)", ylabel="Wall clock time (m)")
    for ax, cap in ((a, "(a) Accuracy"), (b, "(b) Latency")):
        ax.legend(loc="lower right" if ax is a else "upper left", frameon=False, fontsize=10.5)
        ax.text(0.5, -0.27, cap, transform=ax.transAxes, ha="center", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 5 형식
def fig5(acc_by_tau, out_path):
    """acc_by_tau: {tau: [최고 정확도, ...]} → 박스플롯 (평균은 x 표시)."""
    taus = list(range(1, 11))
    data = [acc_by_tau.get(t, []) for t in taus]
    fig, ax = plt.subplots(figsize=(7.2, 5))
    hatches = ["", "////", "....", "\\\\\\\\", "xxxx", "||||", "----", "++++", "oo", "**"]
    positions = [t for t, d in zip(taus, data) if d]
    if positions:
        bp = ax.boxplot([d for d in data if d], positions=positions, widths=0.7, patch_artist=True,
                        showmeans=True, meanprops={"marker": "x", "markeredgecolor": "black", "markersize": 7},
                        medianprops={"color": "black"})
        for patch, t in zip(bp["boxes"], positions):
            patch.set(facecolor="white", hatch=hatches[t - 1], edgecolor="black")
    ax.set_xticks(taus)
    ax.set_xticklabels([])
    ax.set_xlim(0.3, 10.7)
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor="white", edgecolor="black", hatch=h) for h in hatches], [str(t) for t in taus],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=10, frameon=False,
              handlelength=1.0, handletextpad=0.3, columnspacing=0.8)
    lo = min([min(d) for d in data if d], default=95)
    ax.set_ylim(min(95, np.floor(lo)), 99)
    ax.set_xlabel("Lag tolerance")
    ax.set_ylabel("Global model accuracy (%)")
    ax.grid(axis="y", color="#dddddd")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 11 형식 (K=30 부분)
def fig11(wall, paper_wall, out_path):
    """wall/paper_wall: {(method, C): 분}. 논문 Fig.11의 K=30 묶음과 재현값을 나란히."""
    hatches = {0.1: "", 0.3: "\\\\\\\\", 0.5: "++"}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    groups = [("논문 K=30", paper_wall), ("재현 K=30", wall)]
    width = 0.12
    for g, (gname, vals) in enumerate(groups):
        k = 0
        for method, color in (("BSA", GREEN), ("SASAFL", RED)):
            for c in (0.1, 0.3, 0.5):
                v = vals.get((method, c))
                x = g + (k - 2.5) * width
                if v is not None:
                    ax.bar(x, v, width * 0.9, color="white" if hatches[c] else color, edgecolor=color,
                           hatch=hatches[c], linewidth=1.5, label=f"{method}, C={c}" if g == 0 else None)
                    ax.text(x, v + 3, f"{v:.0f}", ha="center", fontsize=8)
                k += 1
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("Wall clock time (m)")
    ax.grid(axis="y", color="#dddddd")
    _legend_top(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ 표 (Table 2 형식)
def _fmt(v, kind):
    if v is None:
        return "-"
    return f"{int(v):,}" if kind == "rta" else f"{v:,.2f}"


def _better(a, b, higher):
    if a is None and b is None:
        return None
    if a is None:
        return 1
    if b is None:
        return 0
    if a == b:
        return None
    return (0 if a > b else 1) if higher else (0 if a < b else 1)


def table2_rows(dataset, paper, ours):
    """paper/ours: {C: {method: (acc, rta, tta)}} -> [(label, cells[4], bold_idx)]"""
    rows = []
    for c in (0.1, 0.3, 0.5):
        rows.append(("C", f"{c}", set()))
        for label, idx, kind, higher in (("Accuracy", 0, "acc", True), ("Round-to-accuracy", 1, "rta", False),
                                         ("Time-to-accuracy", 2, "tta", False)):
            p = [paper[c][m][idx] for m in ("BSA", "SASAFL")]
            o = [ours[c][m][idx] for m in ("BSA", "SASAFL")]
            bold = set()
            w = _better(p[0], p[1], higher)
            if w is not None:
                bold.add(w)
            w = _better(o[0], o[1], higher)
            if w is not None:
                bold.add(2 + w)
            rows.append((label, [_fmt(p[0], kind), _fmt(p[1], kind), _fmt(o[0], kind), _fmt(o[1], kind)], bold))
    return rows


def table2_markdown(dataset, rows, notes):
    target = TARGET_ACC[dataset]
    out = [f"| {dataset} (목표 {target:.0f}%) | 논문 BSA | 논문 SASAFL | 재현 BSA | 재현 SASAFL |",
           "|---|---:|---:|---:|---:|"]
    for label, cells, bold in rows:
        if label == "C":
            out.append(f"| **C = {cells}** | | | | |")
        else:
            out.append(f"| {label} | " + " | ".join(f"**{v}**" if i in bold else v for i, v in enumerate(cells)) + " |")
    if notes:
        out += ["", "주의:"] + [f"- {n}" for n in notes]
    return "\n".join(out)


def table2_png(dataset, rows, notes, out_path, subtitle):
    """논문 Table 2 모양(가로줄만 있는 표)으로 그린다."""
    n = len(rows)
    fig, ax = plt.subplots(figsize=(10.5, 0.5 * (n + 6) + 0.35 * len(notes)))
    ax.axis("off")
    top = n + 5.5 + 0.7 * len(notes)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, top)
    xs = [0.2, 4.9, 6.4, 8.2, 9.7]
    y = top - 0.5
    ax.text(0.2, y, f"Table 2 비교 — {dataset}, K=30 (목표 정확도 {TARGET_ACC[dataset]:.0f}%)",
            fontsize=15, fontweight="bold")
    y -= 0.6
    ax.text(0.2, y, subtitle, fontsize=10.5, color="#555555")
    y -= 0.5
    ax.hlines(y, 0.1, 9.9, color="black", lw=1.4)
    y -= 0.55
    ax.text(0.2, y, f"Dataset: {dataset}", fontsize=12)
    ax.text((xs[1] + xs[2]) / 2 - 0.3, y, "논문 (Yu et al., 2024)", ha="center", fontsize=12)
    ax.text((xs[3] + xs[4]) / 2 - 0.3, y, "재현 (본 실험)", ha="center", fontsize=12)
    ax.hlines(y - 0.3, xs[1] - 1.2, xs[2] + 0.1, color="black", lw=0.8)
    ax.hlines(y - 0.3, xs[3] - 1.2, xs[4] + 0.1, color="black", lw=0.8)
    y -= 0.7
    ax.text(0.2, y, "Method", fontsize=12)
    for x, h in zip(xs[1:], ["BSA", "SASAFL", "BSA", "SASAFL"]):
        ax.text(x, y, h, ha="right", fontsize=12)
    y -= 0.35
    ax.hlines(y, 0.1, 9.9, color="black", lw=0.8)
    for i, (label, cells, bold) in enumerate(rows):
        y -= 0.5
        if label == "C":
            ax.text(0.2, y, "C", fontsize=12)
            ax.text(xs[1] - 1.2, y, cells, fontsize=12)
            y -= 0.25
            ax.hlines(y, 0.1, 9.9, color="black", lw=0.6)
            continue
        ax.text(0.2, y, label, fontsize=12)
        for j, v in enumerate(cells):
            ax.text(xs[j + 1], y, v, ha="right", fontsize=12, fontweight="bold" if j in bold else "normal")
        if label == "Time-to-accuracy":
            y -= 0.25
            ax.hlines(y, 0.1, 9.9, color="black", lw=1.4 if i == n - 1 else 0.8)
    for k, note in enumerate(notes):
        y -= 0.5 if k else 0.7
        ax.text(0.2, y, ("주의: " if k == 0 else "         ") + note, fontsize=9.5, color="#d03b3b")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def simple_table_png(title, header, rows, out_path, subtitle=None):
    """가로줄 표 (Table 4 등)."""
    fig, ax = plt.subplots(figsize=(1.6 * len(header) + 1, 0.5 * (len(rows) + 4)))
    ax.axis("off")
    top = len(rows) + 3.2
    ax.set_xlim(0, len(header))
    ax.set_ylim(0, top)
    y = top - 0.4
    ax.text(0, y, title, fontsize=14, fontweight="bold")
    if subtitle:
        y -= 0.55
        ax.text(0, y, subtitle, fontsize=10, color="#555555")
    y -= 0.45
    ax.hlines(y, 0, len(header), color="black", lw=1.4)
    y -= 0.5
    for j, h in enumerate(header):
        ax.text(j + 0.05, y, h, fontsize=11)
    y -= 0.3
    ax.hlines(y, 0, len(header), color="black", lw=0.8)
    for row in rows:
        y -= 0.5
        for j, v in enumerate(row):
            ax.text(j + 0.05, y, str(v), fontsize=11)
    y -= 0.3
    ax.hlines(y, 0, len(header), color="black", lw=1.4)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def table2_full_png(values, title, out_path, note=None):
    """논문 Table 2와 같은 배치. values[dataset][C][method] = (acc, rta, tta)."""
    datasets = [("MNIST", "MNIST"), ("FashionMNIST", "FashionMNIST"), ("CIFAR10", "CIFAR-10")]
    fig, ax = plt.subplots(figsize=(12, 9.2))
    ax.axis("off")
    ax.set_xlim(0, 12)
    top = 21
    ax.set_ylim(0, top)
    xs = [0.3] + [2.9 + 1.55 * i for i in range(6)]       # 열 오른쪽 끝 기준
    y = top - 0.6
    ax.text(0.3, y, title, fontsize=16, fontweight="bold")
    if note:
        ax.text(0.3, y - 0.75, note, fontsize=10.5, color="#555555")
    y -= 1.6
    ax.hlines(y, 0.2, 11.9, color="black", lw=1.4)
    y -= 0.7
    ax.text(xs[0], y, "Dataset", fontsize=13)
    for k, (_, label) in enumerate(datasets):
        x0 = xs[1 + 2 * k] - 1.1
        ax.text(x0, y, label, fontsize=13)
        ax.hlines(y - 0.3, x0, xs[2 + 2 * k] + 0.1, color="black", lw=0.8)
    y -= 0.8
    ax.text(xs[0], y, "Method", fontsize=13)
    for k in range(3):
        ax.text(xs[1 + 2 * k] + 0.45, y, "BSA", ha="right", fontsize=13)
        ax.text(xs[2 + 2 * k] + 0.45, y, "SASAFL", ha="right", fontsize=13)
    y -= 0.35
    ax.hlines(y, 0.2, 11.9, color="black", lw=0.8)
    for ci, c in enumerate((0.1, 0.3, 0.5)):
        y -= 0.65
        ax.text(xs[0], y, "C", fontsize=13)
        ax.text(xs[1] - 1.1, y, f"{c}", fontsize=13)
        y -= 0.3
        ax.hlines(y, 0.2, 11.9, color="black", lw=0.6)
        for label, idx, kind, higher in (("Accuracy", 0, "acc", True), ("Round-to-accuracy", 1, "rta", False),
                                         ("Time-to-accuracy", 2, "tta", False)):
            y -= 0.6
            ax.text(xs[0], y, label, fontsize=13)
            for k, (key, _) in enumerate(datasets):
                pair = [values.get(key, {}).get(c, {}).get(m, (None, None, None))[idx] for m in ("BSA", "SASAFL")]
                w = _better(pair[0], pair[1], higher)
                for j, v in enumerate(pair):
                    ax.text(xs[1 + 2 * k + j] + 0.45, y, _fmt(v, kind), ha="right", fontsize=13,
                            fontweight="bold" if w == j else "normal")
        y -= 0.35
        ax.hlines(y, 0.2, 11.9, color="black", lw=1.4 if ci == 2 else 0.8)
    ax.set_ylim(y - 0.4, top)
    fig.set_size_inches(12, 0.44 * (top - y + 0.4))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def table4_full_png(paper_rows, our_rows, out_path, note=None):
    """논문 Table 4 배치(데이터셋 × C, 9칸). rows: [(라벨, {dataset: {C: 값}})]."""
    datasets = [("MNIST", "MNIST"), ("FashionMNIST", "FashionMNIST"), ("CIFAR10", "CIFAR-10")]
    rows = paper_rows + our_rows
    fig, ax = plt.subplots(figsize=(13, 0.6 * (len(rows) + 5)))
    ax.axis("off")
    top = len(rows) + 4.2
    ax.set_xlim(0, 13)
    ax.set_ylim(0, top)
    y = top - 0.5
    ax.text(0.2, y, "Table 4 비교 — SASAFL wall clock time (분), K=30 행", fontsize=15, fontweight="bold")
    if note:
        y -= 0.6
        ax.text(0.2, y, note, fontsize=10, color="#555555")
    y -= 0.55
    ax.hlines(y, 0.1, 12.9, color="black", lw=1.4)
    y -= 0.55
    ax.text(0.2, y, "Dataset", fontsize=12)
    for k, (_, label) in enumerate(datasets):
        ax.text(3.2 + 3.2 * k, y, label, fontsize=12)
        ax.hlines(y - 0.25, 3.1 + 3.2 * k, 5.9 + 3.2 * k, color="black", lw=0.8)
    y -= 0.6
    ax.text(0.2, y, "C", fontsize=12)
    for k in range(3):
        for j, c in enumerate((0.1, 0.3, 0.5)):
            ax.text(3.2 + 3.2 * k + 0.95 * j, y, f"{c}", fontsize=12)
    y -= 0.3
    ax.hlines(y, 0.1, 12.9, color="black", lw=0.8)
    for label, vals in rows:
        y -= 0.55
        ax.text(0.2, y, label, fontsize=12)
        for k, (key, _) in enumerate(datasets):
            for j, c in enumerate((0.1, 0.3, 0.5)):
                v = vals.get(key, {}).get(c)
                ax.text(3.2 + 3.2 * k + 0.95 * j, y, "-" if v is None else f"{v:,.0f}", fontsize=12)
    y -= 0.3
    ax.hlines(y, 0.1, 12.9, color="black", lw=1.4)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def stack_vertical(paths, out_path, gap=30):
    imgs = [Image.open(p).convert("RGB") for p in paths]
    w = max(i.width for i in imgs)
    canvas = Image.new("RGB", (w, sum(i.height for i in imgs) + gap * (len(imgs) - 1)), "white")
    y = 0
    for img in imgs:
        canvas.paste(img, (0, y))
        y += img.height + gap
    canvas.save(out_path)
