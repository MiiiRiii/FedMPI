"""논문 그림·표와 같은 형식의 재현 그림을 만들고, 왼쪽 논문 원본 / 오른쪽 재현으로 붙인다.

그림 안에는 논문과 같은 영어 축·범례·표 머리만 넣는다 (덧붙이는 설명 문구 없음).
출력 (각 실험 폴더와 E6에):
  compare_fig5.png  compare_fig7.png  compare_fig8.png  compare_fig9.png  compare_fig11.png
  compare_table2.png  compare_table4.png
  reproduced/*.png  (오른쪽에 들어가는 재현 그림 단독)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

import analysis as A  # noqa: E402
import paper_values as P  # noqa: E402

PAPER = Path(__file__).resolve().parents[1] / "paper_figures"
SERIF = {"font.family": ["Liberation Serif"], "font.size": 13}

# 이전 형식(한국어 포함)으로 만들어진 파일들 — 새로 만들기 전에 지운다
OLD_FILES = ["table2_comparison.png", "table4_k30.png", "fig7a_mnist_k30.png", "fig7c_fmnist_k30.png",
             "fig11_fmnist_wallclock_k30.png", "fig8a_cifar10_k30.png", "fig8_layout_as_paper.png",
             "fig9_cifar10_sasafl_k30.png", "fig5_lag_tolerance.png", "table2_paper.png", "table2_reproduced.png",
             "fig7_layout_as_paper.png", "compare_fig2.png", "compare_fig3.png"]


def side_by_side(left_path, right_path, out_path, height=1000, gap=40):
    """왼쪽 논문 원본, 오른쪽 재현. 텍스트 없이 높이만 맞춘다."""
    row([left_path, right_path], out_path, height, gap)


def row(paths, out_path, height=1000, gap=40, labels=None):
    """그림들을 왼쪽부터 차례로 가로로 붙인다 (높이만 맞춤).
    labels가 있으면 각 그림 위에 한 줄짜리 머리표(영어)를 단다 — err 비교 그림에서 열의 의미를 밝히는 용도."""
    from PIL import ImageDraw, ImageFont
    imgs = [Image.open(p).convert("RGB") for p in paths]
    imgs = [im.resize((int(im.width * height / im.height), height), Image.LANCZOS) for im in imgs]
    head = 170 if labels else 0
    canvas = Image.new("RGB", (sum(im.width for im in imgs) + gap * (len(imgs) + 1), height + gap * 2 + head), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(LABEL_FONT, 46) if labels else None
    x = gap
    for i, im in enumerate(imgs):
        canvas.paste(im, (x, gap + head))
        if labels:
            draw.rectangle([x, gap, x + im.width, gap + head - 20], fill=LABEL_BG[i % len(LABEL_BG)])
            draw.multiline_text((x + im.width // 2, gap + (head - 20) // 2), labels[i], fill="black", font=font,
                                anchor="mm", align="center", spacing=10)
        x += im.width + gap
    canvas.save(out_path)


LABEL_FONT = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"
LABEL_BG = ["#e6e6e6", "#d7ecd9", "#f6d6d6"]   # 논문(회색) · 적용(초록) · 미적용(빨강)
ERR_LABELS = ["Paper (Yu et al., 2024)",
              "Reproduction: adaptive selection ON\n(current code, quota adapts to global loss)",
              "Reproduction: adaptive selection OFF\n(original code c0b819b, quota fixed at C·K)"]


# ------------------------------------------------------------------ 재현 그림 (논문 형식)
def fig11_reproduced(wall, out_path, wall_k10=None, reps=None, reps_k10=None):
    """논문 Fig.11 형식: FashionMNIST 500라운드 wall clock. 논문의 (K=30, K=100) 묶음 자리에 재현의 (K=10, K=30)."""
    hatches = {0.1: "", 0.3: "\\\\\\\\", 0.5: "++"}
    with plt.rc_context(SERIF):
        fig, ax = plt.subplots(figsize=(8, 5))
        width = 0.12
        top = max([v for d in (wall, wall_k10 or {}) for v in d.values() if v is not None], default=1)
        for g, (vals, ns) in enumerate(((wall_k10 or {}, reps_k10 or {}), (wall, reps or {}))):
            k = 0
            for method, color in (("BSA", A.GREEN), ("SASAFL", A.RED)):
                for c in (0.1, 0.3, 0.5):
                    v = vals.get((method, c))
                    x = g + (k - 2.5) * width
                    ax.bar(x, v if v is not None else 0, width * 0.9, color="white" if hatches[c] else color,
                           edgecolor=color, hatch=hatches[c], linewidth=1.5, label=f"{method}, C={c}" if g == 0 else None)
                    # 막대 위: 재현 반복 수/논문 반복 수
                    ax.text(x, (v or 0) + 0.01 * top, f"{ns.get((method, c), 0)}/{A.PAPER_REPS}", ha="center",
                            va="bottom", fontsize=6.5, rotation=90)
                    k += 1
        ax.set_ylim(0, top * 1.12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["K=10", "K=30"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel("Wall clock time (m)")
        ax.grid(axis="y", color="#dddddd")
        A._legend_top(ax, ncol=3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def table2_reproduced(values, out_path, reps=None):
    with plt.rc_context(SERIF):
        A.table2_full_png(values, "Table 2", out_path, note=None, reps=reps)


def table4_reproduced(row, rounds, out_path, row_k10=None, reps=None, reps_k10=None):
    """논문 Table 4 형식 (Dataset × C). row/row_k10: {dataset: {C: 분}} → K=30, K=10 행."""
    datasets = [("MNIST", "MNIST"), ("FashionMNIST", "FashionMNIST"), ("CIFAR10", "CIFAR-10")]
    with plt.rc_context(SERIF):
        fig, ax = plt.subplots(figsize=(13, 3.8))
        ax.axis("off")
        ax.set_xlim(0, 13)
        ax.set_ylim(0, 4.2)
        y = 3.9
        ax.text(0.2, y, "Table 4", fontsize=15, fontweight="bold")
        y -= 0.5
        ax.hlines(y, 0.1, 12.9, color="black", lw=1.4)
        y -= 0.55
        ax.text(0.2, y, "Dataset", fontsize=13)
        for k, (_, label) in enumerate(datasets):
            ax.text(3.2 + 3.2 * k, y, label, fontsize=13)
            ax.hlines(y - 0.25, 3.1 + 3.2 * k, 5.9 + 3.2 * k, color="black", lw=0.8)
        y -= 0.6
        ax.text(0.2, y, "C", fontsize=13)
        for k in range(3):
            for j, c in enumerate((0.1, 0.3, 0.5)):
                ax.text(3.2 + 3.2 * k + 0.95 * j, y, f"{c}", fontsize=13)
        y -= 0.3
        ax.hlines(y, 0.1, 12.9, color="black", lw=0.8)
        for label, vals, ns in (("K=10", row_k10 or {}, reps_k10), ("K=30", row, reps)):
            y -= 0.55
            ax.text(0.2, y, label, fontsize=13)
            for k, (key, _) in enumerate(datasets):
                for j, c in enumerate((0.1, 0.3, 0.5)):
                    v = vals.get(key, {}).get(c)
                    ax.text(3.2 + 3.2 * k + 0.95 * j, y, "-" if v is None else f"{v:,.0f}", fontsize=13)
            if ns is not None:   # 재현 반복 수/논문 반복 수
                y -= 0.45
                ax.text(0.5, y, "Repetitions", fontsize=11, color="#444444")
                for k, (key, _) in enumerate(datasets):
                    for j, c in enumerate((0.1, 0.3, 0.5)):
                        ax.text(3.2 + 3.2 * k + 0.95 * j, y, f"{ns.get(key, {}).get(c, 0)}/{A.PAPER_REPS}",
                                fontsize=11, color="#444444")
        y -= 0.3
        ax.hlines(y, 0.1, 12.9, color="black", lw=1.4)
        ax.set_ylim(y - 0.2, 4.2)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


# ------------------------------------------------------------------ 전체 생성
def build_all(mcs, e5_acc_by_tau, exp_dirs, max_rounds, mcs_k10=None):
    """mcs: {'MNIST': mc, 'FashionMNIST': mc, 'CIFAR10': mc} (mc = {(method, C): run}) — K=30
    mcs_k10: 같은 형식, K=10 (현재 코드 SASAFL). 논문의 K=30 칸 자리에, K=30 재현은 논문의 K=100 칸 자리에 그린다.
    exp_dirs: {'E1'..'E8': Path}"""
    mcs_k10 = mcs_k10 or {ds: {} for ds in ("MNIST", "FashionMNIST", "CIFAR10")}
    for d in exp_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        (d / "reproduced").mkdir(exist_ok=True)
        for name in OLD_FILES:
            (d / name).unlink(missing_ok=True)

    rep = exp_dirs["E6"] / "reproduced"
    # Fig.7 (2×2) / Fig.8: 재현 K=10 → 논문 K=30 칸, 재현 K=30 → 논문 K=100 칸
    with plt.rc_context(SERIF):
        A.fig7_grid(mcs["MNIST"], mcs["FashionMNIST"], rep / "fig7.png", mcs_k10["MNIST"], mcs_k10["FashionMNIST"])
        A.fig8_grid(mcs["CIFAR10"], rep / "fig8.png", xmax=max_rounds, cifar_k10=mcs_k10["CIFAR10"])
        A.fig_sasafl_c(mcs["CIFAR10"], rep / "fig9.png", xmax=max_rounds)
        A.fig5(e5_acc_by_tau, rep / "fig5.png")
    wall = {k: A.wall_at_round(v, 500) for k, v in mcs["FashionMNIST"].items()}
    wall_k10 = {k: A.wall_at_round(v, 500) for k, v in mcs_k10["FashionMNIST"].items()}
    reps11 = {k: A.n_reps(v) for k, v in mcs["FashionMNIST"].items()}
    reps11_k10 = {k: A.n_reps(v) for k, v in mcs_k10["FashionMNIST"].items()}
    fig11_reproduced(wall, rep / "fig11.png", wall_k10, reps11, reps11_k10)
    ours = {ds: {c: {m: A.metrics(mcs[ds].get((m, c)), ds) for m in ("BSA", "SASAFL")} for c in (0.1, 0.3, 0.5)}
            for ds in ("MNIST", "FashionMNIST", "CIFAR10")}
    reps2 = {ds: {c: {m: A.n_reps(mcs[ds].get((m, c))) for m in ("BSA", "SASAFL")} for c in (0.1, 0.3, 0.5)}
             for ds in ("MNIST", "FashionMNIST", "CIFAR10")}
    table2_reproduced(ours, rep / "table2.png", reps2)
    t4 = {ds: {c: A.wall_at_round(mcs[ds].get(("SASAFL", c)), min(P.TABLE4_K30[ds]["rounds"], max_rounds))
               for c in (0.1, 0.3, 0.5)} for ds in ("MNIST", "FashionMNIST", "CIFAR10")}
    t4_k10 = {ds: {c: A.wall_at_round(mcs_k10[ds].get(("SASAFL", c)), min(P.TABLE4_K30[ds]["rounds"], max_rounds))
                   for c in (0.1, 0.3, 0.5)} for ds in ("MNIST", "FashionMNIST", "CIFAR10")}
    reps4 = {ds: {c: A.n_reps(mcs[ds].get(("SASAFL", c))) for c in (0.1, 0.3, 0.5)} for ds in ("MNIST", "FashionMNIST", "CIFAR10")}
    reps4_k10 = {ds: {c: A.n_reps(mcs_k10[ds].get(("SASAFL", c))) for c in (0.1, 0.3, 0.5)}
                 for ds in ("MNIST", "FashionMNIST", "CIFAR10")}
    table4_reproduced(t4, None, rep / "table4.png", t4_k10, reps4, reps4_k10)

    pairs = {"fig5": "paper_fig5.jpg", "fig7": "paper_fig7.jpg", "fig8": "paper_fig8.jpg", "fig9": "paper_fig9.jpg",
             "fig11": "paper_fig11.jpg", "table2": "paper_table2.png", "table4": "paper_table4.png"}
    where = {"E1": ["fig7", "table2", "table4"], "E2": ["fig7", "fig11", "table2", "table4"],
             "E4": ["fig8", "fig9", "table2", "table4"], "E5": ["fig5"],
             "E6": ["fig5", "fig7", "fig8", "fig9", "fig11", "table2", "table4"],
             "E8": ["fig7", "fig8", "fig11", "table4"]}
    made = {}
    for eid, names in where.items():
        d = exp_dirs[eid]
        for name in names:
            src = rep / f"{name}.png"
            if eid != "E6":
                (d / "reproduced" / f"{name}.png").write_bytes(src.read_bytes())
            side_by_side(PAPER / pairs[name], src, d / f"compare_{name}.png")
            made.setdefault(eid, []).append(f"compare_{name}.png")
    return made


def build_err(mcs, err_sasafl, exp_dirs, max_rounds, mcs_k10_err=None):
    """적응형 quota가 꺼진 원저자 코드(c0b819b)로 돌린 SASAFL을 기존 비교 그림 오른쪽에 붙인다.
    왼쪽부터: 논문 원본 / 재현(현재 코드, quota 적응 동작) / 재현(원저자 코드, quota 고정).
    err_sasafl: {'MNIST': {C: run}, 'FashionMNIST': {...}, 'CIFAR10': {...}}. BSA는 기존 실행을 그대로 쓴다."""
    rep = exp_dirs["E6"] / "reproduced"
    rep.mkdir(parents=True, exist_ok=True)
    err = {ds: {**{k: v for k, v in mcs[ds].items() if k[0] == "BSA"},
                **{("SASAFL", c): err_sasafl.get(ds, {}).get(c) for c in (0.1, 0.3, 0.5)}}
           for ds in ("MNIST", "FashionMNIST", "CIFAR10")}
    k10 = mcs_k10_err or {ds: {} for ds in ("MNIST", "FashionMNIST", "CIFAR10")}
    with plt.rc_context(SERIF):
        A.fig7_grid(err["MNIST"], err["FashionMNIST"], rep / "fig7_err.png", k10["MNIST"], k10["FashionMNIST"])
        A.fig_sasafl_c(err["CIFAR10"], rep / "fig9_err.png", xmax=max_rounds)
    made = {}
    for eid in ("E7", "E6", "E8"):
        d = exp_dirs[eid]
        (d / "reproduced").mkdir(parents=True, exist_ok=True)
        for name, paper in (("fig7", "paper_fig7.jpg"), ("fig9", "paper_fig9.jpg")):
            if eid != "E6":
                (d / "reproduced" / f"{name}_err.png").write_bytes((rep / f"{name}_err.png").read_bytes())
            row([PAPER / paper, rep / f"{name}.png", rep / f"{name}_err.png"], d / f"compare_{name}_err.png",
                labels=ERR_LABELS)
            made.setdefault(eid, []).append(f"compare_{name}_err.png")
    return made
