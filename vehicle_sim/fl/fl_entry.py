#!/usr/bin/env python3
"""torchrun이 rank마다 실행하는 진입점: 시뮬레이션 클래스를 끼운 뒤 원본 Start.py를 그대로 실행한다.

환경 변수 SIM_RUN_DIR: world.json / config.json / events.jsonl이 있는 실행 폴더
데이터 분할: config.fl.split이 'dirichlet'이면 라벨 비율을 Dirichlet(α)로 나눈다 (8-2, VEFL·FeLiX의 α=0.1).
             그 외에는 원본 create_dataset(gaussian/uniform) 그대로.
"""

import faulthandler
import os
import runpy
import signal
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
REPO = SIM.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SIM / "simcore"))
sys.path.insert(0, str(HERE))
os.chdir(REPO)
faulthandler.register(signal.SIGUSR1, all_threads=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402
import sim_sasafl  # noqa: E402
from methods.FedAvg import FedAvgServer as M_FEDAVG_SERVER  # noqa: E402
from utils import data_utils  # noqa: E402

sim_sasafl.install()
CFG = C.read_json(Path(os.environ["SIM_RUN_DIR"]) / "config.json", C.DEFAULT_CONFIG)
_original_create_dataset = data_utils.create_dataset


def create_dataset_sim(num_clients, dataset_name, iid, split):
    if CFG["fl"].get("split") != "dirichlet":
        return _original_create_dataset(num_clients, dataset_name, iid, split)
    alpha = float(CFG["fl"].get("alpha", 0.1))
    datasets, test_dataset = _original_create_dataset(num_clients, dataset_name, "True", "uniform")  # 전체를 한 번 모은다
    data = torch.cat([d.tensors[0] for d in datasets])
    labels = torch.cat([d.tensors[1] for d in datasets]).long()
    transform = datasets[0].transform
    rng = np.random.default_rng()
    parts = [[] for _ in range(num_clients)]
    for c in range(int(labels.max()) + 1):
        idx = np.nonzero(labels.numpy() == c)[0]
        rng.shuffle(idx)
        share = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(share) * len(idx)).astype(int)[:-1]
        for k, chunk in enumerate(np.split(idx, cuts)):
            parts[k].extend(chunk.tolist())
    out = []
    for k in range(num_clients):
        if len(parts[k]) < 64:                       # 너무 작은 차량은 최소 64개 보장
            parts[k].extend(rng.choice(len(labels), 64 - len(parts[k]), replace=False).tolist())
        ii = torch.tensor(parts[k])
        out.append(data_utils.CustomTensorDataset((data[ii], labels[ii]), transform=transform))
    C.emit(os.environ["SIM_RUN_DIR"], "fl", example="8-2",
           msg=f"Dirichlet α={alpha}로 라벨 분할 (차량별 데이터 수 {[len(p) for p in parts]})")
    return out, test_dataset


M_FEDAVG_SERVER.create_dataset = create_dataset_sim

sys.argv = [str(REPO / "Start.py")] + sys.argv[1:]
runpy.run_path(str(REPO / "Start.py"), run_name="__main__")
