"""SASAFL 원본 파일을 고치지 않고, 서브클래스로 '차량 환경'을 끼워 넣는다.

원본 메서드의 동작은 그대로 옮겨 두고(장애가 없으면 원본과 같은 순서·같은 통신), 사이사이에 다음을 넣었다.
  - 주행 환경(world.json)과 설정(config.json)을 읽어 학습 중단·건너뛰기·지연·전송 끊김을 결정
  - 차량 상태(client_<id>.json), 서버 상태(server.json), 사건(events.jsonl)을 화면용으로 기록
원본과 다른 점 (화면을 위해 바꾼 것)
  - 원본 runClient/wait_until_can_update_global_model은 쉬지 않고 도는 busy-wait이라 CPU를 다 쓴다 → 짧게 sleep
  - 'rejoin'(설정 fl.rejoin): 원본 SASAFL은 한 번 빠진 차량이 다시 선택되지 못한다(선택 = 직전 집계 참여).
    켜면 차량이 연결 회복 후 마지막 모델로 스스로 다시 학습해 보낸다. 끄면 원본 그대로 영구 제외.
  - 업로드: 원본은 dist.send 한 번. 여기서는 전송 시간(모델 크기/대역폭)을 조각 단위로 흘려보내며 연결을 확인하고,
    끊기면 실제 dist.send를 하지 않는다(서버는 아무것도 받지 못함 = SASAFL이 실제로 보는 모습).
"""

import copy
import fcntl
import os
import random
import threading
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader

import common as C
from methods.SASAFL import SASAFL as M_SASAFL
from methods.SASAFL import SASAFLClient as M_CLIENT
from methods.SASAFL import SASAFLServer as M_SERVER
from utils.model_utils import TensorBuffer
from utils.utils import printLog

RUN = Path(os.environ["SIM_RUN_DIR"])
ABORT = "ABORT"


class Env:
    """world.json / config.json 읽기 캐시 (프로세스마다 하나)"""

    def __init__(self):
        self._w_t, self._w = 0.0, {}
        self._c_t, self._c, self._c_m = 0.0, None, -1

    def world(self, vid):
        if time.time() - self._w_t > 0.1:
            self._w = C.read_json(RUN / "world.json", self._w) or self._w
            self._w_t = time.time()
        return (self._w.get("vehicles") or {}).get(str(vid), {})

    def world_all(self):
        self.world(0)
        return self._w.get("vehicles") or {}

    def cfg(self):
        if time.time() - self._c_t > 1.0 or self._c is None:
            try:
                m = (RUN / "config.json").stat().st_mtime
            except FileNotFoundError:
                m = 0
            if m != self._c_m or self._c is None:
                self._c = C.read_json(RUN / "config.json", self._c or C.DEFAULT_CONFIG)
                self._c_m = m
            self._c_t = time.time()
        return self._c


ENV = Env()


def link_reason(w):
    """연결이 없는 이유 → 유스케이스 id"""
    if not w.get("present", True):
        return "8-3", "맵 밖"
    if w.get("zones", {}).get("tunnel"):
        return "2-6", "터널"
    if w.get("engine_off"):
        return "1-5", "시동 꺼짐"
    if w.get("ota"):
        return "1-2", "OTA 설치 중"
    if not w.get("gilbert_good", True):
        return "5-3", "음영지역 단절"
    if not w.get("cell"):
        return "2-1", "기지국 범위 밖"
    if w.get("zones", {}).get("security"):
        return "3-4", "보안 게이트웨이 차단"
    return "2-1", "연결 없음"


# ================================================================== 차량 (클라이언트)
class SimClient(M_CLIENT.SASAFLClient):

    # ---------------------------------------------------------------- 공통
    def ev(self, example, msg, **kw):
        C.emit(RUN, "fl", vid=self.id, example=example, msg=msg, **kw)

    def vcfg(self):
        return C.merged_vehicle(ENV.cfg(), self.id)

    def w(self):
        return ENV.world(self.id)

    def setup(self, cluster_type):
        super().setup(cluster_type)
        cfg = ENV.cfg()
        v = self.vcfg()
        self.energy = float(cfg["energy"]["budget_j"])
        self.new_data = float(v.get("min_data", 32)) * 2
        self.ecu = float(v["ecu_factor"]) if v.get("ecu_factor") else random.uniform(1.0, 2.2)   # 4-1 (VEFL 범위)
        self.status, self.detail = "idle", ""
        self.queue = []                    # 보관 후 전송(5-x)용 [텐서, 버전, 만든 시각]
        self.need_rejoin = False
        self.server_version = 0
        self.counters = {k: 0 for k in ("trained", "sent", "dropped", "aborted", "skipped", "queued", "burst")}
        self.wasted_mb = 0.0
        self.skip_active = None
        self.queued_while_off = False
        labels = self.dataset.tensors[1]
        if v.get("label_noise", 0) > 0:      # 7-2 라벨 오류: 시작할 때 일부 라벨을 무작위로 바꾼다
            n = len(labels)
            idx = torch.randperm(n)[: int(n * float(v["label_noise"]))]
            labels = labels.clone()
            labels[idx] = torch.randint(0, 10, (len(idx),))
            self.dataset.tensors = (self.dataset.tensors[0], labels)
            self.ev("7-2", f"라벨 {len(idx)}/{n}개를 틀리게 바꿈")
        hist = torch.bincount(labels.long(), minlength=10).tolist()
        self.label_hist = hist
        self.label_skew = max(hist) / max(1, sum(hist))
        self.model_mb = sum(t.numel() for t in self.model.state_dict().values()) * 4 / 1e6
        threading.Thread(target=self._status_loop, daemon=True).start()

    def _status_loop(self):
        last = time.time()
        while True:
            now = time.time()
            dt, last = now - last, now
            try:
                cfg, w, v = ENV.cfg(), self.w(), self.vcfg()
                if w.get("charging"):
                    self.energy = min(float(cfg["energy"]["budget_j"]),
                                      self.energy + float(cfg["energy"]["charge_rate_jps"]) * dt)
                if w.get("moving"):
                    self.new_data += float(v.get("data_rate_per_s", 5)) * dt
                C.write_json(RUN / f"client_{self.id}.json", {
                    "vid": self.id, "t": now, "status": self.status, "detail": self.detail,
                    "version": self.local_model_version, "server_version": self.server_version,
                    "energy": round(self.energy, 1), "new_data": int(self.new_data), "ecu": round(self.ecu, 2),
                    "queue": len(self.queue), "need_rejoin": self.need_rejoin, "counters": self.counters,
                    "wasted_mb": round(self.wasted_mb, 2), "label_hist": self.label_hist,
                    "label_skew": round(self.label_skew, 2), "n_data": len(self.dataset),
                    "model_mb": round(self.model_mb, 2), "lr": v.get("lr_override") or self.lr})
            except Exception as e:  # 상태 기록 실패가 학습을 멈추지 않게
                printLog(f"CLIENT {self.id}", f"status error {e}")
            time.sleep(0.3)

    def set_status(self, status, detail=""):
        self.status, self.detail = status, detail

    # ---------------------------------------------------------------- 글로벌 모델 수신 (원본 + 6-x 수신 실패)
    def receive_global_model_from_server(self, is_ongoing_local_update_flag, terminate_FL_flag, lag_tolerance):
        self.received_global_model = self.model_controller.Model()
        model_state_dict = self.model.state_dict()
        flatten_model = TensorBuffer(list(model_state_dict.values()))
        self.global_model_info = torch.zeros(len(flatten_model.buffer) + 3)

        while True:
            dist.recv(tensor=self.global_model_info, src=0)
            global_model_version = int(self.global_model_info[-3].item())
            if global_model_version == -1:
                terminate_FL_flag.set()
                break
            am_i_picked = self.global_model_info[-1].item()
            global_loss = self.global_model_info[-2].item()
            self.server_version = global_model_version

            w = self.w()
            if not (w.get("link_up") and w.get("server_ok")):
                # 서버는 모든 차량에 보내지만 이 차량은 받지 못함 (프로세스는 메시지를 비워 서버가 멈추지 않게 한다)
                if am_i_picked == 1 or is_ongoing_local_update_flag.is_set():
                    ex, why = link_reason(w)
                    ex6 = "6-2" if w.get("ota") else "6-1"
                    self.ev(ex6, f"새 모델 v{global_model_version} 수신 실패({why}) → v{self.local_model_version}로 계속",
                            cause=ex)
                    if am_i_picked == 1 and not is_ongoing_local_update_flag.is_set():
                        self.need_rejoin = True      # 원본이라면 이 차량은 다시는 학습을 시작하지 못한다
                continue

            flatten_model.buffer = self.global_model_info[:-3]
            flatten_model.unpack(model_state_dict.values())

            if is_ongoing_local_update_flag.is_set() and global_model_version >= 1:
                if global_model_version - self.local_model_version >= lag_tolerance:
                    self.received_global_model.load_state_dict(model_state_dict)
                    self.replace_global_model_during_local_update.set()
                elif global_loss < self.last_global_loss:
                    self.received_global_model.load_state_dict(model_state_dict)
                    self.replace_global_model_during_local_update.set()
                else:
                    continue
            elif am_i_picked == 1:
                self.model.load_state_dict(model_state_dict)
                self.local_model_version = global_model_version
                self.last_global_loss = global_loss
                self.need_rejoin = False
                is_ongoing_local_update_flag.set()

    # ---------------------------------------------------------------- 참여 조건 (3-x, 8-x): 미달이면 학습을 시작하지 않는다
    def gate(self):
        cfg, w, v = ENV.cfg(), self.w(), self.vcfg()
        if not w.get("present", True):
            return "8-3", "맵 밖(이탈)"
        if v.get("leave"):
            return ("8-2" if self.label_skew > 0.5 else "8-3"), f"차량 이탈 (라벨 쏠림 {self.label_skew:.0%})"
        if not w.get("available", True):
            return "8-1", "가용 시간 아님"
        if w.get("engine_off"):
            return "1-5", "시동 꺼짐"
        if w.get("ota"):
            return "1-2", "OTA 설치 중"
        if self.energy <= 0:
            return "1-3", "배터리 방전 (충전소에서 충전 필요)"
        if v.get("energy_gate", True) and self.energy < float(cfg["energy"]["min_energy_j"]):   # FedQoS식 에너지 확인
            return "3-1", f"잔여 에너지 {self.energy:.0f} J < 기준 {cfg['energy']['min_energy_j']} J"
        if self.new_data < float(v.get("min_data", 32)):
            return "3-2", f"새 데이터 {self.new_data:.0f}개 < {v.get('min_data')}"
        if v.get("charging_only") and not w.get("charging"):
            return "3-3", "충전 중일 때만 참여"
        if w.get("zones", {}).get("security"):
            return "3-4", "보안 게이트웨이가 외부 서버 연결 차단"
        return None

    # ---------------------------------------------------------------- 로컬 학습 (원본 + 중단·지연·오염)
    def train(self, terminate_flag):
        cfg, v = ENV.cfg(), self.vcfg()
        self.set_status("training", f"v{self.local_model_version}")
        start = time.time()
        self.model.train()
        lr = float(v["lr_override"]) if v.get("lr_override") else self.lr
        optimizer = SGD(self.model.parameters(), lr=lr, momentum=0.9)
        loss_function = CrossEntropyLoss()
        dataloader = DataLoader(self.dataset, self.batch_size, shuffle=True)
        start_weights = copy.deepcopy(self.model.state_dict())
        crash_at = random.random() < float(v.get("crash_prob", 0.0))            # 1-1
        crash_batch = random.randint(0, max(0, len(dataloader) * self.current_local_epoch - 1)) if crash_at else -1
        preempt_rolled = False
        drift_logged = adas_logged = False
        self.slow_s = 0.0
        e_cfg = cfg["energy"]
        per_batch_j = random.uniform(float(e_cfg["round_j_min"]), float(e_cfg["round_j_max"])) / \
            max(1, len(dataloader) * self.current_local_epoch)
        batch_no = 0
        epoch_train_loss = 0.0
        e = 0
        while e < self.current_local_epoch:
            if self.replace_global_model_during_local_update.is_set():
                e = 0
                epoch_train_loss = 0.0
                self.current_local_epoch -= 1
                if self.current_local_epoch < 2:
                    self.current_local_epoch = 2
                self.local_model_version = int(self.global_model_info[-3].item())
                self.last_global_loss = self.global_model_info[-2].item()
                self.model = copy.deepcopy(self.received_global_model)
                self.replace_global_model_during_local_update.clear()
                start_weights = copy.deepcopy(self.model.state_dict())
                continue
            for data, labels in dataloader:
                t_b = time.time()
                w = self.w()
                # 학습 중단 (패턴 1): 서버에는 아무것도 가지 않는다
                if w.get("engine_off") or not w.get("present", True):
                    return self._abort("1-5", "주차·시동 꺼짐으로 학습 중단")
                if w.get("ota"):
                    return self._abort("1-2", "OTA 설치·재부팅으로 학습 중단")
                if batch_no == crash_batch:
                    return self._abort("1-1", "계산 도중 돌발 중단")
                if self.energy <= 0:
                    return self._abort("1-3", "학습 중 배터리 방전")
                if w.get("zones", {}).get("school") and not preempt_rolled:
                    preempt_rolled = True
                    if random.random() < float(v.get("safety_preempt_prob", 0.0)):
                        return self._abort("1-4", "안전 기능(긴급 제동)이 연산을 가져가 FL 프로세스 종료")
                if w.get("zones", {}).get("weather") and float(v.get("drift_noise", 0)) > 0:   # 8-4 입력 분포 변화
                    data = data + float(v["drift_noise"]) * torch.randn_like(data)
                    if not drift_logged:
                        drift_logged = True
                        self.ev("8-4", "강우 구역에서 수집한 데이터로 학습 (입력 분포 변화)")
                optimizer.zero_grad()
                outputs = self.model.forward(data)
                loss = loss_function(outputs, labels)
                if e == 0:
                    epoch_train_loss += loss.detach().item()
                loss.backward()
                optimizer.step()
                batch_no += 1
                self.energy -= per_batch_j
                # 지연 (패턴 4): ECU 속도 배수, 보호구역 주행 보조 연산 점유
                spent = time.time() - t_b
                slow = self.ecu - 1.0
                if w.get("zones", {}).get("school"):
                    slow += float(v.get("adas_slowdown", 0.0))
                    if not adas_logged:
                        adas_logged = True
                        self.ev("4-5", "주행 보조 기능이 연산을 점유해 학습 지연")
                if slow > 0:
                    time.sleep(spent * slow)
                    self.slow_s += spent * (self.ecu - 1.0)
            e += 1
            self.set_status("training", f"v{self.local_model_version} · {e}/{self.current_local_epoch} epoch")
            if terminate_flag.is_set():
                return -1
        self.total_train_time += time.time() - start
        self.counters["trained"] += 1
        self.new_data = max(0.0, self.new_data - float(v.get("min_data", 32)))
        self._corrupt(start_weights, v)
        self.train_time = time.time() - start
        if self.ecu >= 1.4 and self.slow_s > 1.0:
            self.ev("4-1", f"ECU ×{self.ecu:.1f}로 학습이 {self.slow_s:.1f}초 늦어짐 ({self.train_time:.1f}초)")
        return epoch_train_loss / len(dataloader)

    def _abort(self, example, msg):
        self.counters["aborted"] += 1
        self.need_rejoin = True
        self.set_status("dead", msg)
        self.ev(example, msg + " → 서버는 아무 응답도 받지 못함")
        return ABORT

    def _corrupt(self, start_weights, v):
        """패턴 7: 제시간·정상 형식이지만 내용이 틀린 업데이트"""
        sd = self.model.state_dict()
        changed = []
        if float(v.get("weight_noise", 0)) > 0:                        # 7-4 센서 고장·불량 노드
            for k in sd:
                sd[k] = sd[k] + float(v["weight_noise"]) * torch.randn_like(sd[k]) * sd[k].std()
            changed.append(("7-4", f"가중치 노이즈 ×{v['weight_noise']}"))
        if float(v.get("backdoor_boost", 0)) > 0:                      # 7-3 백도어(업데이트 증폭)
            for k in sd:
                sd[k] = start_weights[k] + (1 + float(v["backdoor_boost"])) * (sd[k] - start_weights[k])
            changed.append(("7-3", f"업데이트 {1 + float(v['backdoor_boost']):.0f}배 증폭(백도어)"))
        if changed:
            self.model.load_state_dict(sd)
            for ex, msg in changed:
                self.ev(ex, msg)
        if v.get("lr_override"):
            self.ev("7-1", f"학습률 {v['lr_override']}로 학습 (정상 {self.lr})")

    # ---------------------------------------------------------------- 업로드 (원본 dist.send 한 번 → 시간·끊김 반영)
    def build_update(self, utility):
        v = self.vcfg()
        version = self.local_model_version + int(v.get("clock_skew", 0) or 0)       # 7-5 시계 오차 → 버전 오차
        if v.get("clock_skew"):
            self.ev("7-5", f"시계 오차로 버전 {self.local_model_version} → {version}로 보고")
        flatten_model = TensorBuffer(list(self.model.state_dict().values()))
        info = flatten_model.buffer.tolist()
        info.append(utility)
        info.append(version)
        return torch.tensor(info), version

    def send_local_model_to_server(self, utility):
        self.current_local_epoch = self.local_epoch
        tensor, version = self.build_update(utility)
        result = self.upload(tensor, version)
        if result == "queued":
            self.queue.append((tensor, version, time.time()))
            self.counters["queued"] += 1
        return result

    def _cell_slot(self, cell, capacity):
        """기지국 동시 업로드 수 제한 (4-2): 셀마다 capacity개의 잠금 파일 중 하나를 잡는다"""
        for k in range(capacity):
            f = open(RUN / f".cell_{cell}_{k}.lock", "w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return f
            except BlockingIOError:
                f.close()
        return None

    def upload(self, tensor, version):
        cfg, v = ENV.cfg(), self.vcfg()
        n = ENV.cfg()["net"]
        frags = max(1, int(n["fragments"]))
        t0 = time.time()
        delays = {}
        # 5-1 TDMA: 그룹 G개면 자기 순서가 올 때까지 기다린다
        G = int(n.get("tdma_groups", 1))
        if G > 1:
            slot = float(n["tdma_slot_s"])
            grp = self.id % G
            self.set_status("waiting", f"TDMA 그룹 {grp}/{G} 순서 대기")
            while int(time.time() / slot) % G != grp:
                time.sleep(0.05)
            delays["tdma"] = time.time() - t0
            if delays["tdma"] > slot * 0.5:
                self.ev("5-1", f"TDMA 순서 대기 {delays['tdma']:.1f}초 (그룹 {grp}/{G})")
        # 연결 확인 + 셀 슬롯 (4-2)
        w = self.w()
        if not (w.get("link_up") and w.get("server_ok")):
            return self._link_fail(w, 0, frags, v, "전송 시작 전")
        t_slot = time.time()
        lock = None
        while lock is None:
            w = self.w()
            if not (w.get("link_up") and w.get("server_ok")):
                return self._link_fail(w, 0, frags, v, "기지국 대기 중")
            lock = self._cell_slot(w["cell"], int(n["cell_capacity"]))
            if lock is None:
                self.set_status("waiting", f"기지국 {w['cell']} 혼잡 — 빈 슬롯 대기")
                time.sleep(0.2)
        delays["congestion"] = time.time() - t_slot
        if delays["congestion"] > 1.0:
            self.ev("4-2", f"기지국 {w['cell']} 혼잡으로 {delays['congestion']:.1f}초 대기")
        try:
            cell0, region0 = w["cell"], w["region"]
            # 5-2 셀 가장자리(품질 낮음)면 보관 후 전송 차량은 보류
            if v.get("store_forward") and w.get("quality", 1) < float(n.get("edge_quality", 0.15)):
                self.ev("5-2", f"통신 품질 {w['quality']:.2f}로 낮아 전송 보류")
                return "queued"
            retx_spent = 0.0
            i = 0
            while i < frags:
                w = self.w()
                bw = float(n["bandwidth_mbps"]) * (float(n["weather_bw_mult"]) if w.get("zones", {}).get("weather") else 1.0)
                frag_s = self.model_mb * 8 / frags / max(0.1, bw)
                self.set_status("uploading", f"{i}/{frags} 조각 · {bw:.0f} Mbps")
                time.sleep(frag_s)
                w = self.w()
                fail = None
                if not (w.get("link_up") and w.get("server_ok")):
                    fail = link_reason(w)
                elif w.get("cell") != cell0 and n.get("handover_drops", True):
                    fail = ("2-4", f"핸드오버 {cell0}→{w.get('cell')}")
                elif w.get("region") != region0 and random.random() < float(n["region_instability"]):
                    fail = ("2-5", f"권역 경계 {region0}→{w.get('region')} 망 불안정")
                elif self.energy <= 0:
                    fail = ("2-7", "전송 중 배터리 방전")
                elif w.get("speed_kmh", 0) > float(n["doppler_kmh"]) and random.random() < float(n["doppler_fail"]):
                    fail = ("2-2", f"{w['speed_kmh']:.0f} km/h 도플러로 채널 추정 실패")
                elif random.random() < float(n["frag_fail"]):
                    fail = ("2-3", "무선 패킷 오류")
                if fail is None:
                    self.energy -= float(cfg["energy"]["upload_j"]) / frags
                    i += 1
                    continue
                # 재전송 가능한 오류(패킷·도플러)는 제한 시간 안에서 같은 조각을 다시 보낸다 (4-3)
                if fail[0] in ("2-2", "2-3") and v.get("retransmit", True) and \
                        retx_spent + float(n["retx_timeout_s"]) <= float(n["retx_total_s"]):
                    self.set_status("waiting", f"재전송 대기 {n['retx_timeout_s']}초 ({fail[1]})")
                    self.ev(fail[0], f"{fail[1]} ({i + 1}/{frags} 조각) → {n['retx_timeout_s']}초 뒤 재전송")
                    time.sleep(float(n["retx_timeout_s"]))
                    retx_spent += float(n["retx_timeout_s"])
                    continue
                return self._link_fail(w, i, frags, v, fail[1], forced=fail[0])
            # 전송 완료: 실제로 서버에 보낸다 (원본과 같은 dist.send)
            dist.send(tensor=tensor, dst=0)
            total = time.time() - t0
            self.counters["sent"] += 1
            self.need_rejoin = False
            if retx_spent > 0:
                self.ev("4-3", f"재전송 대기로 {retx_spent:.0f}초 늦게 도착")
            up_s = self.model_mb * 8 / float(n["bandwidth_mbps"])
            if total > up_s * 3 and total > 2:
                self.ev("4-4", f"업로드에 {total:.1f}초 (대역폭·혼잡·재전송 포함)")
            C.emit(RUN, "upload", vid=self.id, msg=f"v{version} 업로드 완료 {total:.1f}초", version=version,
                   delays={k: round(x, 2) for k, x in delays.items()})
            if random.random() < float(v.get("dup_prob", 0)):              # 9-3 수신 확인만 끊겨 재전송
                dist.send(tensor=tensor, dst=0)
                self.ev("9-3", "수신 확인 손실로 같은 업데이트를 한 번 더 전송")
            self.set_status("idle", "업로드 완료, 다음 선택 대기")
            return "sent"
        finally:
            if lock is not None:
                fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()

    def _link_fail(self, w, sent_frags, frags, v, where, forced=None):
        ex, why = (forced, where) if forced else link_reason(w)
        wasted = self.model_mb * sent_frags / frags
        if v.get("store_forward") and ex in ("2-1", "2-6", "5-3", "1-5", "2-4", "3-4", "2-5", "2-2", "2-3"):
            self.set_status("queued", f"{why} → 보관 후 연결 회복 시 전송")
            if w.get("engine_off"):
                self.queued_while_off = True
            self.ev("5-2" if ex not in ("2-6", "5-3") else "5-3", f"{why}: 업데이트 보관 (연결 회복 후 전송)",
                    cause=ex)
            return "queued"
        self.counters["dropped"] += 1
        self.wasted_mb += wasted
        self.need_rejoin = True
        self.set_status("dead", f"{why} → 업데이트 폐기")
        pat = int(ex.split("-")[0])
        if pat == 2 or sent_frags > 0:
            self.ev(ex if pat == 2 else "2-1", f"{why}: {sent_frags}/{frags} 조각 전송 후 끊김, {wasted:.1f} MB 낭비",
                    wasted_mb=round(wasted, 2), cause=ex)
        else:
            self.ev(ex, f"{why}: 업로드 시작 못 함 → 서버는 아무것도 받지 못함")
        return "dropped"

    def flush_queue(self):
        """보관한 업데이트를 연결이 돌아오면 한꺼번에 보낸다 (5-x). 원본 서버는 차량당 버퍼 1칸이라 덮어씀"""
        w = self.w()
        if not self.queue or not (w.get("link_up") and w.get("server_ok")):
            return
        items, self.queue = self.queue, []
        ex = "5-4" if self.queued_while_off else ("5-3" if len(items) > 1 else "5-2")
        self.queued_while_off = False
        sent = 0
        for tensor, version, made in items:
            r = self.upload(tensor, version)
            if r != "sent":
                self.queue.append((tensor, version, made))
                continue
            sent += 1
        if sent:
            self.counters["burst"] += 1
            self.ev(ex, f"보관한 업데이트 {sent}개를 한꺼번에 전송 (버전 {[it[1] for it in items]})", n=sent)


# ================================================================== 실행 루프 (원본 SASAFL.runClient/runServer + 기록)
class SimSASAFL(M_SASAFL.SASAFL):

    def runClient(self, Client):
        is_ongoing = threading.Event()
        terminate = threading.Event()
        listen = threading.Thread(target=Client.receive_global_model_from_server,
                                  args=(is_ongoing, terminate, self.lag_tolerance), daemon=True)
        listen.start()
        while True:
            if terminate.is_set():
                break
            cfg = ENV.cfg()
            if Client.queue:
                Client.flush_queue()
            if is_ongoing.is_set():
                g = Client.gate()
                if g:
                    if Client.skip_active != g[0]:
                        Client.skip_active = g[0]
                        Client.counters["skipped"] += 1
                        Client.ev(g[0], f"{g[1]} → 이번엔 참여하지 않음" + (" (서버에 알릴 신호가 SASAFL에 없음)"
                                                                       if g[0].startswith("3") else ""))
                    Client.set_status("skipped", g[1])
                    time.sleep(0.3)
                    continue
                Client.skip_active = None
                utility = Client.train(terminate)
                if terminate.is_set() or utility == -1:
                    break
                if utility == ABORT:
                    is_ongoing.clear()
                    continue
                r = Client.send_local_model_to_server(utility)
                is_ongoing.clear()
                # 보관 후 전송 차량은 끊긴 동안 자기 모델로 계속 학습해 업데이트를 쌓는다 (최대 3개)
                if r == "queued" and len(Client.queue) < 3:
                    Client.need_rejoin = True
            elif Client.need_rejoin and cfg["fl"].get("rejoin", True):
                w = Client.w()
                can_store = Client.vcfg().get("store_forward") and Client.queue and len(Client.queue) < 3
                if (w.get("link_up") and w.get("server_ok") and not Client.gate()) or (can_store and not Client.gate()):
                    Client.ev(None, f"복귀: 마지막 모델 v{Client.local_model_version}로 다시 학습 "
                                    f"(서버는 v{Client.server_version})")
                    Client.need_rejoin = False
                    is_ongoing.set()
                else:
                    g = Client.gate()
                    why = g[1] if g else link_reason(w)[1]
                    Client.set_status("dead" if not Client.queue else "queued", f"복귀 대기: {why}")
            elif Client.need_rejoin:
                Client.set_status("excluded", "원본 SASAFL: 다시 선택되지 않음 (rejoin 꺼짐)")
            time.sleep(0.05)
        printLog(f"CLIENT {Client.id}", "FL 프로세스를 종료합니다.")
        dist.barrier()

    def runServer(self, Server):
        clients_idx = [idx for idx in range(1, Server.num_clients + 1)]
        picked_client_idx = clients_idx
        minimum_num_picked_client = int(Server.num_clients * Server.selection_ratio)
        current_num_picked_client = minimum_num_picked_client
        listen_local_update = threading.Thread(target=Server.receive_local_model_from_any_clients, daemon=True)
        global_acc, global_loss = Server.evaluate()
        Server.last_global_loss = global_loss
        Server.history = [{"round": 0, "acc": round(global_acc * 100, 2), "loss": round(global_loss, 4),
                           "t": time.time(), "quota": current_num_picked_client, "picked": []}]
        Server.publish(current_num_picked_client, "start")
        current_FL_start = time.time()
        listen_local_update.start()
        while True:
            current_round_start = time.time()
            Server.send_global_model_to_clients(clients_idx, picked_client_idx, global_loss)
            picked_client_idx = Server.wait_until_can_update_global_model(current_num_picked_client)
            stal = {i: Server.current_round - int(Server.local_model_version[i]) for i in picked_client_idx}
            Server.refine_received_local_model(picked_client_idx)
            coefficient = Server.calculate_coefficient(picked_client_idx)
            Server.average_aggregation(picked_client_idx, coefficient)
            global_acc, global_loss = Server.evaluate()
            Server.current_round += 1
            used = current_num_picked_client
            if Server.current_round == Server.target_rounds or global_acc >= Server.target_accuracy:
                break
            current_num_picked_client = Server.get_next_round_minimum_local_model(
                global_loss, current_num_picked_client, minimum_num_picked_client)
            dups = sorted({i for i in picked_client_idx if picked_client_idx.count(i) > 1})
            rec = {"round": Server.current_round, "acc": round(global_acc * 100, 2), "loss": round(global_loss, 4),
                   "t": time.time(), "dur": round(time.time() - current_round_start, 2), "quota": used,
                   "next_quota": current_num_picked_client,
                   "picked": [{"vid": i, "staleness": stal[i], "version": int(Server.local_model_version[i]),
                               "weight": round(float(coefficient[i]), 3)} for i in picked_client_idx],
                   "dups": dups, "elapsed": round(time.time() - current_FL_start, 1)}
            Server.history.append(rec)
            C.emit(RUN, "round", msg=f"{Server.current_round}라운드 집계: 차량 {picked_client_idx} "
                                     f"정확도 {rec['acc']}%", round=rec)
            if dups:
                wsum = sum(float(coefficient[i]) for i in picked_client_idx)
                C.emit(RUN, "fl", vid=dups[0], example=None,
                       msg=f"차량 {dups}: 같은 라운드에 두 번 집계됨 — 버퍼가 차량당 1칸이라 최신 것만 남음. "
                           f"원본 calculate_coefficient가 그 차량 비중을 합으로 두 번 나눠 비중 합이 {wsum:.4f} (1이어야 함)")
            Server.publish(current_num_picked_client, "aggregated")
        Server.terminate(clients_idx)
        dist.barrier()


# ================================================================== 서버
class SimServer(M_SERVER.SASAFLServer):

    def publish(self, quota, phase, waiting_s=0.0):
        C.write_json(RUN / "server.json", {
            "t": time.time(), "phase": phase, "round": self.current_round, "quota": quota,
            "cached": self.num_cached_local_model, "cached_ids": list(self.cached_client_idx),
            "waiting_s": round(waiting_s, 1), "history": self.history[-200:],
            "last_arrivals": getattr(self, "arrivals", [])[-30:]})

    def receive_local_model_from_any_clients(self):
        self.arrivals = []
        while not self.terminate_FL.is_set() or len(self.idle_clients) + self.num_cached_local_model < self.num_clients:
            temp_local_model = TensorBuffer(list(self.model.state_dict().values()))
            local_model_info = torch.zeros(len(temp_local_model.buffer) + 2)
            req = dist.irecv(tensor=local_model_info)
            req.wait()
            with self.cached_client_idx_lock and self.num_cached_local_model_lock:
                src = req.source_rank()
                dup = src in self.cached_client_idx
                self.cached_client_idx.append(src)
                self.num_cached_local_model += 1
                temp_local_model.buffer = local_model_info[:-2]
                self.flatten_client_models[src] = copy.deepcopy(temp_local_model)
                self.local_utility[src] = local_model_info[-2]
                self.local_model_version[src] = local_model_info[-1]
            ver = int(local_model_info[-1].item())
            arr = {"t": time.time(), "vid": src, "version": ver, "staleness": self.current_round - ver,
                   "overwrite": dup}
            self.arrivals.append(arr)
            C.emit(RUN, "arrive", vid=src, msg=f"업데이트 도착 v{ver} (staleness {self.current_round - ver})"
                   + (" — 같은 차량 업데이트가 버퍼에 이미 있어 덮어씀" if dup else ""),
                   version=ver, staleness=arr["staleness"], overwrite=dup)

    def wait_until_can_update_global_model(self, num_local_model_limit):
        t0 = time.time()
        warned = False
        last_pub = 0
        while True:
            with self.cached_client_idx_lock and self.num_cached_local_model_lock:
                if self.num_cached_local_model >= num_local_model_limit:
                    picked = [self.cached_client_idx.pop(0) for _ in range(self.num_cached_local_model)]
                    self.num_cached_local_model = 0
                    return picked
            waited = time.time() - t0
            if time.time() - last_pub > 0.5:
                self.publish(num_local_model_limit, "waiting", waited)
                last_pub = time.time()
            if waited > 60 and not warned:          # 9-2: 보낼 수 있는 차량이 quota보다 적으면 끝없이 기다린다
                warned = True
                alive = sum(1 for i in range(1, self.num_clients + 1)
                            if (C.read_json(RUN / f"client_{i}.json", {}) or {}).get("status") in
                            ("training", "uploading", "waiting", "queued"))
                C.emit(RUN, "fl", example="9-2",
                       msg=f"버퍼 {self.num_cached_local_model}/{num_local_model_limit}로 {waited:.0f}초째 대기 "
                           f"(학습·전송 중인 차량 {alive}대) — SASAFL에는 타임아웃이 없음")
            time.sleep(0.01)


def install():
    """Start.py가 쓰는 클래스 이름을 시뮬레이션 클래스로 바꾼다 (원본 파일은 그대로)."""
    M_SASAFL.SASAFL = SimSASAFL
    M_CLIENT.SASAFLClient = SimClient
    M_SERVER.SASAFLServer = SimServer
