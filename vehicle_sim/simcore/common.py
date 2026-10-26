"""차량 연합학습 장애 시뮬레이터 — 공용 정의 (맵, 설정 기본값, 39개 유스케이스 목록, 파일 입출력).

프로세스 구성 (모두 같은 실행 폴더 RUN_DIR을 공유):
  world.py      차량 이동·셀·터널·음영 등 주행 환경을 0.1초마다 계산 → world.json
  fl_entry.py   실제 SASAFL(torchrun) 실행. 차량(클라이언트)은 world.json을 보고 학습·전송·끊김을 결정
                → client_<id>.json, server.json, events.jsonl
  serve.py      위 파일들을 묶어 화면(web/)에 내보내고, 설정 변경·시작/정지를 받는다 → config.json
SASAFL 원본 파일(methods/SASAFL/*)은 수정하지 않는다. 동작은 fl/sim_sasafl.py의 서브클래스로 끼워 넣는다.
"""

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # vehicle_sim/
REPO = ROOT.parent                                    # FedMPI/

# ------------------------------------------------------------------ 파일 입출력
def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def emit(run_dir, kind, vid=None, example=None, msg="", **extra):
    """events.jsonl에 사건 한 줄을 추가한다. example은 유스케이스 id(예: '2-6')."""
    ev = {"t": round(time.time(), 3), "kind": kind, "vid": vid, "example": example, "msg": msg}
    if example:
        ev["pattern"] = int(example.split("-")[0])
    ev.update(extra)
    line = json.dumps(ev, ensure_ascii=False) + "\n"
    with open(Path(run_dir) / "events.jsonl", "a") as f:   # 작은 한 줄 append는 프로세스 간에도 섞이지 않는다
        f.write(line)


# ------------------------------------------------------------------ 맵 (단위: m, x 0~2000, z 0~1400)
# 문서의 수치(반경 500 m, 고속 26.82 m/s ≈ 97 km/h, 터널 80 km/h 등)를 실제 축척으로 쓴다.
MAP = {
    "size": [2000, 1400],
    "roads": [
        {"id": "highway", "name": "고속도로", "pts": [[0, 300], [2000, 300]], "speed": 100, "width": 28},
        {"id": "conn_w", "name": "서측 진입로", "pts": [[300, 300], [300, 600]], "speed": 50, "width": 16},
        {"id": "conn_e", "name": "동측 진입로", "pts": [[1700, 300], [1700, 700]], "speed": 50, "width": 16},
        {"id": "ring", "name": "도심 순환로", "pts": [[300, 600], [1100, 600], [1100, 1200], [300, 1200], [300, 600]],
         "speed": 45, "width": 18},
        {"id": "suburb", "name": "교외 순환로", "pts": [[1100, 700], [1800, 700], [1800, 1200], [1100, 1200]],
         "speed": 55, "width": 18},
    ],
    # 기지국(셀). 서로 겹치는 곳은 핸드오버, 어느 반경에도 없는 곳은 음영(범위 밖)
    "cells": [
        {"id": "A", "name": "도심 기지국 A", "pos": [650, 900], "r": 480, "region": "W", "color": "#3b82f6"},
        {"id": "H1", "name": "고속도로 서측 H1", "pos": [300, 330], "r": 360, "region": "W", "color": "#06b6d4"},
        {"id": "H2", "name": "고속도로 동측 H2", "pos": [1600, 330], "r": 360, "region": "E", "color": "#a855f7"},
        {"id": "C", "name": "교외 기지국 C", "pos": [1450, 950], "r": 430, "region": "E", "color": "#f97316"},
    ],
    "region_border_x": 1100,          # 서쪽 W 권역 / 동쪽 E 권역 (지역 간 망 불안정, 지역별 참여 치우침)
    "zones": [
        {"id": "tunnel", "name": "터널", "kind": "tunnel", "rect": [850, 280, 1150, 320]},
        {"id": "shadow", "name": "빌딩 숲(음영지역)", "kind": "shadow", "rect": [450, 1050, 800, 1260]},
        {"id": "congest", "name": "혼잡 교차로", "kind": "congestion", "rect": [250, 850, 350, 1050], "speed": 15},
        {"id": "school", "name": "어린이 보호구역", "kind": "school", "rect": [1060, 820, 1140, 980], "speed": 30},
        {"id": "security", "name": "보안 구역(사내 게이트웨이)", "kind": "security", "rect": [1740, 850, 1900, 1050]},
        {"id": "weather", "name": "강우 구역", "kind": "weather", "circle": [1450, 850, 250]},
    ],
    "places": [
        {"id": "home", "name": "집·주차장", "kind": "home", "pos": [1500, 1200]},
        {"id": "gas", "name": "주유소·충전소", "kind": "gas", "pos": [300, 750]},
        {"id": "service", "name": "서비스센터(OTA)", "kind": "service", "pos": [700, 600]},
        {"id": "datacenter", "name": "FL 집계 서버", "kind": "server", "pos": [1000, 1330]},
        {"id": "gate_in", "name": "진입 게이트", "kind": "gate", "pos": [0, 300]},
        {"id": "gate_out", "name": "이탈 게이트", "kind": "gate", "pos": [2000, 300]},
    ],
}

# 경로: [x, z] 또는 [x, z, 정차 종류]. 정차 종류 home(시동 꺼짐) / gas(충전) / service(OTA, 확률적)
ROUTES = {
    "commute": {"name": "고속도로 통근", "loop": True,
                "pts": [[300, 600], [300, 300], [1700, 300], [1700, 700], [1800, 700], [1800, 1200],
                        [1500, 1200, "home"], [1100, 1200], [1100, 600], [300, 600]]},
    "city": {"name": "도심 순환", "loop": True,
             "pts": [[300, 600], [700, 600, "service"], [1100, 600], [1100, 1200], [300, 1200],
                     [300, 750, "gas"], [300, 600]]},
    "suburb": {"name": "교외 순환", "loop": True,
               "pts": [[1100, 700], [1800, 700], [1800, 1200], [1500, 1200, "home"], [1100, 1200], [1100, 700]]},
    "visitor": {"name": "통과 차량(합류·이탈)", "loop": False, "pts": [[0, 300], [2000, 300]]},
}
# 차량 i(1부터)의 경로. 앞 8대만으로도 네 경로가 모두 나오도록 섞어 둔다 (통과 차량 = 합류·이탈 8-3)
VEHICLE_ROUTES = ["commute", "city", "suburb", "visitor", "commute", "city", "suburb", "visitor",
                  "city", "commute", "suburb", "city", "commute", "visitor", "city", "suburb"]

# ------------------------------------------------------------------ 설정 (화면에서 바꿈. restart=True는 학습 재시작 필요)
DEFAULT_CONFIG = {
    "fl": {"K": 8, "dataset": "MNIST", "C": 0.3, "tau": 9, "epochs": 1, "lr": 0.01, "batch": 32,
           "split": "gaussian", "alpha": 0.1, "rejoin": True},
    "net": {"bandwidth_mbps": 55.7, "fragments": 10, "frag_fail": 0.01, "doppler_kmh": 60, "doppler_fail": 0.15,
            "cell_capacity": 2, "tdma_groups": 1, "tdma_slot_s": 2.0, "retx_timeout_s": 5.0, "retx_total_s": 20.0,
            "gilbert_p": 0.85, "gilbert_q": 0.7, "region_instability": 0.3, "handover_drops": True,
            "weather_bw_mult": 0.5, "edge_quality": 0.15},
    "energy": {"budget_j": 1000, "round_j_min": 20, "round_j_max": 30, "min_energy_j": 100,
               "charge_rate_jps": 40, "upload_j": 2},
    "world": {"availability_mode": "off", "availability_min": 0.1, "availability_max": 0.8,
              "availability_period_s": 600, "weather_on": True, "visitor_gap_mean_s": 60,
              "home_park_s": [60, 120], "ota_prob": 0.3, "ota_s": [31, 101], "charge_s": [30, 60],
              "speed_scale": 1.0},
    "vehicle_default": {"ecu_factor": None, "crash_prob": 0.02, "safety_preempt_prob": 0.3, "adas_slowdown": 1.5,
                        "store_forward": False, "retransmit": True, "charging_only": False, "energy_gate": True,
                        "min_data": 32, "data_rate_per_s": 5, "lr_override": None, "label_noise": 0.0,
                        "weight_noise": 0.0, "backdoor_boost": 0.0, "clock_skew": 0, "dup_prob": 0.0,
                        "leave": False, "drift_noise": 0.3},
    "vehicles": {},
}


def merged_vehicle(cfg, vid):
    v = dict(cfg.get("vehicle_default", {}))
    v.update(cfg.get("vehicles", {}).get(str(vid), {}))
    return v


# ------------------------------------------------------------------ 39개 유스케이스 (문서 "패턴별 예시와 근거")
# trigger: 맵에서 어디서/어떤 설정으로 일어나는지. evidence: 실측 / 논문 실험 / 가정 / 계산
PATTERNS = {1: "아예 안 옴", 2: "보내다가 끊김", 3: "미리 빠짐", 4: "늦게 옴", 5: "모아서 한꺼번에",
            6: "옛 모델로 학습해 옴", 7: "내용이 틀림", 8: "참여 구성이 치우침", 9: "서버 쪽 문제"}
USECASES = [
    ("1-1", "계산 도중 돌발 중단 (업데이트의 약 2%)", "FeLiX(Google FL) · 실측", "차량 설정: 계산 중단 확률"),
    ("1-2", "OTA 설치와 재부팅 (31–101초)", "Bhattacharjee 외 2023 · 실측", "서비스센터 정차"),
    ("1-3", "배터리 방전 (1000 J, 라운드당 20–30 J)", "FedQoS·VEFL · 논문 실험(계산)", "에너지 설정 · 충전소"),
    ("1-4", "안전 기능 선점 (긴급 제동 등)", "FailBench 주입 방식 · 가정", "어린이 보호구역"),
    ("1-5", "시동 꺼짐 (주차 중 학습 중단)", "Bonawitz 외 2019 유사 사례 · 가정", "집·주차장 정차"),
    ("2-1", "기지국 범위 이탈 (체류 5초 미만 28%)", "VEFL · 논문 실험", "셀 사이 음영(범위 밖)"),
    ("2-2", "고속 주행 도플러 (60–100 km/h)", "VR-VFL · 논문 실험", "고속도로 + 도플러 실패 확률"),
    ("2-3", "무선 패킷 오류", "Salehi·Hossain 2021 · 논문 실험", "조각당 전송 실패 확률"),
    ("2-4", "기지국 핸드오버", "Chang 외 2024 · 논문 실험", "셀 경계 통과"),
    ("2-5", "지역 간 망 불안정", "NVIDIA FLARE 운영 · 실측", "권역 경계(x=1100) 통과"),
    ("2-6", "터널 진입 (1 km @80 km/h ≈ 45초)", "가정(계산)", "고속도로 터널"),
    ("2-7", "전송 중 배터리 방전", "가정", "에너지 0에서 업로드"),
    ("3-1", "잔여 에너지 부족", "FedQoS · 논문 실험", "에너지 최소 기준"),
    ("3-2", "학습할 데이터 부족", "FedQoS · 논문 실험", "차량 설정: 최소 데이터·수집 속도"),
    ("3-3", "충전·비과금망 조건 미달", "Bonawitz 외 2019 · 실측", "차량 설정: 충전 중일 때만 참여"),
    ("3-4", "보안 정책 차단 (게이트웨이)", "가정", "보안 구역"),
    ("4-1", "하드웨어 성능 차이 (최대 약 2.2배)", "FeLiX·VEFL · 실측(계산)", "차량 설정: ECU 속도 배수"),
    ("4-2", "기지국 혼잡 (동시 수용 한계)", "Bhattacharjee 외·VEFL · 실측", "셀 동시 업로드 수 · 혼잡 교차로"),
    ("4-3", "재전송 대기 (메시지당 5초)", "NVIDIA FLARE 문서 · 실측", "재전송 제한 시간"),
    ("4-4", "상향 대역폭 부족 (55.7 / 70.9 Mbps)", "FedQoS · 논문 실험", "대역폭 · 강우 구역"),
    ("4-5", "주행 보조 기능의 연산 점유", "가정", "어린이 보호구역 감속"),
    ("5-1", "TDMA 송신 순서 대기 (G−1라운드)", "Song·Hong 2024 · 논문 실험", "TDMA 그룹 수"),
    ("5-2", "통신 상태가 나빠 전송 보류", "FedQoS · 논문 실험", "셀 가장자리 · 보관 후 전송"),
    ("5-3", "터널·음영지역 연속 단절 (Gilbert)", "Gilbert 1960 · 가정", "빌딩 숲 · 터널 + 보관 후 전송"),
    ("5-4", "재시동 후 보관분 업로드", "가정", "주차 중 보관분 → 출발 시 전송"),
    ("6-1", "새 모델 수신 실패", "Huang 외 2023 · 논문 실험", "배포 시점에 범위 밖·터널"),
    ("6-2", "OTA 배포 중 버전 혼재", "Bhattacharjee 외·FedBuff · 실측", "서비스센터 OTA 후 복귀"),
    ("7-1", "학습 설정 오류 (학습률)", "Huang 외 2023 · 논문 실험", "차량 설정: 학습률 덮어쓰기"),
    ("7-2", "라벨 오류", "Huang 외 2023 · 논문 실험", "차량 설정: 라벨 오류 비율(시작 시)"),
    ("7-3", "백도어 공격", "Bagdasaryan 외 2020 · 논문 실험", "차량 설정: 업데이트 증폭"),
    ("7-4", "센서 고장·불량 노드", "Wu 외 2023 · 논문 실험", "차량 설정: 가중치 노이즈"),
    ("7-5", "차량 시계 오차", "SyncFed 2025 · 논문 실험", "차량 설정: 버전 오차"),
    ("8-1", "참여 가능 기기 비율 변동 (10–80%)", "FeLiX(MobiPerf) · 실측", "전체 설정: 가용률 주기"),
    ("8-2", "라벨이 쏠린 차량 이탈 (α=0.1)", "VEFL·FeLiX·FedLC · 논문 실험", "차량 설정: 이탈 (라벨 쏠림 차량)"),
    ("8-3", "새 차량 합류, 기존 차량 이탈", "Chang 외 2024·Ruan 외 2021 · 논문 실험", "진입·이탈 게이트"),
    ("8-4", "계절·날씨·운전자 변화", "FedDrift 2023 · 논문 실험", "강우 구역(입력 분포 변화)"),
    ("9-1", "집계 서버 장애", "FailBench·Dispersed FL · 논문 실험", "버튼: 서버 장애"),
    ("9-2", "집계 버퍼 무한 대기", "연구실 SASAFL 재현 · 실측", "살아 있는 차량 < quota"),
    ("9-3", "같은 업데이트 중복 도착", "가정", "차량 설정: 중복 전송 확률"),
]
assert len(USECASES) == 39


def new_run_dir():
    d = ROOT / "runs" / time.strftime("%Y%m%d_%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    (d / "events.jsonl").touch()
    cur = ROOT / "runs" / "current"
    if cur.is_symlink() or cur.exists():
        cur.unlink()
    cur.symlink_to(d.name)
    return d
