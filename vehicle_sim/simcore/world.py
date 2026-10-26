#!/usr/bin/env python3
"""주행 환경 시뮬레이터: 차량 위치와 통신 상태를 0.1초마다 계산해 RUN_DIR/world.json에 쓴다.

차량마다 계산하는 값
  pos, speed_kmh, cell(서빙 기지국, 없으면 범위 밖), region(W/E), zone 플래그(터널·음영·보호구역·보안·강우·혼잡)
  link_up        무선 연결 여부 = 셀 안 + 터널 아님 + 음영지역 Gilbert 상태 '좋음'
  server_ok      집계 서버까지 도달 가능 = link_up + 보안 구역 아님
  quality        셀 가장자리일수록 낮음 (1 - 거리/반경)
  engine_off / charging / ota  정차 상태 (집 / 충전소 / 서비스센터)
  present        맵 안에 있는지 (통과 차량은 이탈 후 일정 시간 뒤 재진입)
  available      가용률 주기(8-1)에 따라 참여 가능한지
사건(핸드오버, 터널 진입, 주차, OTA, 합류·이탈)은 events.jsonl에 kind="world"로 남긴다.
사용법: python world.py RUN_DIR
"""

import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MAP, ROUTES, VEHICLE_ROUTES, DEFAULT_CONFIG, read_json, write_json, emit, merged_vehicle  # noqa: E402

DT = 0.1


def in_rect(p, r):
    return r[0] <= p[0] <= r[2] and r[1] <= p[1] <= r[3]


def zone_flags(p):
    f = {}
    for z in MAP["zones"]:
        if "rect" in z:
            hit = in_rect(p, z["rect"])
        else:
            cx, cz, rr = z["circle"]
            hit = math.hypot(p[0] - cx, p[1] - cz) <= rr
        f[z["kind"]] = hit
    return f


def road_speed(p):
    """현재 위치의 제한 속도 (km/h): 구역 제한 > 도로 제한"""
    for z in MAP["zones"]:
        if "speed" in z and "rect" in z and in_rect(p, z["rect"]):
            return z["speed"]
    best = 45
    for rd in MAP["roads"]:
        pts = rd["pts"]
        for a, b in zip(pts, pts[1:]):
            if seg_dist(p, a, b) < rd["width"]:
                best = rd["speed"]
                if rd["id"] == "highway":
                    return best
    return best


def seg_dist(p, a, b):
    ax, az = a[:2]
    bx, bz = b[:2]
    dx, dz = bx - ax, bz - az
    L2 = dx * dx + dz * dz
    t = 0 if L2 == 0 else max(0, min(1, ((p[0] - ax) * dx + (p[1] - az) * dz) / L2))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (az + t * dz))


class Vehicle:
    def __init__(self, vid, route_id, phase):
        self.vid = vid
        self.route_id = route_id
        self.route = ROUTES[route_id]
        self.pts = self.route["pts"]
        self.seg = 0
        self.s = 0.0                      # 현재 구간에서 간 거리
        self.stop_left = 0.0
        self.stop_kind = None
        self.present = True
        self.respawn_at = None
        self.gilbert_good = True
        self.cell = None
        self.region = None
        self.in_tunnel = False
        self.available = True
        self.pos = list(self.pts[0][:2])
        self.speed = 0.0
        # 경로 위 시작 위치를 phase(0~1)로 흩어 놓는다
        total = sum(self._seg_len(i) for i in range(len(self.pts) - 1))
        d = phase * total
        while d > self._seg_len(self.seg) and self.seg < len(self.pts) - 2:
            d -= self._seg_len(self.seg)
            self.seg += 1
        self.s = d
        self._update_pos()

    def _seg_len(self, i):
        a, b = self.pts[i], self.pts[i + 1]
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def _update_pos(self):
        a, b = self.pts[self.seg], self.pts[self.seg + 1]
        L = self._seg_len(self.seg) or 1
        t = min(1.0, self.s / L)
        self.pos = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
        self.heading = math.atan2(b[1] - a[1], b[0] - a[0])


class World:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.cfg = read_json(self.run_dir / "config.json", DEFAULT_CONFIG)
        K = int(self.cfg["fl"]["K"])
        routes = [VEHICLE_ROUTES[(i - 1) % len(VEHICLE_ROUTES)] for i in range(1, K + 1)]
        count = {}
        self.vehicles = []
        for i, r in enumerate(routes, start=1):
            n = routes.count(r)
            k = count.get(r, 0)
            count[r] = k + 1
            self.vehicles.append(Vehicle(i, r, (k + 0.5) / n))
        self.t0 = time.time()
        self.cfg_mtime = 0

    def reload_cfg(self):
        p = self.run_dir / "config.json"
        try:
            m = p.stat().st_mtime
        except FileNotFoundError:
            return
        if m != self.cfg_mtime:
            self.cfg = read_json(p, self.cfg)
            self.cfg_mtime = m

    def ev(self, v, msg, **kw):
        emit(self.run_dir, "world", vid=v.vid, msg=msg, **kw)

    def serving_cell(self, v):
        if self.in_tunnel(v):
            return None, 0.0
        best, q = None, 0.0
        for c in MAP["cells"]:
            d = math.hypot(v.pos[0] - c["pos"][0], v.pos[1] - c["pos"][1])
            if d <= c["r"]:
                qq = 1 - d / c["r"]
                if qq > q:
                    best, q = c, qq
        return best, q

    def in_tunnel(self, v):
        return zone_flags(v.pos).get("tunnel", False)

    def availability_target(self, now):
        w = self.cfg["world"]
        if w.get("availability_mode") != "cycle":
            return 1.0
        lo, hi, per = w["availability_min"], w["availability_max"], max(30, w["availability_period_s"])
        return lo + (hi - lo) * 0.5 * (1 - math.cos(2 * math.pi * (now - self.t0) / per))

    def step_vehicle(self, v, now):
        wcfg = self.cfg["world"]
        vcfg = merged_vehicle(self.cfg, v.vid)
        # 이탈(8-2, 8-3): 차량 설정 leave=True면 맵 밖으로
        if vcfg.get("leave") and v.present:
            v.present = False
            v.respawn_at = None
            self.ev(v, "맵에서 이탈 (차량 설정)", example=None)
        if not v.present:
            if vcfg.get("leave"):
                return
            if v.respawn_at is None:
                v.respawn_at = now + random.expovariate(1.0 / max(1, wcfg["visitor_gap_mean_s"]))
            if now >= v.respawn_at:
                v.present = True
                v.seg, v.s = 0, 0.0
                v._update_pos()
                self.ev(v, "진입 게이트로 새로 합류")
            return
        # 정차 중
        if v.stop_left > 0:
            v.stop_left -= DT
            v.speed = 0.0
            if v.stop_left <= 0:
                self.ev(v, {"home": "시동 켬, 출발", "gas": "충전 끝, 출발", "service": "OTA 설치·재부팅 완료"}[v.stop_kind],
                        stop=v.stop_kind, phase="end")
                v.stop_kind = None
            return
        # 이동
        limit = road_speed(v.pos)
        v.speed = limit * random.uniform(0.9, 1.0)
        v.s += v.speed / 3.6 * DT * float(wcfg.get("speed_scale", 1.0))
        while v.s >= v._seg_len(v.seg):
            v.s -= v._seg_len(v.seg)
            v.seg += 1
            if v.seg >= len(v.pts) - 1:
                if v.route["loop"]:
                    v.seg = 0
                else:                        # 통과 차량: 이탈 게이트 도착
                    v.present = False
                    v.respawn_at = None
                    v.seg, v.s = 0, 0.0
                    self.ev(v, "이탈 게이트로 맵을 떠남")
                    return
            node = v.pts[v.seg]
            if len(node) > 2:
                self.maybe_stop(v, node[2], wcfg, vcfg)
                if v.stop_left > 0:
                    v.s = 0.0
                    break
        v._update_pos()

    def maybe_stop(self, v, kind, wcfg, vcfg):
        if kind == "home":
            v.stop_left = random.uniform(*wcfg["home_park_s"])
            self.ev(v, "집 도착, 시동 끔 (주차)", stop="home", phase="start")
        elif kind == "gas":
            energy = (read_json(self.run_dir / f"client_{v.vid}.json", {}) or {}).get("energy", 1000)
            if energy < 0.5 * self.cfg["energy"]["budget_j"] or random.random() < 0.3:
                v.stop_left = random.uniform(*wcfg["charge_s"])
                self.ev(v, "충전소 정차, 충전 시작", stop="gas", phase="start")
        elif kind == "service":
            if random.random() < wcfg["ota_prob"]:
                v.stop_left = random.uniform(*wcfg["ota_s"])
                self.ev(v, "서비스센터 OTA 설치 시작 (재부팅)", stop="service", phase="start")
        if v.stop_left > 0:
            v.stop_kind = kind

    def link(self, v, now):
        ncfg = self.cfg["net"]
        z = zone_flags(v.pos)
        if not self.cfg["world"].get("weather_on", True):
            z["weather"] = False
        cell, q = self.serving_cell(v) if v.present else (None, 0.0)
        tunnel = z.get("tunnel", False)
        if tunnel != v.in_tunnel:
            self.ev(v, "터널 진입" if tunnel else "터널 통과", zone="tunnel")
            v.in_tunnel = tunnel
        # 음영지역(빌딩 숲): Gilbert 2상태. p = 좋음 유지 확률, q = 나쁨 유지 확률 (초당)
        if z.get("shadow"):
            p_stay = ncfg["gilbert_p"] if v.gilbert_good else ncfg["gilbert_q"]
            if random.random() > p_stay ** DT:
                v.gilbert_good = not v.gilbert_good
        else:
            v.gilbert_good = True
        cid = cell["id"] if cell else None
        if cid != v.cell and v.present:
            if v.cell and cid:
                self.ev(v, f"핸드오버 {v.cell} → {cid}", handover=[v.cell, cid])
            elif v.cell and not cid and not tunnel:
                self.ev(v, f"기지국 {v.cell} 범위 이탈")
            elif cid and not v.cell:
                self.ev(v, f"기지국 {cid} 범위 진입")
            v.cell = cid
        region = "W" if v.pos[0] < MAP["region_border_x"] else "E"
        if v.region and region != v.region and v.present:
            self.ev(v, f"권역 경계 통과 {v.region} → {region}", region_cross=[v.region, region])
        v.region = region
        link_up = bool(v.present and cid and not tunnel and v.gilbert_good and v.stop_kind != "home"
                       and v.stop_kind != "service")
        return {
            "pos": [round(v.pos[0], 1), round(v.pos[1], 1)], "heading": round(v.heading, 3),
            "speed_kmh": round(v.speed, 1), "route": v.route_id, "present": v.present,
            "cell": cid, "quality": round(q, 3), "region": region, "zones": z, "gilbert_good": v.gilbert_good,
            "link_up": link_up, "server_ok": link_up and not z.get("security", False),
            "engine_off": v.stop_kind == "home" or not v.present, "charging": v.stop_kind == "gas",
            "ota": v.stop_kind == "service", "stop_left": round(max(0, v.stop_left), 1),
            "available": v.available, "moving": v.speed > 1,
        }

    def run(self):
        last_avail = 0
        while True:
            now = time.time()
            self.reload_cfg()
            if self.cfg.get("_control", {}).get("stop_world"):
                break
            # 가용률(8-1): 목표 비율이 되도록 무작위로 참여 가능 차량을 고른다 (10초마다)
            if now - last_avail > 10:
                tgt = self.availability_target(now)
                for v in self.vehicles:
                    was = v.available
                    v.available = random.random() < tgt
                    if was != v.available and self.cfg["world"].get("availability_mode") == "cycle":
                        self.ev(v, f"가용 상태 {'켜짐' if v.available else '꺼짐'} (목표 가용률 {tgt:.0%})")
                last_avail = now
            state = {}
            for v in self.vehicles:
                self.step_vehicle(v, now)
                state[str(v.vid)] = self.link(v, now)
            write_json(self.run_dir / "world.json", {"t": now, "elapsed": now - self.t0,
                                                     "availability_target": self.availability_target(now),
                                                     "vehicles": state})
            time.sleep(max(0.0, DT - (time.time() - now)))


if __name__ == "__main__":
    World(sys.argv[1]).run()
