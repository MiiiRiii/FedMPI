#!/usr/bin/env python3
"""화면 서버 + 실행 관리. 표준 라이브러리만 쓴다 (FL 환경에 flask 없음).

  python vehicle_sim/serve.py [--port 8765]
  → 브라우저에서 http://localhost:8765  (원격이면 VS Code 포트 포워딩)

API
  GET  /api/static   맵·경로·39개 유스케이스·설정 기본값 (처음 한 번)
  GET  /api/state?ev=<오프셋>   world / server / clients / 새 사건 / 실행 상태
  POST /api/config   설정 일부 변경 (깊은 병합). 학습 중에도 바로 반영되는 값과 재시작이 필요한 값이 있다
  POST /api/control  {"action": "start" | "stop" | "server_crash"}
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "simcore"))
import common as C  # noqa: E402

PY = os.environ.get("SIM_PY", "/home/wise-175/anaconda3/envs/FL/bin/python")
WEB = HERE / "web"
SAVED_CFG = HERE / "config_last.json"


def deep_merge(base, patch):
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        elif v is None and k in base and isinstance(base, dict) and k.isdigit():
            base.pop(k)
        else:
            base[k] = v
    return base


class Manager:
    def __init__(self):
        self.lock = threading.Lock()
        self.run_dir = None
        self.procs = {}
        self.started = None
        self.note = ""
        self.cfg = C.read_json(SAVED_CFG, None) or json.loads(json.dumps(C.DEFAULT_CONFIG))
        for k, v in C.DEFAULT_CONFIG.items():          # 새로 생긴 설정 키 채우기
            if isinstance(v, dict):
                self.cfg.setdefault(k, {})
                for kk, vv in v.items():
                    self.cfg[k].setdefault(kk, vv)
        cur = HERE / "runs" / "current"
        if cur.exists():
            self.run_dir = cur.resolve()

    def save_cfg(self):
        C.write_json(SAVED_CFG, self.cfg)
        if self.run_dir:
            C.write_json(self.run_dir / "config.json", self.cfg)

    def status(self):
        st = {}
        for name, p in self.procs.items():
            st[name] = "running" if p.poll() is None else f"exit {p.returncode}"
        fl = self.procs.get("fl")
        if fl is not None and fl.poll() is not None and not getattr(fl, "_reported", False):
            fl._reported = True
            self.report_fl_exit(fl.returncode)
        return st

    def report_fl_exit(self, code):
        """학습이 끝난 원인을 fl.log 마지막 예외에서 찾아 사건으로 남긴다 (9-1과 연결)"""
        try:
            tail = (self.run_dir / "fl.log").read_text(errors="replace").splitlines()[-400:]
        except FileNotFoundError:
            tail = []
        errs = [ln.strip() for ln in tail if ln.strip().startswith(("ValueError", "RuntimeError", "TypeError",
                                                                    "AttributeError", "IndexError", "KeyError"))]
        cause = errs[-1] if errs else ""
        if "NaN" in cause:
            msg = ("서버 종료: 오염된 업데이트로 global loss가 NaN → SASAFL quota 계산 int(NaN) 예외 "
                   "(SASAFLServer.py:194). 패턴 7 업데이트가 서버 전체를 멈춤")
        elif self.note:
            msg = "서버 프로세스 강제 종료 후 torchrun이 모든 차량을 종료함"
        else:
            msg = f"학습 프로세스 종료 (exit {code}) {cause}"
        if code != 0:
            self.note = self.note or msg
            C.emit(self.run_dir, "fl", example="9-1", msg=msg)

    def start(self):
        with self.lock:
            self.stop_locked()
            self.run_dir = C.new_run_dir()
            C.write_json(self.run_dir / "config.json", self.cfg)
            fl = self.cfg["fl"]
            K = int(fl["K"])
            env = dict(os.environ, SIM_RUN_DIR=str(self.run_dir), PYTHONUNBUFFERED="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            self.procs["world"] = subprocess.Popen(
                [PY, str(HERE / "simcore" / "world.py"), str(self.run_dir)], env=env,
                stdout=open(self.run_dir / "world.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
            port = 29880 + int(time.time()) % 90
            args = ["--selection_ratio", str(fl["C"]), "--round", "100000", "--batch_size", str(fl["batch"]),
                    "--local_epochs", str(fl["epochs"]), "--lr", str(fl["lr"]), "--target_acc", "1.0",
                    "--dataset", fl["dataset"], "--iid", "False",
                    "--split", "gaussian" if fl["split"] == "dirichlet" else fl["split"],
                    "--method", "SASAFL", "--lag_tolerance", str(fl["tau"]), "--wandb_on", "False",
                    "--repeat", "1", "--cluster_type", "KISTI", "--num_threads", "1" * (K + 1)]
            self.procs["fl"] = subprocess.Popen(
                [PY, "-m", "torch.distributed.run", "--nnodes=1", f"--nproc_per_node={K + 1}",
                 f"--rdzv_id=vfl{port}", "--master_addr=127.0.0.1", f"--master_port={port}",
                 f"--rdzv_endpoint=127.0.0.1:{port}", str(HERE / "fl" / "fl_entry.py")] + args,
                cwd=str(C.REPO), env=env, stdout=open(self.run_dir / "fl.log", "w"),
                stderr=subprocess.STDOUT, start_new_session=True)
            self.started = time.time()
            self.note = ""
            C.emit(self.run_dir, "control", msg=f"시뮬레이션 시작: 차량 {K}대, {fl['dataset']}, C={fl['C']}, τ={fl['tau']}")

    def stop_locked(self):
        for name, p in list(self.procs.items()):
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait(timeout=10)
        self.procs = {}

    def stop(self):
        with self.lock:
            self.stop_locked()
            if self.run_dir:
                C.emit(self.run_dir, "control", msg="시뮬레이션 정지")

    def server_crash(self):
        """9-1 집계 서버 장애: SASAFL에는 서버 복구 경로가 없어 torchrun 작업 전체가 끝난다."""
        p = self.procs.get("fl")
        if not p or p.poll() is not None:
            return
        server_pid = None
        for line in subprocess.run(["ps", "-o", "pid=,args=", "--ppid", str(p.pid)], capture_output=True,
                                   text=True).stdout.splitlines():
            pid, args = line.strip().split(" ", 1)
            server_pid = server_pid or int(pid)          # 첫 번째 자식 = rank 0 (local_rank 0)
        C.emit(self.run_dir, "fl", example="9-1",
               msg="집계 서버 프로세스 강제 종료 — SASAFL은 체크포인트·서버 복구가 없어 학습 전체가 멈춘다")
        if server_pid:
            os.kill(server_pid, signal.SIGKILL)
        self.note = "서버 장애: 다시 시작하면 학습이 처음(0라운드)부터 시작됩니다"


MGR = Manager()


def read_events(path, offset, limit=400):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if offset > size:
                offset = 0
            if size - offset > 400_000:              # 너무 뒤처졌으면 최근 것만
                offset = size - 400_000
            f.seek(offset)
            data = f.read()
    except FileNotFoundError:
        return [], 0
    lines = data.split(b"\n")
    tail = lines.pop()                               # 마지막 줄이 덜 쓰였을 수 있음
    end = offset + len(data) - len(tail)
    out = []
    for ln in lines[-limit:]:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    return out, end


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/static":
            return self.send(200, {"map": C.MAP, "routes": C.ROUTES, "vehicle_routes": C.VEHICLE_ROUTES,
                                   "patterns": C.PATTERNS, "usecases": C.USECASES, "defaults": C.DEFAULT_CONFIG})
        if u.path == "/api/state":
            off = int(parse_qs(u.query).get("ev", ["0"])[0])
            rd = MGR.run_dir
            if not rd:
                return self.send(200, {"run": None, "config": MGR.cfg, "procs": {}, "events": [], "ev": 0})
            K = int(MGR.cfg["fl"]["K"])
            evs, end = read_events(rd / "events.jsonl", off)
            return self.send(200, {
                "run": rd.name, "procs": MGR.status(), "note": MGR.note, "started": MGR.started,
                "config": MGR.cfg, "world": C.read_json(rd / "world.json", {}),
                "server": C.read_json(rd / "server.json", {}),
                "clients": {str(i): C.read_json(rd / f"client_{i}.json", {}) for i in range(1, K + 1)},
                "events": evs, "ev": end})
        path = WEB / ("index.html" if u.path in ("/", "") else u.path.lstrip("/"))
        if path.is_file() and WEB in path.resolve().parents:
            ctype = {"html": "text/html; charset=utf-8", "js": "text/javascript; charset=utf-8",
                     "css": "text/css; charset=utf-8"}.get(path.suffix[1:], "application/octet-stream")
            return self.send(200, path.read_bytes(), ctype)
        self.send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        u = urlparse(self.path)
        if u.path == "/api/config":
            deep_merge(MGR.cfg, body)
            MGR.save_cfg()
            return self.send(200, {"ok": True, "config": MGR.cfg})
        if u.path == "/api/control":
            act = body.get("action")
            if act == "start":
                MGR.start()
            elif act == "stop":
                MGR.stop()
            elif act == "server_crash":
                MGR.server_crash()
            elif act == "reset_config":
                MGR.cfg = json.loads(json.dumps(C.DEFAULT_CONFIG))
                MGR.save_cfg()
            return self.send(200, {"ok": True, "procs": MGR.status()})
        self.send(404, {"error": "not found"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"차량 FL 시뮬레이터: http://{a.host}:{a.port}", flush=True)
    try:
        srv.serve_forever()
    finally:
        MGR.stop()


if __name__ == "__main__":
    main()
