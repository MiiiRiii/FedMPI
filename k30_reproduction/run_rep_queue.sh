#!/usr/bin/env bash
# 남은 1회차(E5 τ) + 2~10회차 반복 실행 전체를 두 노드가 claim 방식으로 나눠 돌린다.
# - 순서: queue_order.txt (run_pipeline.py --print-queue) — 회차 단위로 한 벌씩 완성 (2회차 전체 → 3회차 전체 …)
# - 실행 이름을 wise-175의 k30_reproduction/claims/에 먼저 차지한 노드만 돌린다. 끝난 실행은 미리 차지돼 있어 다시 돌지 않는다.
# - 중단·재부팅 후 다시 실행하면 이어서 간다 (자기가 차지했던 미완료 실행은 다시 시작).
set -u
HOST=$(hostname)
CLAIMS=/home/wise-175/Mando/FedMPI/k30_reproduction/claims
export K30_CLAIM_DIR=$CLAIMS
case "$HOST" in
    wise-175)
        cd /home/wise-175/Mando/FedMPI
        PIPE_PY=/home/wise-175/anaconda3/envs/FL/bin/python
        owner() { cat "$CLAIMS/$1/host" 2>/dev/null; }
        ;;
    wise-176)
        cd /home/wise-176/Mando/FedMPI
        PIPE_PY=/home/wise-176/miniconda3/bin/python
        export K30_CLAIM_SSH=wise-175@210.107.197.175
        export MASTER_ADDR=210.107.197.176 MASTER_REPO=/home/wise-176/Mando/FedMPI MASTER_PY=/home/wise-176/miniconda3/envs/FL/bin/python
        # 175에 닿지 않으면(ssh 종료 코드 255) 건너뛰지 않고 닿을 때까지 기다린다 (2026-10-23: 175 정지 중 568개를 건너뛰고 끝난 일)
        owner() {
            local out rc
            while true; do
                out=$(ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$K30_CLAIM_SSH" "cat $CLAIMS/$1/host 2>/dev/null; true"); rc=$?
                (( rc != 255 )) && { echo "$out"; return; }
                echo "[$(date '+%F %T')] wise-175 접속 실패 — 2분 뒤 다시 시도 ($1)" >&2; sleep 120
            done
        }
        # 176은 터보를 켜면 P코어가 계산을 틀린다 (2026-10-19 진단)
        if [[ "$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null)" != "1" && "${K30_ALLOW_TURBO:-0}" != "1" ]]; then
            echo "[$(date '+%F %T')] wise-176: 터보가 켜져 있어 실행하지 않음 (no_turbo=1 설정 후 다시 실행)"; exit 3
        fi
        ;;
    *) echo "unknown host $HOST" >&2; exit 2 ;;
esac
# 이전 queue에서 넘겨받은 실행(같은 노드에서 아직 도는 run_pipeline)이 있으면 끝날 때까지 기다린다
while pgrep -f "k30_reproduction/pipeline/run_pipeline.py --only" > /dev/null; do sleep 60; done
$PIPE_PY k30_reproduction/pipeline/run_pipeline.py --print-queue > k30_reproduction/queue_order.txt
echo "[$(date '+%F %T')] $HOST 반복 queue 시작 ($(wc -l < k30_reproduction/queue_order.txt)개 항목)"
while read -r eid name rel; do
    o=$(owner "$name")
    if [[ -n "$o" && "$o" != "$HOST" ]]; then continue; fi          # 다른 노드가 차지(또는 이미 완료)
    f="k30_reproduction/$rel/result.json"
    if [[ "$o" == "$HOST" && -f "$f" ]] && grep -q '"outcome": "completed"' "$f"; then continue; fi
    $PIPE_PY k30_reproduction/pipeline/run_pipeline.py --only "$eid" --names "$name" < /dev/null
done < k30_reproduction/queue_order.txt
echo "[$(date '+%F %T')] $HOST REP_QUEUE_DONE"
