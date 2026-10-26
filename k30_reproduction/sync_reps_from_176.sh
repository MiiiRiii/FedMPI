#!/usr/bin/env bash
# wise-175에서 실행: 10분마다 wise-176이 차지한(claims/*/host = wise-176) 실행 폴더를 가져오고 그림·SUMMARY·반복 파일을 다시 만든다.
# queue_order.txt로 실행 이름 → 폴더 위치를 찾는다. 전체 queue가 끝나면 REPS_ALL_DONE을 찍고 끝난다.
cd /home/wise-175/Mando/FedMPI
PY=/home/wise-175/anaconda3/envs/FL/bin/python
K=k30_reproduction
R176=wise-176@210.107.197.176:/home/wise-176/Mando/FedMPI
while true; do
    while read -r eid name rel; do
        [ "$(cat $K/claims/$name/host 2>/dev/null)" = wise-176 ] || continue
        [ -f "$K/$rel/result.json" ] && grep -q '"outcome": "completed"' "$K/$rel/result.json" && continue
        mkdir -p "$K/$rel"
        rsync -a --exclude pids -e "ssh -o BatchMode=yes -o ConnectTimeout=10" "$R176/$K/$rel/" "$K/$rel/" 2>&1 | tail -1
    done < $K/queue_order.txt
    $PY $K/pipeline/run_pipeline.py --report-only --quiet > /dev/null 2>&1
    done_n=0; total=0
    while read -r eid name rel; do
        total=$((total+1)); [ -f "$K/$rel/result.json" ] && grep -q '"outcome": "completed"' "$K/$rel/result.json" && done_n=$((done_n+1))
    done < $K/queue_order.txt
    echo "[$(date '+%F %T')] 동기화: 완료 $done_n/$total"
    if (( done_n >= total )); then echo "[$(date '+%F %T')] REPS_ALL_DONE"; break; fi
    sleep 600
done
