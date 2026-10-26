#!/usr/bin/env bash
# 반복 실험(남은 E5 1회차 + 2~10회차 전체)을 wise-175·wise-176에 claim 방식으로 나눠 실행한다. wise-175에서 실행.
# 이미 돌고 있는 것은 다시 띄우지 않는다. 재부팅 후에는 두 노드 crontab @reboot이 다시 띄운다.
#   bash k30_reproduction/start_repeats.sh
#   tmux attach -t k30 → 1 status · 2 log · 3 reps(175 실행) · 4 sync(176 결과 가져오기)
#   176 진행: ssh wise-176@210.107.197.176 tail -f /home/wise-176/Mando/FedMPI/k30_reproduction/reps_console.log
export PATH=$HOME/bin:/home/wise-175/anaconda3/envs/tools/bin:$PATH
REPO=/home/wise-175/Mando/FedMPI
SESSION=k30
W176=wise-176@210.107.197.176
cd "$REPO"

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -n pipeline -c "$REPO"
    tmux new-window -t "$SESSION:1" -n status -c "$REPO" "watch -n 30 bash k30_reproduction/status.sh"
    tmux new-window -t "$SESSION:2" -n log -c "$REPO" \
        "bash -c 'while true; do f=\$(ls -t k30_reproduction/runs/*/run.log k30_reproduction/experiments/*/runs/*/run.log 2>/dev/null | head -1); echo \"=== \$f\"; timeout 600 tail -n 400 -f \"\$f\" | grep --line-buffered \"번째 글로벌 모델\"; done'"
fi
for w in 5 6; do tmux kill-window -t "$SESSION:$w" 2>/dev/null; done   # 끝난 E7·E8 창 정리

if pgrep -f "^bash k30_reproduction/run_rep_queue.sh" > /dev/null; then
    echo "175: 실행 중"
else
    tmux kill-window -t "$SESSION:3" 2>/dev/null
    tmux new-window -t "$SESSION:3" -n reps -c "$REPO" \
        "bash k30_reproduction/run_rep_queue.sh 2>&1 | tee -a k30_reproduction/reps_console.log; exec bash"
    echo "175: tmux $SESSION:3 에서 시작"
fi

if ssh -o BatchMode=yes -o ConnectTimeout=10 "$W176" 'pgrep -f "^bash k30_reproduction/run_rep_queue.sh" > /dev/null'; then
    echo "176: 실행 중"
else
    ssh -n -f -o BatchMode=yes -o ConnectTimeout=10 "$W176" \
        'cd /home/wise-176/Mando/FedMPI && setsid nohup bash k30_reproduction/run_rep_queue.sh >> k30_reproduction/reps_console.log 2>&1 < /dev/null &' \
        && echo "176: 시작 요청 (터보가 켜져 있으면 스스로 종료 — 176의 reps_console.log 확인)" || echo "176: 접속 실패"
fi

if pgrep -f "^bash k30_reproduction/sync_reps_from_176.sh" > /dev/null; then
    echo "동기화: 실행 중"
else
    tmux kill-window -t "$SESSION:4" 2>/dev/null
    tmux new-window -t "$SESSION:4" -n sync -c "$REPO" \
        "sleep 120; bash k30_reproduction/sync_reps_from_176.sh 2>&1 | tee -a k30_reproduction/sync_console.log; exec bash"
    echo "동기화: tmux $SESSION:4 에서 시작"
fi
echo "접속: tmux attach -t $SESSION   (빠져나오기: Ctrl+b 누른 뒤 d)"
