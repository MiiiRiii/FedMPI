# 실험 접속·확인·재시작 방법 (tmux)

## tmux 위치

- 설치: conda 환경 `tools`의 tmux 3.7c (`/home/wise-175/anaconda3/envs/tools/bin/tmux`). 2026-10-12 설치, sudo 없이 설치하기 위해 이 경로를 썼다.
- 바로가기: `~/bin/tmux`. `~/bin`은 로그인 셸에서 `~/.profile`이 PATH에 넣는다.
- `tmux: command not found`가 나오면 다음 중 하나를 쓴다.
  - `export PATH=$HOME/bin:$PATH`
  - 전체 경로 `/home/wise-175/anaconda3/envs/tools/bin/tmux`

## 접속 (어느 터미널에서든: SSH, VS Code 터미널)

```bash
tmux ls                 # 세션 목록. 'k30'이 보여야 한다
tmux attach -t k30      # 접속
```

| 창 | 이름 | 내용 |
|---|---|---|
| 0 | pipeline | 실험 본체 (tmux로 시작한 경우 여기서 출력이 흐른다) |
| 1 | status | 30초마다 `status.sh` 상태 화면 |
| 2 | log | 지금 실행의 라운드별 정확도 |

| 키 | 동작 |
|---|---|
| `Ctrl+b` 누른 뒤 `0` / `1` / `2` | 창 이동 |
| `Ctrl+b` 누른 뒤 `d` | **빠져나오기 (실험은 계속 돈다)** |
| `Ctrl+b` 누른 뒤 `[` | 스크롤 모드 (방향키로 이동, `q`로 종료) |

주의: 창 안에서 `Ctrl+c`를 누르거나 `exit`하지 않는다. 0번 창에서 누르면 실험이 멈춘다. 나올 때는 항상 `Ctrl+b d`를 쓴다.

## tmux 없이 한 번만 확인

```bash
bash ~/Mando/FedMPI/k30_reproduction/status.sh
cat  ~/Mando/FedMPI/k30_reproduction/PROGRESS.md
```

## 새로 시작 / 중단 후 다시 시작 (다음부터는 이 방법)

```bash
cd ~/Mando/FedMPI && bash k30_reproduction/start_in_tmux.sh
tmux attach -t k30
```

- 실험이 이미 돌고 있으면 새로 시작하지 않고 관찰 창만 만든다. 중복 실행을 막기 위해서다.
- 완료된 실행은 건너뛰고 남은 실행부터 이어 간다.
- 서버(175)가 재부팅되면 tmux 세션도 사라진다. 이때 위 두 줄로 다시 시작한다.

## 현재(2026-10-12) 상태 메모

- 지금 돌고 있는 파이프라인은 tmux 이전에 `setsid nohup`으로 띄운 것이다. tmux 0번 창에는 출력이 나오지 않는다.
  - 출력 로그: `k30_reproduction/pipeline_console.log`
  - 진행 기록: `k30_reproduction/PROGRESS.md`
  - 화면 확인: 1번(status), 2번(log) 창
- 이 파이프라인도 세션과 분리돼 있어 로그아웃해도 계속 돈다. 이후 재시작은 `start_in_tmux.sh`로 한다.

## E7 (적응형 quota 꺼진 원저자 코드) — 2026-10-16 추가
- tmux 세션 `k30`의 **3번 창 `err`**: `start_err_after_current.sh`가 E5 `LT_tau1_rep1`이 끝나길 기다렸다가 기존 파이프라인을 멈추고 E7 9개 실행을 시작한다.
- 콘솔 기록: `k30_reproduction/err_console.log` (끝나면 `ERR_DONE`)
- 결과: `experiments/E7_adaptive_off_2023code/` — `compare_fig7_err.png`, `compare_fig9_err.png`, `SUMMARY.md` (E6에도 같은 그림)
- E5 나머지(τ=2–10)는 보류 상태. 이어 돌리려면 E7 종료 후 `bash k30_reproduction/start_in_tmux.sh`.

## E7 두 노드 실행 (2026-10-19 변경)
- 시작/재개: `bash k30_reproduction/start_e7_two_nodes.sh` (wise-175에서). 재부팅 시 두 노드 crontab `@reboot`이 자동으로 다시 띄운다.
- 9개 실행을 두 노드가 `experiments/E7_adaptive_off_2023code/claims/`에 먼저 이름을 차지하는 방식으로 나눈다 (중복 실행 없음, 한쪽이 멈추면 다른 쪽이 남은 것을 가져감).
- tmux `k30`: 3 e7(175, 오전에 띄운 고정 4개) · 5 queue(3이 끝나면 이어서 queue) · 4 sync(176 결과 10분마다 가져오기)
- 176 진행: `ssh wise-176@210.107.197.176 tail -f /home/wise-176/Mando/FedMPI/k30_reproduction/e7_console.log`
- **176은 터보가 켜져 있으면 실행하지 않는다** (P코어 계산 오류). 176에서 `echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo` 후 위 시작 스크립트를 다시 실행.
- crontab 항목 제거: `crontab -e`에서 `# k30 E7 자동 재개` 줄 삭제 (두 노드 모두).

## E8 (K=10) 추가 — 2026-10-19
- 시작/재개 스크립트가 `start_two_nodes.sh`로 바뀜 (E7 → E8 순서로 두 노드가 claim 방식으로 나눠 실행). 재부팅 시 crontab이 자동 실행.
- E8 = K=10에서 BSA · SASAFL(현재 코드) · SASAFL(원저자 코드) × MNIST/FashionMNIST/CIFAR-10 × C 0.1/0.3/0.5 = 27개, 각 500라운드.
- 논문의 K=100 칸 자리에 K=10이 들어간다: compare_fig7 (b)(d), compare_fig8 (b), compare_fig11, compare_table4, compare_fig7_err (b)(d).
- tmux k30 창 6 `e8`(175), 176은 `e8_console.log`. 동기화 창 4는 `sync_from_176.sh` (E7·E8 모두).

## 반복 실험 (논문: 설정당 10회 평균) — 2026-10-22 시작
- 시작/재개: `bash k30_reproduction/start_repeats.sh` (wise-175에서). 재부팅 시 두 노드 crontab `@reboot`이 자동 재개.
- 대상: 남은 E5 1회차(τ=2–8,10) + E1·E2·E4·E5·E7·E8 2~10회차 (총 575개 실행). 순서는 `queue_order.txt` — 회차 단위(2회차 전체 → 3회차 전체 …)라 중간에 멈춰도 완성된 회차까지는 한 벌씩 갖춰진다.
- 두 노드 분배: `k30_reproduction/claims/<실행이름>/host` (먼저 차지한 노드가 실행, 끝난 실행은 미리 차지돼 있음).
- 실행 폴더 이름: 1회차는 기존 이름, 2회차부터 `<이름>_rep2` … `_rep10` (E5는 `LT_tau<τ>_rep<회차>`). τ=9는 E1 SASAFL C=0.3 반복을 재사용.
- 반복 값 파일: 각 실험 폴더의 `repeats/runs_index.csv`(실행별) · `repeats/summary_by_setting.csv`(설정별 평균·표준편차) · `repeats/curves/<설정>.csv`(라운드별 회차 값·평균).
- 그림: 곡선은 완료된 반복의 라운드별 평균, Fig.5는 τ별 반복 값 전체로 박스플롯, 표(Table 2·4, Fig.11)는 반복별 값의 평균.
- tmux k30: 3 reps(175) · 4 sync(10분마다 176 결과 가져와 그림 재생성). 176: `reps_console.log`.
