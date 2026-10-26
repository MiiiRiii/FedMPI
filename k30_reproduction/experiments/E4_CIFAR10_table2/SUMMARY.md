# E4. CIFAR-10 — Table 2(CIFAR-10 열), Fig. 8(a), Fig. 9, Table 4(CIFAR-10)

마지막 갱신: 2026-10-26 10:33

**재현 설정**: K=30, wise-175 단일 노드, CPU 카테고리 1·2·3·4·8코어 × 6명(논문 K=30 구성), batch 32, local epoch 5, τ=9, ζ=0.9, non-IID(gaussian), 설정당 반복 10회 목표(논문: average of ten repetitions)

## 비교 그림 (왼쪽: 논문 원본, 오른쪽: 재현)

![compare_fig8.png](compare_fig8.png)

![compare_fig9.png](compare_fig9.png)

![compare_table2.png](compare_table2.png)

![compare_table4.png](compare_table4.png)

## 반복 실행 상태 (그림은 완료된 반복의 평균 곡선, 표는 반복별 값의 평균)

| 설정 (1회차 이름) | 방법 | C | 완료 반복 | 최고 정확도 평균 ± 표준편차 | 소요 평균(분) |
|---|---|---:|---:|---:|---:|
| `CIFAR10_BSA_C0.1` | BSA | 0.1 | 2/10 | 38.95 ± 2.45 | 37 |
| `CIFAR10_BSA_C0.3` | BSA | 0.3 | 2/10 | 47.37 ± 0.55 | 186 |
| `CIFAR10_BSA_C0.5` | BSA | 0.5 | 2/10 | 51.59 ± 1.20 | 318 |
| `CIFAR10_SASAFL_C0.1` | SASAFL | 0.1 | 2/10 | 41.30 ± 3.02 | 193 |
| `CIFAR10_SASAFL_C0.3` | SASAFL | 0.3 | 2/10 | 45.82 ± 1.47 | 321 |
| `CIFAR10_SASAFL_C0.5` | SASAFL | 0.5 | 2/10 | 52.55 ± 1.41 | 498 |

반복 값 파일: `repeats/runs_index.csv`(실행별), `repeats/summary_by_setting.csv`(설정별 평균·표준편차), `repeats/curves/<설정>.csv`(라운드별 회차 값과 평균)

## Table 2 비교

| CIFAR10 (목표 55%) | 논문 BSA | 논문 SASAFL | 재현 BSA | 재현 SASAFL |
|---|---:|---:|---:|---:|
| **C = 0.1** | | | | |
| Accuracy | 42.50 | **58.59** | 38.95 | **41.30** |
| Round-to-accuracy | - | **1,490** | - | - |
| Time-to-accuracy | - | **1,104.42** | - | - |
| **C = 0.3** | | | | |
| Accuracy | 51.08 | **58.25** | **47.37** | 45.82 |
| Round-to-accuracy | - | **1,728** | - | - |
| Time-to-accuracy | - | **1,281.03** | - | - |
| **C = 0.5** | | | | |
| Accuracy | **56.82** | 56.77 | 51.59 | **52.55** |
| Round-to-accuracy | 2,348 | **2,140** | - | - |
| Time-to-accuracy | 1,809.73 | **1,680.49** | - | - |

주의:
- CIFAR10_BSA_C0.1: 완료 반복 2/10 (완료된 반복만 평균)
- CIFAR10_BSA_C0.3: 완료 반복 2/10 (완료된 반복만 평균)
- CIFAR10_BSA_C0.5: 완료 반복 2/10 (완료된 반복만 평균)
- CIFAR10_SASAFL_C0.1: 완료 반복 2/10 (완료된 반복만 평균)
- CIFAR10_SASAFL_C0.3: 완료 반복 2/10 (완료된 반복만 평균)
- CIFAR10_SASAFL_C0.5: 완료 반복 2/10 (완료된 반복만 평균)

- Accuracy = 최고 test 정확도, Round-to-accuracy = 목표 첫 도달 라운드, Time-to-accuracy = 서버 setup 시작부터 목표 도달까지(분). `-` = 도달 못 함/미실행
- Time-to-accuracy는 단일 노드 과다 할당(thread 109개 / 논리 CPU 32개) 환경이라 논문과 절대값 비교가 어렵다.

### 관찰

- C=0.1: 정확도 차이(SASAFL−BSA) 논문 +16.09%p, 재현 +2.35%p → 논문과 같은 방향
- C=0.3: 정확도 차이(SASAFL−BSA) 논문 +7.17%p, 재현 -1.55%p → **논문과 반대 방향**
- C=0.5: 정확도 차이(SASAFL−BSA) 논문 -0.05%p, 재현 +0.96%p → **논문과 반대 방향**

**라운드 축소**: 논문은 CIFAR-10을 3000라운드 돌렸지만 여기서는 500라운드로 고정했다. 그래서 논문 그림과는 0~500라운드 구간만 비교할 수 있다.

## Table 4 (K=30 행)

SASAFL로 500라운드를 마치는 데 걸린 시간(분). **주의: 논문은 1000라운드 기준, 재현은 500라운드 기준이라 직접 비교 불가 (라운드당 시간 비교용: 논문 1000R 값을 500/1000배 하면 대략적인 기준이 된다).**

|  | C=0.1 | C=0.3 | C=0.5 |
|---|---:|---:|---:|
| 논문 K=30 (1000R) | 741 | 742 | 787 |
| 재현 K=30 (500R) | 193 | 321 | 498 |

<!-- /table4 -->

