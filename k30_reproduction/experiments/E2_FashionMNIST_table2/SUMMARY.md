# E2. FashionMNIST — Table 2(FashionMNIST 열), Fig. 7(c), Fig. 11(K=30), Table 4(FashionMNIST)

마지막 갱신: 2026-10-26 10:33

**재현 설정**: K=30, wise-175 단일 노드, CPU 카테고리 1·2·3·4·8코어 × 6명(논문 K=30 구성), batch 32, local epoch 5, τ=9, ζ=0.9, non-IID(gaussian), 설정당 반복 10회 목표(논문: average of ten repetitions)

## 비교 그림 (왼쪽: 논문 원본, 오른쪽: 재현)

![compare_fig7.png](compare_fig7.png)

![compare_fig11.png](compare_fig11.png)

![compare_table2.png](compare_table2.png)

![compare_table4.png](compare_table4.png)

## 반복 실행 상태 (그림은 완료된 반복의 평균 곡선, 표는 반복별 값의 평균)

| 설정 (1회차 이름) | 방법 | C | 완료 반복 | 최고 정확도 평균 ± 표준편차 | 소요 평균(분) |
|---|---|---:|---:|---:|---:|
| `FMNIST_BSA_C0.1` | BSA | 0.1 | 2/10 | 62.26 ± 1.38 | 32 |
| `FMNIST_BSA_C0.3` | BSA | 0.3 | 2/10 | 74.47 ± 0.02 | 198 |
| `FMNIST_BSA_C0.5` | BSA | 0.5 | 2/10 | 78.43 ± 1.14 | 311 |
| `FMNIST_SASAFL_C0.1` | SASAFL | 0.1 | 2/10 | 60.61 ± 0.89 | 161 |
| `FMNIST_SASAFL_C0.3` | SASAFL | 0.3 | 2/10 | 71.14 ± 1.21 | 350 |
| `FMNIST_SASAFL_C0.5` | SASAFL | 0.5 | 2/10 | 81.60 ± 0.24 | 689 |

반복 값 파일: `repeats/runs_index.csv`(실행별), `repeats/summary_by_setting.csv`(설정별 평균·표준편차), `repeats/curves/<설정>.csv`(라운드별 회차 값과 평균)

## Table 2 비교

| FashionMNIST (목표 70%) | 논문 BSA | 논문 SASAFL | 재현 BSA | 재현 SASAFL |
|---|---:|---:|---:|---:|
| **C = 0.1** | | | | |
| Accuracy | 54.72 | **74.09** | **62.26** | 60.61 |
| Round-to-accuracy | - | **248** | - | - |
| Time-to-accuracy | - | **105.24** | - | - |
| **C = 0.3** | | | | |
| Accuracy | 66.00 | **71.59** | **74.47** | 71.14 |
| Round-to-accuracy | - | **420** | 363 | **313** |
| Time-to-accuracy | - | **178.80** | **144.54** | 222.60 |
| **C = 0.5** | | | | |
| Accuracy | 73.83 | **73.94** | 78.43 | **81.60** |
| Round-to-accuracy | 376 | **283** | 157 | **91** |
| Time-to-accuracy | 185.34 | **142.21** | **99.52** | 123.49 |

주의:
- FMNIST_BSA_C0.1: 완료 반복 2/10 (완료된 반복만 평균)
- FMNIST_BSA_C0.3: 완료 반복 2/10 (완료된 반복만 평균)
- FMNIST_BSA_C0.5: 완료 반복 2/10 (완료된 반복만 평균)
- FMNIST_SASAFL_C0.1: 완료 반복 2/10 (완료된 반복만 평균)
- FMNIST_SASAFL_C0.3: 완료 반복 2/10 (완료된 반복만 평균)
- FMNIST_SASAFL_C0.5: 완료 반복 2/10 (완료된 반복만 평균)

- Accuracy = 최고 test 정확도, Round-to-accuracy = 목표 첫 도달 라운드, Time-to-accuracy = 서버 setup 시작부터 목표 도달까지(분). `-` = 도달 못 함/미실행
- Time-to-accuracy는 단일 노드 과다 할당(thread 109개 / 논리 CPU 32개) 환경이라 논문과 절대값 비교가 어렵다.

### 관찰

- C=0.1: 정확도 차이(SASAFL−BSA) 논문 +19.37%p, 재현 -1.66%p → **논문과 반대 방향**
- C=0.3: 정확도 차이(SASAFL−BSA) 논문 +5.59%p, 재현 -3.33%p → **논문과 반대 방향**
  - round-to-accuracy BSA/SASAFL = 1.16배 (논문 BSA 미도달) → 1보다 크면 SASAFL이 더 빨리 도달
- C=0.5: 정확도 차이(SASAFL−BSA) 논문 +0.11%p, 재현 +3.17%p → 논문과 같은 방향
  - round-to-accuracy BSA/SASAFL = 1.72배 (논문 1.33배) → 1보다 크면 SASAFL이 더 빨리 도달

## Table 4 (K=30 행)

SASAFL로 500라운드를 마치는 데 걸린 시간(분).

|  | C=0.1 | C=0.3 | C=0.5 |
|---|---:|---:|---:|
| 논문 K=30 (500R) | 212 | 213 | 251 |
| 재현 K=30 (500R) | 161 | 350 | 689 |

<!-- /table4 -->

