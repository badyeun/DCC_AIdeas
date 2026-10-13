# DCC AIdeas

2026 데이터+AI 크리에이터 캠프 예선 프로젝트입니다.

## 프로젝트 종료 기록 — 2026-10-13

코드와 실험 기록을 보존하는 저장소입니다. 최신 Mission 3 실험은 **ver29**이며,
Modal L4에서 후보 비교를 완료했습니다. 목표였던 **Macro F1 0.7 초과는 확인하지 못했습니다.**

- [프로젝트 종료 요약](records/mission3/project_closeout.txt)
- [ver1~ver9 실험 기록](src/mission3/experiment_log.md)
- [ver1~ver24 인수인계 및 평가 조건](docs/mission3_handoff_through_ver24.txt)
- [ver24~ver29 결과·검증 기록](records/mission3/)
- [최신 Modal 실행 안내](docs/mission3_ver29_modal_guide.md)
- [최신 Modal 노트북](src/mission3/mission3_ver29_modal.ipynb)

| 평가 범위 | 실험 | Macro F1 | 의미 |
| --- | --- | ---: | --- |
| 공식 Validation 3,640건 | ver25 전체 학습, Qwen 단독 | 0.654785 | 후반 고정 판정 실험의 전체 학습 결과 |
| 내부 development 4,380건 | ver28 KLUE + KURE | 0.655299 | 후보 선택용 점수 |
| 내부 confirmation 4,380건 | ver28 KLUE + KURE | 0.645243 | 선택 후 확인용 점수 |
| 내부 development 4,380건 | ver29 네 모델 평균 | 0.657211 | 후보 선택용 점수 |
| 내부 confirmation 4,380건 | ver29 네 모델 평균 | 0.645462 | ver28 대비 +0.000219, 뚜렷한 개선 미확인 |

서로 다른 평가 범위의 점수를 직접 순위 비교하지 않습니다. ver29는 `screening_complete`이며
전체 Training 재학습·공식 Validation 평가·최종 제출 모델 생성까지 끝난 결과가 아닙니다.
공식 대회 Test/리더보드 결과는 이 저장소의 위 수치로 대신하지 않습니다.

원본 음성·통화 JSON, 학습 가중치, 입력 ZIP, 행별 예측 및 캐시는 이번 업로드에서 제외했습니다.
결과 문서에 쓰인 파일 해시와 경로는 분석 당시의 기록이며, 해당 대형 파일이 GitHub에 포함됐다는 뜻이 아닙니다.
**로컬 파일 삭제는 취소되어 데이터·모델·압축파일을 그대로 보관하고 있습니다.**

## 시작하기

1. `data/` 폴더에 제공받은 데이터를 둡니다. 데이터와 모델 파일은 Git에 올리지 않습니다.
2. 초기 노트북은 `notebooks/`, 버전별 Mission 3 노트북과 학습 코드는 `src/mission3/`에서 확인합니다.
3. 재실행할 때는 해당 버전의 README와 `requirements.txt`를 확인합니다. 원본 데이터와 가중치는 별도로 필요합니다.

Mission 1 baseline의 학습 및 주최 측 규격 추론 명령은 [Mission 1 안내](src/mission1/mission1.md)를 참고합니다.
Mission 2 MFCC+RandomForest baseline의 학습 및 제출 명령은 [Mission 2 안내](src/mission2/mission2.md)를 참고합니다.

## 노트북

- `mission1_gender.ipynb`: 신고자 성별 분류
- `mission2_speaker.ipynb`: 신고자/119대원 분류
- `mission3_symptom.ipynb`: 환자 증상 인식

## 주의

- 외부 데이터는 사용하지 않습니다.
- Validation 데이터는 학습에 사용하지 않습니다.
- 최종 제출 전에는 각 미션 결과와 실행 방법을 다시 확인합니다.
