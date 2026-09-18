# DCC AIdeas

2026 데이터+AI 크리에이터 캠프 예선 프로젝트입니다.

## 시작하기

1. `data/` 폴더에 제공받은 데이터를 둡니다. 데이터와 모델 파일은 Git에 올리지 않습니다.
2. 각 미션은 `notebooks/`의 노트북에서 실험합니다.
3. 필요한 패키지가 생기면 `requirements.txt`에 추가합니다.

Mission 1 baseline의 학습 및 주최 측 규격 추론 명령은 [Mission 1 안내](src/mission1/mission1.md)를 참고합니다.
Mission 2 MFCC+RandomForest baseline의 학습 및 제출 명령은 [Mission 2 안내](src/mission2/mission2.md)를 참고합니다.
Mission 3 TF-IDF+Logistic Regression baseline의 학습 및 제출 명령은 [Mission 3 안내](src/mission3/mission3.md)를 참고합니다.

## 노트북

- `mission1_gender.ipynb`: 신고자 성별 분류
- `mission2_speaker.ipynb`: 신고자/119대원 분류
- `mission3_symptom.ipynb`: 환자 증상 인식

## 주의

- 외부 데이터는 사용하지 않습니다.
- Validation 데이터는 학습에 사용하지 않습니다.
- 최종 제출 전에는 각 미션 결과와 실행 방법을 다시 확인합니다.
