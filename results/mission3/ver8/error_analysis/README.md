# Training 내부 오답 검토

`review_samples_template.csv`는 오심·두통·전신쇠약별 FP 20개/FN 20개를
무작위 추출하는 기본 설정입니다. 해당 오류가 부족하면 있는 만큼만 추출합니다.
확신이 높은 오답만 뽑는 방식이 아닙니다. 이 CSV는 원본 정답을 변경하지 않습니다.

검토할 때는 `review_samples.csv`로 복사하여 review_category와 review_note를 채우세요.
재실행은 template만 갱신하며, 복사본은 읽거나 덮어쓰지 않습니다.

- `model_missed_evidence`: 전사에 근거가 있는데 모델이 놓침
- `question_negation_history`: 질문·부정·과거 증상을 잘못 해석
- `text_label_unclear`: 전사만으로 정답 근거 또는 라벨링 기준을 판단하기 어려움
- `other`: 기타; 구체적인 근거를 review_note에 기록

정답 불일치 의심과 확정된 라벨 오류를 구분하세요. 키워드 유무만으로 자동 판정하지 않습니다.
문장·원본 라벨·가중치·임계값을 이 검토 결과로 자동 변경하는 기능은 없습니다.
이 오류 집계는 원래 보정 집합에 이미 맞춘 임계값 기준이며 새 모델 성능 추정치가 아닙니다.
