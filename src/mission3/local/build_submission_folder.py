"""미션 3 제출 폴더를 조립하고, 규격 항목이 전부 들어갔는지 검사한다.

출제문제 12쪽이 요구하는 제출물:
  1. inference.py (11쪽의 실행 규격을 만족)
  2. 학습 코드 파일 (학습 로그가 포함된 .ipynb)
  3. 학습 완료된 모델 가중치 (.pt / .pth / .ckpt)
  4. 모델 코드 파일 (.py)
  5. requirements.txt
  6. 결과 정리 PPT  <- 미션 1·2·3 통합 단일 파일이므로 이 폴더에 넣지 않는다

운영 메일 추가 사항:
  * 미션별 코드 폴더를 만들어 총 3개 폴더 + pptx 1개로 제출
  * 미션별 inference.py 실행 한 번으로 결과 파일 생성
  * 사전학습 모델을 쓰면 tokenizer / config / weight 를 제출물에 포함
  * requirements.txt 로 환경 재현 가능
  * README 또는 코드 주석으로 실행 방법 설명 (권장)
  * 코드 내 불필요부분 삭제, 주석 필수

가중치는 복사하지 않고 **하드링크**를 건다. 5.82 GB 를 한 번 더 쓰면 디스크가
모자라고, 같은 내용을 두 벌 두면 어느 쪽이 최신인지 헷갈린다.

Run:  python src/mission3/local/build_submission_folder.py [--dst ...] [--copy]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT / "src" / "mission3" / "local"
WEIGHTS = ROOT / "results" / "mission3" / "local" / "submission_final16_fp16"

# 제출 폴더에 들어갈 코드. (원본, 제출 폴더 안 경로, 역할)
CODE = [
    (LOCAL / "inference.py", "inference.py", "실행 진입점 (11쪽 규격)"),
    (LOCAL / "train_local.py", "train_local.py", "모델 정의 · 학습 코드"),
    (LOCAL / "common_local.py", "common_local.py", "공통 상수 · 데이터 적재"),
    (LOCAL / "normalize.py", "normalize.py", "동의어 통합 전처리"),
    (LOCAL / "tfidf_model.py", "tfidf_model.py", "TF-IDF 모델 적재 · 예측"),
    (LOCAL / "mission3_final.ipynb", "mission3_final.ipynb",
     "학습 코드 (학습 로그 포함)"),
    (ROOT / "docs" / "mission3_pkg11" / "requirements.txt", "requirements.txt",
     "실행 환경"),
]

README = """# Mission 3 — 환자 증상 인식

119 신고 대화의 전사 텍스트만 읽고 9개 증상을 예측합니다.
판정 임계값은 전 증상 **0.5** 입니다.

공식 Validation 3,640통화 실측 **Macro F1 = 0.65746**

---

## 실행

```bash
pip install -r requirements.txt

python inference.py \\
    --audio_dir {wav 폴더} \\
    --label_dir {주석 JSON 폴더} \\
    --ckpt_path ./weights \\
    --output ./outputs/mission3.csv
```

Mission 3 은 음성을 읽지 않지만 `--audio_dir` 를 받습니다(채점 인터페이스
호환). `--device cpu` 로 CPU 실행도 됩니다.

**네트워크가 필요하지 않습니다.** 체크포인트가 인코더 설정을 직접 들고 있어
사전학습 가중치를 받지 않으며, 토크나이저도 `weights/tokenizer/` 에
포함했습니다. `HF_HUB_OFFLINE=1` 에서 동일한 결과를 확인했습니다.

출력 CSV 형식은 `label file name,symptom` 이고, 증상 칸은 리스트 문자열
입니다(예: `['고열', '구토', '오심']`).

## 구성

| 파일 | 역할 |
|---|---|
| `inference.py` | 실행 진입점 |
| `train_local.py` | 모델 정의 · 학습 코드 |
| `common_local.py` | 공통 상수 · 데이터 적재 |
| `normalize.py` | 동의어 통합 전처리 (일부 멤버가 사용) |
| `tfidf_model.py` | TF-IDF 모델 적재 · 예측 |
| `mission3_final.ipynb` | 학습 코드 + 실제 학습 로그 |
| `requirements.txt` | 실행 환경 |
| `weights/*.ckpt` | 신경망 16개 (fp16 저장) |
| `weights/tfidf.pkl` | TF-IDF + 로지스틱 회귀 |
| `weights/tokenizer/` | KLUE 토크나이저 (16개 멤버 공용) |

## 모델

신경망 16개가 각자 9개 증상의 확률을 내고, 그 확률을 단순 평균한 뒤 TF-IDF
선형모델의 확률을 10% 섞어 0.5 에서 판정합니다.

```
최종 확률 = 0.9 × (신경망 16개 확률의 평균) + 0.1 × (TF-IDF 확률)
```

사전학습 체크포인트는 `klue/roberta-base`, `klue/roberta-large`,
`SangwonYoon/klue-roberta-large-tapt` 세 개이며 모두 오픈소스입니다.
멤버는 11개 계열로 나뉘고, 헤드(mean / attention / CLS)·학습률·전처리·
도메인 적응 여부가 서로 다릅니다. 자세한 내용과 실험 기록은
`mission3_final.ipynb` 에 있습니다.

## 계산 효율

| 항목 | 값 |
|---|---|
| Total 파라미터 | 2,902.70M |
| Active 파라미터 | 2,902.70M (앙상블이므로 모든 멤버가 모든 통화를 읽습니다) |
| GPU | NVIDIA GeForce RTX 5060 Ti, VRAM 8 GiB |
| CUDA / PyTorch | 12.8 / 2.11.0+cu128 · bf16 autocast |
| Validation batch size | 32 |
| 전체 추론 시간 | 6.1분 (3,640통화) |
| 통화당 평균 | 101 ms |

가중치는 fp16 으로 저장해 용량을 11.62 GB -> 5.82 GB 로 줄였습니다. 추론
시에는 fp32 로 적재되므로 연산 정밀도와 무관하며, 채점 환경이 fp16 연산을
지원하는지와 관계없이 동작합니다.

## 규칙 준수

- 입력은 `utterances[].text` 만. speaker · 시간 · 성별 · 주소 · 중증도 미사용
- Training 은 서울 데이터만. 외부 데이터 없음
- 공식 Validation 은 학습 · 보정에 사용하지 않음
- 판정 임계값 전 증상 0.5. 증상별 임계값 없음
- 상용 AI API 미사용. 오픈소스 사전학습 체크포인트만 사용
- 도메인 적응 MLM 도 Training 전사만 사용(라벨 미사용)
"""


def link_or_copy(src: Path, dst: Path, copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
        return "복사"
    try:
        os.link(src, dst)
        return "하드링크"
    except OSError:
        shutil.copy2(src, dst)
        return "복사(링크 실패)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default=str(ROOT / "_submit" / "mission3"))
    ap.add_argument("--copy", action="store_true",
                    help="하드링크 대신 실제 복사 (다른 드라이브로 옮길 때)")
    args = ap.parse_args()
    dst = Path(args.dst)

    if not WEIGHTS.is_dir():
        raise SystemExit(f"가중치 폴더가 없습니다: {WEIGHTS}")
    missing = [str(s) for s, _, _ in CODE if not s.exists()]
    if missing:
        raise SystemExit("원본 파일이 없습니다:\n  " + "\n  ".join(missing))

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    print(f"{'파일':<26}{'방식':>14}{'크기':>12}  역할")
    for src, rel, role in CODE:
        how = link_or_copy(src, dst / rel, args.copy)
        print(f"{rel:<26}{how:>14}{src.stat().st_size/1024:>10.1f} KB  {role}")

    (dst / "README.md").write_text(README, encoding="utf-8")
    print(f"{'README.md':<26}{'생성':>14}"
          f"{len(README.encode('utf-8'))/1024:>10.1f} KB  실행 방법 · 모델 설명")

    n = 0
    for f in sorted(WEIGHTS.rglob("*")):
        if f.is_file():
            link_or_copy(f, dst / "weights" / f.relative_to(WEIGHTS), args.copy)
            n += 1
    size = sum(f.stat().st_size for f in (dst / "weights").rglob("*")
               if f.is_file()) / 1e9
    print(f"{'weights/ (' + str(n) + '개)':<26}"
          f"{'하드링크' if not args.copy else '복사':>14}{size:>10.2f} GB  "
          f"가중치 · 토크나이저 · TF-IDF")

    print("\n=== 제출 규격 점검 (출제문제 12쪽) ===")
    checks = []
    checks.append(("inference.py", (dst / "inference.py").exists()))
    nb = dst / "mission3_final.ipynb"
    has_log = False
    if nb.exists():
        cells = json.loads(nb.read_text(encoding="utf-8"))["cells"]
        has_log = any(c.get("outputs") for c in cells)
    checks.append(("학습 코드 .ipynb", nb.exists()))
    checks.append(("  └ 학습 로그(셀 출력) 포함", has_log))
    ck = list((dst / "weights").glob("*.ckpt"))
    checks.append((f"가중치 .ckpt ({len(ck)}개)", len(ck) == 16))
    checks.append(("모델 코드 .py", (dst / "train_local.py").exists()))
    checks.append(("requirements.txt", (dst / "requirements.txt").exists()))
    checks.append(("토크나이저 포함 (오프라인용)",
                   (dst / "weights" / "tokenizer").is_dir()))
    checks.append(("TF-IDF 모델 포함", (dst / "weights" / "tfidf.pkl").exists()))
    checks.append(("README (실행 방법)", (dst / "README.md").exists()))
    ok = True
    for label, passed in checks:
        ok = ok and passed
        print(f"  [{'OK' if passed else '실패'}] {label}")
    print(f"\n조립 {'완료' if ok else '실패'}: {dst}")
    total = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e9
    print(f"  총 {total:.2f} GB · 파일 {sum(1 for f in dst.rglob('*') if f.is_file())}개")
    if not ok:
        raise SystemExit(1)
    print(f"\n다음 단계:")
    print(f"  python src/mission3/local/verify_submission.py --dir {dst}/weights")
    print(f"  (PPT 는 미션 1·2·3 통합 단일 파일이므로 이 폴더 바깥에 둡니다)")


if __name__ == "__main__":
    main()
