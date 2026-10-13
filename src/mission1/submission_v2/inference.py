"""
Mission 1: 119 신고자 성별 분류 (M/F).

최종 모델 = WavLM 3모델 앙상블(Attention+F0 40%, Stats+F0 40%, Stats No-F0 20%)
          과 ECAPA 화자 임베딩 + 로지스틱 회귀를 25:75로 섞은 확률.

Usage:
    python inference.py \
        --audio_dir AUDIO_DIR \
        --label_dir LABEL_DIR \
        --ckpt_path ./ecapa_gender/ecapa_lr.pt \
        --output outputs/mission1.csv
"""

from pathlib import Path
import runpy


def main():
    target = Path(__file__).resolve().parent / "inference_ecapa_blend.py"
    if not target.is_file():
        raise FileNotFoundError(f"Missing inference script: {target}")
    # 공식 실행 인자는 그대로 전달된다 (sys.argv 공유).
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
