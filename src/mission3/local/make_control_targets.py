"""증류의 대조군: 교사 없이 '그냥 부드럽게만' 만든 타깃.

증류가 이기면 그게 **교사가 아는 것** 때문인지 **타깃이 부드러워진 것**
때문인지 구분해야 합니다. 여기서 만드는 타깃은 모든 통화에 대해 같은 값
(해당 증상의 기저율)이라 교사 지식이 0입니다. 같은 alpha로 섞어 비교하면
증류의 이득 중 순수한 label smoothing 몫이 분리됩니다.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common_local import TARGETS, TRAIN_DIR, load, split

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
names, texts, Y = load(TRAIN_DIR, "train")
tr, _ = split(texts, Y, seed=2026)
rate = Y[tr].mean(axis=0)                      # 학습 분할에서만 계산
soft = np.tile(rate, (len(Y), 1)).astype(np.float32)
np.save(R / "soft_targets_constant.npy", soft)
print("증상별 기저율 (상수 타깃):")
for t, v in zip(TARGETS, rate):
    print(f"  {t:<8} {v:.4f}")
print(f"\n저장: soft_targets_constant.npy  {soft.shape}")
print("alpha 0.3이면 양성 타깃 0.7+0.3*기저율, 음성 타깃 0.3*기저율이 됩니다.")
