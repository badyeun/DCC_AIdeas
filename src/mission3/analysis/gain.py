import csv, json, ast
from pathlib import Path
from collections import Counter
T=("고열","구토","두통","복통","어지러움","열상","오심","전신쇠약","호흡곤란")
vd=Path(r"C:\DevBox\projects\DCC_AIdeas\data\대학부 데이터\Validation\2.라벨링데이터\VL_서울_구급")
truth={p.name:[s for s in T if s in {str(x).strip() for x in (json.loads(p.read_text(encoding="utf-8")).get("symptom") or [])}] for p in vd.rglob("*.json")}
pred={}
with open(r"C:\DevBox\projects\DCC_AIdeas\results\mission3\ver4\mission3.csv",encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): pred[r["label file name"]]=ast.literal_eval(r["symptom"])

def macro(pr):
    tot=0
    for s in T:
        tp=sum(1 for k in truth if s in truth[k] and s in pr[k])
        fp=sum(1 for k in truth if s not in truth[k] and s in pr[k])
        fn=sum(1 for k in truth if s in truth[k] and s not in pr[k])
        tot+= 2*tp/(2*tp+fp+fn) if tp else 0
    return tot/9
base=macro(pred); print(f"현재 Macro F1              : {base:.6f}")

empty=[k for k,v in pred.items() if not v]
# 상한: 빈 예측 108건에 정답 1개씩 정확히 채워 넣었을 때
up=dict(pred)
for k in empty: up[k]=[truth[k][0]]
print(f"빈 예측 108건을 '정답 1개'로 채움 (완벽 가정, 상한): {macro(up):.6f}  ({macro(up)-base:+.6f})")
# 현실: 그 통화들의 정답 분포를 모르니, 가장 흔한 증상(어지러움)으로 일괄
for frac in (1.0,0.5,0.3):
    g=dict(pred); hit=int(len(empty)*frac)
    for i,k in enumerate(empty): g[k]=[truth[k][0]] if i<hit else ["어지러움"]
    print(f"   그중 {frac:.0%}만 맞고 나머지는 오답일 때: {macro(g):.6f}  ({macro(g)-base:+.6f})")

print()
# 오심 과잉예측을 줄이면? 오심 예측 727개 중 494개가 오답
for drop in (0.2,0.4,0.6):
    g={k:[s for s in v] for k,v in pred.items()}
    # 오답인 오심 예측을 drop 비율만큼 제거 (상한 성격: 오답만 골라 지움)
    wrong=[k for k in g if "오심" in g[k] and "오심" not in truth[k]]
    for k in wrong[:int(len(wrong)*drop)]: g[k]=[s for s in g[k] if s!="오심"]
    print(f"오심 오답 예측의 {drop:.0%}를 제거하면: {macro(g):.6f}  ({macro(g)-base:+.6f})")
print()
g={k:[s for s in v] for k,v in pred.items()}
for k in g:
    if "오심" in g[k] and "오심" not in truth[k]: g[k]=[s for s in g[k] if s!="오심"]
print(f"오심 오답을 전부 제거(완전 상한): {macro(g):.6f}  ({macro(g)-base:+.6f})")
