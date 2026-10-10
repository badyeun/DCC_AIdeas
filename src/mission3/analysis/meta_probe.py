"""DIAGNOSTIC ONLY -- these fields must never be model inputs (competition rule).

If a symptom label tracks the dispatch record's own categories (triage,
disasterMedium, urgencyLevel) more tightly than it tracks the transcript, that
tells us the label was assigned from the record, not from what was said.
"""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

TARGETS = ("고열","구토","두통","복통","어지러움","열상","오심","전신쇠약","호흡곤란")
sys.path.insert(0, str(Path(__file__).parent))
from full_analysis import CC, has  # reuse the same cue sets

base = Path(sys.argv[1])
calls = []
for p in sorted((base/"Training/2.라벨링데이터/TL_서울_구급").rglob("*.json")):
    try: a = json.loads(p.read_text(encoding="utf-8"))
    except Exception: continue
    u = a.get("utterances") or a.get("utterences") or []
    t = "\n".join(x.get("text","").strip() for x in u
                  if isinstance(x,dict) and isinstance(x.get("text"),str) and x.get("text").strip())
    raw = {str(s).strip() for s in (a.get("symptom") or [])}
    calls.append({"t":t, "y":[s for s in TARGETS if s in raw],
                  "raw":raw,
                  "med":a.get("disasterMedium"), "urg":a.get("urgencyLevel"),
                  "tri":a.get("triage"), "sent":a.get("sentiment")})

n = len(calls)
print(f"통화 {n:,}개\n")

for field in ("med","urg","tri"):
    vals = Counter(c[field] for c in calls)
    print(f"=== {field} 분포 ===")
    for v,ct in vals.most_common(8): print(f"  {str(v):<24}{ct:7,} ({ct/n:5.1%})")
    print()

print("=== 각 증상이 특정 범주에 쏠리는 정도 ===")
print(f"{'증상':<10}{'빈도':>7}   가장 쏠린 disasterMedium / 그 안에서의 발생률 vs 전체 발생률")
for s in TARGETS:
    pos = [c for c in calls if s in c["y"]]
    base_rate = len(pos)/n
    by = defaultdict(lambda: [0,0])
    for c in calls:
        by[c["med"]][1] += 1
        if s in c["y"]: by[c["med"]][0] += 1
    cand = [(k,a/b,b) for k,(a,b) in by.items() if b >= n*0.02]
    cand.sort(key=lambda x:-x[1])
    k,rate,tot = cand[0]
    print(f"{s:<10}{len(pos):>7,}   {str(k):<22} {rate:6.1%} vs {base_rate:5.1%}  (배율 {rate/base_rate:4.2f})")

print("\n=== 오심: 전사 근거 유무별로 다른 메타데이터를 갖는가 ===")
nau = [c for c in calls if "오심" in c["y"]]
with_ev = [c for c in nau if has(c["t"],"오심")]
no_ev   = [c for c in nau if not has(c["t"],"오심")]
print(f"  오심 라벨 {len(nau):,}건 = 전사근거 있음 {len(with_ev):,} / 없음 {len(no_ev):,}")
for name, grp in (("근거있음",with_ev), ("근거없음",no_ev)):
    if not grp: continue
    med = Counter(c["med"] for c in grp).most_common(3)
    co  = Counter(s for c in grp for s in c["y"] if s!="오심").most_common(4)
    vom = sum(1 for c in grp if has(c["t"],"구토"))
    print(f"  [{name}] med={med}")
    print(f"           동반라벨={co}  구토표현있음={vom/len(grp):.1%}")

print("\n=== 원본 symptom 목록에 오심과 함께 오는 9개 외 라벨 ===")
print(Counter(x for c in nau for x in c["raw"] if x not in TARGETS).most_common(12))

print("\n=== 구토/오심 라벨이 '항상 같이' 오는가 ===")
v = sum(1 for c in calls if "구토" in c["y"])
z = sum(1 for c in calls if "오심" in c["y"])
b = sum(1 for c in calls if "구토" in c["y"] and "오심" in c["y"])
print(f"  구토만 {v-b:,} / 오심만 {z-b:,} / 둘다 {b:,}")
print(f"  구토 표현이 전사에 있는 통화 중 라벨 분포:")
sub = [c for c in calls if has(c["t"],"구토")]
print(f"    전체 {len(sub):,}  구토만 {sum(1 for c in sub if '구토' in c['y'] and '오심' not in c['y']):,}"
      f"  오심만 {sum(1 for c in sub if '구토' not in c['y'] and '오심' in c['y']):,}"
      f"  둘다 {sum(1 for c in sub if '구토' in c['y'] and '오심' in c['y']):,}"
      f"  둘다아님 {sum(1 for c in sub if '구토' not in c['y'] and '오심' not in c['y']):,}")
