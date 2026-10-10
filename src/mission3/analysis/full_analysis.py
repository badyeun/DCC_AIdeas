"""Mission 3: does each symptom label have evidence in the transcript?

Core question: the model can only read utterances[].text. If a label is assigned
from information that never appears in the transcript, no model can recover it.
We measure, per symptom, how often a generous keyword set fires in the text of
calls carrying that label -- and compare that against the model's actual F1.
"""
from __future__ import annotations
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

TARGETS = ("고열","구토","두통","복통","어지러움","열상","오심","전신쇠약","호흡곤란")

# Deliberately GENEROUS: we want an upper bound on "evidence is present".
CUES = {
"고열": [r"열.{0,4}[나났넘]",r"고열",r"발열",r"체온",r"\d+\s*도",r"뜨겁",r"뜨거",r"불덩이",r"미열",r"열이",r"열은",r"열을",r"오한",r"덜덜"],
"구토": [r"토했",r"토하",r"게웠",r"게우",r"게워",r"토물",r"구토",r"오바이트",r"올렸",r"넘어와",r"쏟아"],
"두통": [r"머리.{0,8}아[파프픈픔팠]",r"두통",r"머리.{0,6}[지찌]끈",r"편두통",r"골치",r"뒷골",
        r"머리.{0,6}깨",r"머리.{0,6}띵",r"머리.{0,6}무겁",r"머리.{0,6}통증",r"머리.{0,6}쑤",r"머리.{0,6}흔들"],
"복통": [r"배.{0,8}아[파프픈픔팠]",r"복통",r"아랫배",r"윗배",r"명치",r"속.{0,4}아[파프픈]",
        r"배앓이",r"위경련",r"장염",r"배탈",r"배.{0,6}통증",r"배.{0,6}쥐어"],
"어지러움": [r"어지",r"핑.{0,3}돌",r"빙빙",r"현기증",r"휘청",r"어질",r"중심.{0,4}못"],
"열상": [r"찢어",r"베였",r"베어",r"베인",r"칼에",r"피가",r"출혈",r"상처",r"열상",r"찔렸",r"긁혔",r"꿰매",r"피를",r"피.{0,3}나",r"터졌"],
"오심": [r"메스껍",r"매스껍",r"메슥",r"메쓰",r"울렁",r"느글",r"니글",r"구역질",r"구역감",r"토할",
        r"속.{0,4}안.{0,3}좋",r"체했",r"급체",r"체해",r"오심",r"속.{0,4}불편",r"속.{0,4}미식",r"헛구역"],
"전신쇠약": [r"기운.{0,4}없",r"힘.{0,4}없",r"축.{0,3}처",r"쇠약",r"탈진",r"기력",r"못.{0,3}일어",
          r"못.{0,3}움직",r"늘어져",r"처져",r"기진",r"쳐져",r"쓰러",r"몸.{0,4}못"],
"호흡곤란": [r"숨.{0,6}[차찬참쉬못가]",r"호흡.{0,4}[곤어힘안]",r"헐떡",r"가쁘",r"숨.{0,4}답답",r"산소",r"컥컥",r"쌕쌕",r"호흡곤란",r"과호흡"],
}
CC = {k:[re.compile(p) for p in v] for k,v in CUES.items()}

# Model F1 on official Validation (ver4 = current final submission)
F1 = {"고열":0.6973,"구토":0.6088,"두통":0.5497,"복통":0.8254,"어지러움":0.6590,
      "열상":0.8936,"오심":0.4031,"전신쇠약":0.5933,"호흡곤란":0.6795}


def has(t, s): return any(r.search(t) for r in CC[s])


def load(d: Path):
    out=[]
    for p in sorted(d.rglob("*.json")):
        try: a=json.loads(p.read_text(encoding="utf-8"))
        except Exception: continue
        u=a.get("utterances") or a.get("utterences") or []
        t="\n".join(x.get("text","").strip() for x in u
                    if isinstance(x,dict) and isinstance(x.get("text"),str) and x.get("text").strip())
        raw=[str(s).strip() for s in (a.get("symptom") or []) if str(s).strip()]
        out.append({"name":p.name,"t":t,"raw":raw,"y":[s for s in TARGETS if s in set(raw)]})
    return out


def main(base: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    R={}
    for tag,sub in (("Training","Training/2.라벨링데이터/TL_서울_구급"),
                    ("Validation","Validation/2.라벨링데이터/VL_서울_구급")):
        calls=load(base/sub); n=len(calls); R[tag]={"n":n}
        cnt=Counter(s for c in calls for s in c["y"])

        # ---- 1. evidence rate per symptom -------------------------------
        rows=[]
        for s in TARGETS:
            pos=[c for c in calls if s in c["y"]]
            neg=[c for c in calls if s not in c["y"]]
            ev=sum(1 for c in pos if has(c["t"],s))
            ev_neg=sum(1 for c in neg if has(c["t"],s))
            tp,fp,fn=ev,ev_neg,len(pos)-ev
            rows.append({"symptom":s,"support":len(pos),"rate":round(len(pos)/n,4),
                "label_with_evidence":ev,
                "evidence_rate":round(ev/len(pos),4) if pos else None,
                "label_without_evidence":len(pos)-ev,
                "evidence_without_label":ev_neg,
                "keyword_precision":round(tp/(tp+fp),4) if tp+fp else None,
                "keyword_f1":round(2*tp/(2*tp+fp+fn),4) if tp else None,
                "model_f1":F1[s]})
        R[tag]["evidence"]=rows

        # ---- 2. 구토 vs 오심 -------------------------------------------
        ct=Counter()
        for c in calls:
            ct[(has(c["t"],"구토"),has(c["t"],"오심"),"구토" in c["y"],"오심" in c["y"])]+=1
        R[tag]["vomit_nausea"]=[{"cue_구토":k[0],"cue_오심":k[1],"lab_구토":k[2],"lab_오심":k[3],"n":v}
                                 for k,v in sorted(ct.items(),key=lambda x:-x[1])]
        v,nz=cnt["구토"],cnt["오심"]
        both=sum(1 for c in calls if "구토" in c["y"] and "오심" in c["y"])
        R[tag]["vomit_nausea_pair"]={"구토":v,"오심":nz,"both":both,
            "P(오심|구토)":round(both/v,4) if v else None,
            "P(구토|오심)":round(both/nz,4) if nz else None,
            "오심_only":nz-both,"구토_only":v-both}

        # ---- 3. co-occurrence ------------------------------------------
        co=defaultdict(int)
        for c in calls:
            for i,a in enumerate(c["y"]):
                for b in c["y"][i+1:]: co[tuple(sorted((a,b)))]+=1
        R[tag]["cooccur"]=sorted(({"pair":list(k),"n":vv,
            "P(b|a)":round(vv/cnt[k[0]],4),"P(a|b)":round(vv/cnt[k[1]],4)}
            for k,vv in co.items()),key=lambda d:-d["n"])[:20]

        # ---- 4. non-target labels --------------------------------------
        nt=Counter(s for c in calls for s in c["raw"] if s not in TARGETS)
        R[tag]["nontarget"]={"kinds":len(nt),"top":nt.most_common(25),
            "calls_with_nontarget":sum(1 for c in calls if any(s not in TARGETS for s in c["raw"]))}
        R[tag]["n_label_hist"]=dict(sorted(Counter(len(c["y"]) for c in calls).items()))
        R[tag]["zero_label_calls"]=sum(1 for c in calls if not c["y"])

        # ---- 4b. calls whose transcript carries NO cue for ANY of its labels
        blind=[c for c in calls if c["y"] and not any(has(c["t"],s) for s in c["y"])]
        R[tag]["blind_calls"]={"n":len(blind),"pct":round(len(blind)/n,4),
            "mean_chars":round(sum(len(c["t"]) for c in blind)/len(blind),1) if blind else 0,
            "label_counter":Counter(s for c in blind for s in c["y"]).most_common()}
        lens=sorted(len(c["t"]) for c in calls)
        R[tag]["text_len"]={"mean":round(sum(lens)/n,1),"p10":lens[n//10],"p50":lens[n//2],
            "p90":lens[n*9//10],"under_200_chars":sum(1 for l in lens if l<200)}

        # ---- 5. identical transcript, different labels ------------------
        by=defaultdict(list)
        for c in calls:
            k=re.sub(r"\s+"," ",c["t"]).strip()
            if len(k)>40: by[k].append(c)
        dup={k:v for k,v in by.items() if len(v)>1}
        conf=[(k,v) for k,v in dup.items() if len({tuple(x["y"]) for x in v})>1]
        R[tag]["dup"]={"groups":len(dup),"calls":sum(len(v) for v in dup.values()),
            "conflicting_groups":len(conf),
            "examples":[{"labelsets":[list(x) for x in {tuple(x["y"]) for x in v}],
                         "files":[x["name"] for x in v][:4],"text":k[:200]} for k,v in conf[:10]]}

        if tag=="Training":
            # samples for human review
            L=[]
            def dump(title,sel,k=8):
                L.append(f"\n{'='*100}\n## {title}  (총 {len(sel)}건 중 {min(k,len(sel))}건)\n{'='*100}")
                for c in sel[:k]:
                    L.append(f"\n[{c['name']}] 대상라벨={c['y']} 원본={c['raw']}\n{c['t'][:1000]}")
            dump("오심 라벨 + 오심/구토 표현 모두 없음",
                 [c for c in calls if "오심" in c["y"] and not has(c["t"],"오심") and not has(c["t"],"구토")])
            dump("구토 표현 있는데 구토 라벨 없고 오심만",
                 [c for c in calls if has(c["t"],"구토") and "구토" not in c["y"] and "오심" in c["y"]])
            dump("두통 라벨 + 두통 표현 없음",
                 [c for c in calls if "두통" in c["y"] and not has(c["t"],"두통")])
            dump("두통 표현 있는데 두통 라벨 없음",
                 [c for c in calls if has(c["t"],"두통") and "두통" not in c["y"]])
            dump("열상 라벨 + 열상 표현 있음 (대조군: 잘 맞는 증상)",
                 [c for c in calls if "열상" in c["y"] and has(c["t"],"열상")], 3)
            (outdir/"samples.txt").write_text("\n".join(L),encoding="utf-8")

    (outdir/"full_analysis.json").write_text(json.dumps(R,ensure_ascii=False,indent=2),encoding="utf-8")

    # ---- console summary ----
    for tag in R:
        e=R[tag]["evidence"]
        print(f"\n{'='*104}\n{tag}  통화 {R[tag]['n']:,}개\n{'='*104}")
        print(f"{'증상':<10}{'빈도':>7}{'비율':>8}{'전사근거있음':>13}{'근거율':>9}{'근거없음':>10}{'키워드F1':>10}{'모델F1':>9}")
        for r in sorted(e,key=lambda x:-x["evidence_rate"]):
            print(f"{r['symptom']:<10}{r['support']:>7,}{r['rate']:>8.1%}{r['label_with_evidence']:>13,}"
                  f"{r['evidence_rate']:>9.1%}{r['label_without_evidence']:>10,}"
                  f"{r['keyword_f1']:>10.3f}{r['model_f1']:>9.3f}")
        p=R[tag]["vomit_nausea_pair"]
        print(f"\n  구토 {p['구토']:,} / 오심 {p['오심']:,} / 둘다 {p['both']:,}"
              f"  P(오심|구토)={p['P(오심|구토)']:.1%}  P(구토|오심)={p['P(구토|오심)']:.1%}")
        d=R[tag]["dup"]
        print(f"  전사 중복: {d['groups']:,}그룹 {d['calls']:,}통화, 그중 라벨 불일치 {d['conflicting_groups']:,}그룹")
        print(f"  증상 0개 통화: {R[tag]['zero_label_calls']:,}  라벨 개수분포: {R[tag]['n_label_hist']}")
        b=R[tag]["blind_calls"]; tl=R[tag]["text_len"]
        print(f"  자기 라벨 중 어느 것의 근거도 전사에 없는 통화: {b['n']:,} ({b['pct']:.1%}), 평균 {b['mean_chars']:.0f}자")
        print(f"  전사 길이(자): 평균 {tl['mean']:.0f}  p10 {tl['p10']}  중앙 {tl['p50']}  p90 {tl['p90']}  200자미만 {tl['under_200_chars']:,}")

    # correlation evidence_rate vs model_f1 (Training)
    e=R["Training"]["evidence"]
    xs=[r["evidence_rate"] for r in e]; ys=[r["model_f1"] for r in e]
    mx,my=sum(xs)/9,sum(ys)/9
    num=sum((a-mx)*(b-my) for a,b in zip(xs,ys))
    den=(sum((a-mx)**2 for a in xs)*sum((b-my)**2 for b in ys))**.5
    print(f"\n전사근거율 vs 모델F1 상관계수(피어슨) = {num/den:.3f}")


if __name__=="__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
