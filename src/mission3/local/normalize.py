"""Korean spoken-language normalisation for 119 call transcripts.

Applied identically to every sample, and never conditioned on the label, so it
stays inside the rule "샘플마다 label에 따라 다른 전처리 금지".

The synonym groups come from the surface forms actually observed in the Seoul
Training transcripts while analysing why 오심/구토/두통 score badly
(see src/mission3/data_analysis_label_evidence.md).
"""
from __future__ import annotations

import re

# Each entry maps many spoken spellings onto one canonical token. Canonical
# tokens are ordinary Korean words, so character n-grams still see real text.
SYNONYMS: list[tuple[str, list[str]]] = [
    ("구토함", [r"토했\w*", r"토하\w*", r"토한\w*", r"게웠\w*", r"게우\w*", r"게워\w*",
              r"오바이트\w*", r"토물", r"구토"]),
    ("메스꺼움", [r"메스껍\w*", r"매스껍\w*", r"메슥\w*", r"메쓰꺼\w*", r"울렁\w*", r"느글\w*",
               r"니글\w*", r"구역질", r"구역감", r"헛구역\w*", r"오심",
               r"토할\s*(것|거)\s*같\w*", r"토할라\w*", r"체했\w*", r"급체", r"체해\w*",
               r"속\s*이?\s*안\s*좋\w*", r"속\s*이?\s*미식\w*"]),
    ("머리아픔", [r"머리\s*가?\s*아[파프픈픔팠]\w*", r"머리\s*가?\s*지끈\w*", r"머리\s*가?\s*띵\w*",
               r"머리\s*가?\s*깨질\w*", r"머리\s*가?\s*쑤[시셔]\w*", r"두통", r"편두통",
               r"뒷골\s*이?\s*당기\w*"]),
    ("배아픔", [r"배\s*가?\s*아[파프픈픔팠]\w*", r"복통", r"아랫배\s*가?\s*아\w*",
              r"윗배\s*가?\s*아\w*", r"명치\s*가?\s*아\w*", r"배앓이", r"위경련", r"배탈\w*"]),
    ("어지러움", [r"어지러\w*", r"어지럽\w*", r"어질어질\w*", r"핑\s*돈\w*", r"핑\s*돌\w*",
               r"빙빙\s*돌\w*", r"현기증", r"휘청\w*"]),
    ("숨참", [r"숨\s*이?\s*차\w*", r"숨\s*을?\s*못\s*[쉬쉰]\w*", r"숨\s*쉬기\s*힘\w*",
            r"숨\s*이?\s*가[쁘빠]\w*", r"호흡\s*곤란", r"헐떡\w*", r"과호흡", r"숨\s*이?\s*막\w*"]),
    ("피남", [r"피\s*가?\s*나\w*", r"피\s*를?\s*흘\w*", r"출혈", r"찢어졌\w*", r"찢어지\w*",
            r"베였\w*", r"베인\w*", r"열상"]),
    ("열남", [r"열\s*이?\s*나\w*", r"열\s*이?\s*높\w*", r"고열", r"발열", r"불덩이\w*",
            r"미열", r"오한\w*"]),
    ("기운없음", [r"기운\s*이?\s*없\w*", r"힘\s*이?\s*없\w*", r"축\s*늘어\w*", r"축\s*처[졌지]\w*",
               r"탈진\w*", r"기력\s*이?\s*없\w*", r"못\s*일어나\w*", r"전신\s*쇠약"]),
]
_COMPILED = [(canon, [re.compile(p) for p in pats]) for canon, pats in SYNONYMS]

_REPEAT = re.compile(r"(.)\1{2,}")           # 아아아아 -> 아아
_FILLER = re.compile(r"\b(어+|음+|아+|그+|저기+|뭐+|네+|예+)\b")
_PRIVACY = re.compile(r"\*+\s*\[?개인정보\]?\s*\**|\[개인정보\]")
_SPACE = re.compile(r"\s+")


def normalize(text: str, *, synonyms=True, repeats=True, privacy=True, fillers=False) -> str:
    """전사 텍스트를 표준형으로 바꿉니다. 네 단계를 따로 켜고 끌 수 있습니다.

    앙상블 멤버마다 쓰는 변형이 다르므로(`VARIANTS` 참조), 체크포인트에
    기록된 변형을 추론 시 그대로 적용해야 합니다. 원문을 그대로 쓰는 멤버에
    정규화한 텍스트를 넣으면 학습·추론이 어긋납니다.

    Args:
        text: 한 통화의 발화를 줄바꿈으로 이어 붙인 문자열.
        synonyms: 같은 증상을 가리키는 표현을 대표 표현 하나로 통일합니다
            (예: "토할 것 같다" → "메스꺼움). 9개 대표 표현은 9개 증상에
            1:1 대응하며, 라벨 정보를 쓰지 않고 표현만 묶습니다.
        repeats: 세 번 이상 반복된 같은 글자를 두 번으로 줄입니다
            ("아아아아" → "아아"). 통화체에서 흔한 늘임을 줄여 토큰 수를
            아낍니다.
        privacy: 전화번호·주민번호 형태의 숫자열을 공백으로 지웁니다.
        fillers: "어", "음" 같은 간투사를 지웁니다. 기본값은 끔 -- 간투사가
            당황·호흡곤란의 단서일 수 있어, 쓰는 멤버와 안 쓰는 멤버를
            나눠 두었습니다.

    Returns:
        연속 공백이 하나로 정리된 문자열.
    """
    if privacy:
        text = _PRIVACY.sub(" ", text)
    if repeats:
        text = _REPEAT.sub(r"\1\1", text)
    if synonyms:
        for canon, patterns in _COMPILED:
            for rx in patterns:
                text = rx.sub(canon, text)
    if fillers:
        text = _FILLER.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


VARIANTS = {
    "raw":            dict(synonyms=False, repeats=False, privacy=False, fillers=False),
    "privacy":        dict(synonyms=False, repeats=False, privacy=True,  fillers=False),
    "privacy+repeat": dict(synonyms=False, repeats=True,  privacy=True,  fillers=False),
    "synonym":        dict(synonyms=True,  repeats=True,  privacy=True,  fillers=False),
    "synonym+filler": dict(synonyms=True,  repeats=True,  privacy=True,  fillers=True),
}
