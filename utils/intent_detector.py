"""
사용자 메시지에서 intent를 감지하는 모듈
"""

from config import (
    FAQ_KEYWORDS,
    APPLY_KEYWORDS,
    DETAIL_KEYWORDS,
    ELIGIBILITY_KEYWORDS,
    RESET_KEYWORDS,
)


def detect_intent(user_message: str, last_intent: str | None = None) -> str:
    """
    사용자 메시지에서 intent 감지
    
    Returns:
        "faq" | "apply" | "detail" | "eligibility" | "match"
    """
    text = user_message.strip().lower()
    
    # ✅ 0. 제도 규칙 질문 키워드 (FAQ 우선 처리)
    policy_question_keywords = ["각자", "각각", "같이", "둘이", "둘 다", "동시에", "중복"]
    has_policy_question = any(kw in text for kw in policy_question_keywords)
    
    # ✅ 1. 디테일 요청 (자세히)
    if any(k in text for k in DETAIL_KEYWORDS):
        return "detail"
    
    # ✅ 2. 신청 방법 요청
    if any(k in text for k in APPLY_KEYWORDS):
        return "apply"
    
    # ✅ 3. FAQ (일반적인 질문)
    if any(k in text for k in FAQ_KEYWORDS):
        return "faq"
    
    # ✅ 4. ELIGIBILITY (본인 자격 확인, 제도 질문 아닐 때만)
    if not has_policy_question:
        personal_pronouns = ["나", "저", "우리", "내가", "제가"]
        has_personal = any(p in text for p in personal_pronouns)
        
        eligibility_patterns = [
            "신청 가능",
            "신청할 수 있",
            "받을 수 있",
            "해당",
            "자격",
            "가능해",
            "가능한",
            "되나",
            "되는지",
        ]
    
    # 본인 자격 확인 의도가 명확하면 ELIGIBILITY
    if has_personal and any(pattern in text for pattern in eligibility_patterns):
        return "eligibility"
    
    # ELIGIBILITY 키워드만 있는 경우
    if any(k in text for k in ELIGIBILITY_KEYWORDS):
        return "eligibility"
    
    return "match"