def extract_requested_category(text: str) -> str | None:
    if not text:
        return None
    
    text_lower = text.lower()
    
    # 1. 키워드 매핑 (유사어 → 표준 카테고리)
    keyword_mapping = {
        "교육": ["교육", "자기계발", "배움", "학습", "강의", "수강", "훈련"],
        "일자리": ["일자리", "취업", "구직", "채용", "면접", "이력서"],
        "주거": ["주거", "월세", "전세", "집", "임대", "보증금", "주택"],  # ← 문제!
        "금융": ["금융", "대출", "저축", "통장", "돈", "자금"],
        "생활": ["생활", "생활비", "식비", "용돈"],
        "창업": ["창업", "사업", "자영업", "스타트업"],
        "돌봄": ["돌봄", "보육", "아이돌봄"],
        "정신건강": ["정신건강", "마음건강", "상담", "심리"],
    }
    
    # 2. 매핑된 키워드로 찾기
    for category, keywords in keyword_mapping.items():
        if any(kw in text_lower for kw in keywords):
            return category  # ← "월세" 있으면 바로 "주거" 반환!
    
    # 3. 기본 카테고리 리스트 (직접 매칭)
    CATEGORIES = ["교육", "보호", "돌봄", "생활지원", "정신건강", "일자리", 
                  "서민금융", "마음건강", "금융", "생활", "주거", "창업"]
    
    found = [c for c in CATEGORIES if c in text_lower]
    return found[-1] if found else None