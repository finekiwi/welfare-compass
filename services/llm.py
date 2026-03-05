"""
LLM 서비스 - OpenAI API 호출 담당
"""

import json
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from config import (
    OPENAI_MODEL,
    EXTRACT_TEMPERATURE,
    RESPONSE_TEMPERATURE,
    MAX_TOKENS_EXTRACT,
    MAX_TOKENS_RESPONSE,
)

from services.faq import search_faq, format_faq_context

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ======================
# 사용자 정보 추출
# ======================

EXTRACT_SYSTEM_PROMPT = """⚠️⚠️⚠️ 중요: 반드시 JSON 형식으로만 출력하세요! ⚠️⚠️⚠️

당신은 사용자 메시지에서 정보를 추출하여 **순수 JSON 객체만** 반환하는 AI입니다.

절대로 설명, 텍스트, 마크다운 없이 JSON만 출력하세요.
사용자의 **이번 메시지에서만** 새로 언급된 정보를 추출하세요.

**출력 형식 예시:**
{"age": 27, "income": null, "residence": "서울 마포구", "employment_status": "구직중", "housing_type": "월세", "special_conditions": ["청년"], "needs": ["교육"], "household_size": 1, "has_children": false, "children_ages": []}

당신은 사용자의 메시지에서 복지 매칭에 필요한 정보를 추출하는 AI입니다.

다음 정보를 JSON 형식으로 추출하세요:
- age: 나이 (숫자, 없으면 null)
- income: 월소득 (숫자, 만원 단위, 없으면 null) - "백수/무직"이면 0
- income_type: "월" 또는 "연" (없으면 null)
- income_scope: "개인" 또는 "부부합산" (명시되지 않았으면 null)
- residence: 거주지역 - **반드시 구(區) 단위로 추출**
  * 형식: "서울 {구이름}" (예: "서울 강남구", "서울 마포구")
  * 역/동네 이름은 해당 구로 변환:
    - "신촌" → "서울 서대문구"
    - "홍대" → "서울 마포구"
    - "강남역" → "서울 강남구"
    - "잠실" → "서울 송파구"
    - "성수" → "서울 성동구"
    - "건대" → "서울 광진구"
  * 구 이름이 명확하지 않으면 대표 구로 매핑
  * 없으면 null
- employment_status: 고용상태 - 아래 규칙 적용:
  * "취준생", "취업준비", "구직중", "일자리 찾는 중" → "구직중"
  * "백수", "무직", "일 안 함" → "무직"  
  * "회사 다님", "직장인", "재직중" → "재직"
  * "대학생", "학교 다님" → "학생"
  * "프리랜서", "알바" → "프리랜서"
- housing_type: 주거형태 ("월세", "전세", "자가", "고시원", 없으면 null)
- special_conditions: 특수조건 리스트 (예: ["청년", "한부모", "장애인"], 없으면 [])
- needs: 사용자가 요청하거나 필요로 하는 지원 분야 배열 (없으면 [])
  ⚠️ 매우 중요: 사용자가 특정 분야를 명시적으로 언급하면 반드시 needs에 포함하세요!
  "교육 쪽" 포함 → needs: ["교육"]
  
  명시적 요청 패턴:
  * "~쪽 복지도 알아봐줘" → 해당 분야
  * "~관련은 없나" → 해당 분야
  * "~지원 알려줘" → 해당 분야
  * "~이 필요해" → 해당 분야
  * "~하고 싶어" → 해당 분야
  
  예시:
  * "교육쪽 복지도 알아봐줘" → ["교육"]
  * "IT 교육이 필요해" → ["교육"]
  * "취업하고 싶어요" → ["일자리"]
  * "공부하고 싶어요" → ["교육"]
  * "주거 관련 복지 알려줘" → ["주거"]
  * "일자리 찾고 있어" → ["일자리"]
  * "생활비 지원 필요해" → ["생활"]
  
  분야 키워드 매핑:
  * 교육, 공부, 배움, 학습, 훈련, 자격증, 자기계발, IT교육 → "교육"
  * 일자리, 취업, 구직, 채용 → "일자리"
  * 주거, 월세, 전세, 집, 임대 → "주거"
 
  ⚠️ 주의:
  * "27살이고 월세 살아요" ← 단순 정보 제공, needs는 []
  * "IT 교육이 필요해" ← 명시적 필요, needs: ["교육"]
  * 복수 needs 가능:
  * "취업도 하고 공부도 하고 싶어요" → ["일자리", "교육"]
- household_size: 함께 사는 가구원 수 (숫자, 없으면 null)
  * "저 혼자 살아요" → 1
  * "배우자랑 둘이 살아요" → 2
  * "아이 둘 있어요" → 4 (부부+아이2)
- has_children: 자녀 유무 (true/false, 없으면 null)
  * "아이가 있어요", "자녀 2명", "초등학생 아들" → true
  * "결혼했어요" (자녀 언급 없음) → null
  * "혼자 살아요", "신혼부부" (자녀 언급 없음) → false
- children_ages: 자녀 나이 배열 (예: [7, 10], 없으면 [])
  * "초등학생" → [8] (대략적 나이)
  * "중학생, 고등학생" → [14, 17]
  * 구체적 나이가 없으면 학년 기준 평균 나이 사용

대화 맥락을 고려하여 이전에 언급된 정보도 포함하세요.
반드시 유효한 JSON만 출력하세요. 다른 텍스트는 절대 포함하지 마세요.
JSON 외의 설명, 인사말, 마크다운 코드블록(```) 없이 순수 JSON만 출력하세요."""


def extract_user_info(user_message: str, conversation_history: list) -> dict:
    """GPT를 사용해 사용자 정보 추출"""
    
    messages = [{"role": "system", "content": EXTRACT_SYSTEM_PROMPT}]
    
    # 이전 대화 컨텍스트 추가 (최근 6개)
    for msg in conversation_history[-12:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=EXTRACT_TEMPERATURE,
            max_tokens=MAX_TOKENS_EXTRACT
        )
        
        result = response.choices[0].message.content.strip()

        # JSON 파싱 전처리 - 마크다운 코드블록 제거
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0]
        elif "```" in result:
            result = result.split("```")[1].split("```")[0]
        
        result = result.strip()
        user_info = json.loads(result)

        # 후처리: "~쪽" 패턴 강제 추출
        if 'needs' not in user_info or not user_info['needs']:
            needs = []
            msg_lower = user_message.lower()

            # 명시적 요청 패턴 체크 ("~쪽", "~관련" 등)
            if ('교육' in msg_lower or '공부' in msg_lower or 'it' in msg_lower) and \
               ('쪽' in msg_lower or '관련' in msg_lower or '추천' in msg_lower or '알아봐' in msg_lower):
                needs.append('교육')

            if ('주거' in msg_lower or '월세' in msg_lower or '전세' in msg_lower) and \
               ('쪽' in msg_lower or '관련' in msg_lower or '추천' in msg_lower or '알아봐' in msg_lower):
                needs.append('주거')

            if ('일자리' in msg_lower or '취업' in msg_lower or '구직' in msg_lower) and \
               ('쪽' in msg_lower or '관련' in msg_lower or '추천' in msg_lower or '알아봐' in msg_lower):
                needs.append('일자리')

            if ('생활' in msg_lower or '통장' in msg_lower) and \
               ('쪽' in msg_lower or '관련' in msg_lower or '추천' in msg_lower or '알아봐' in msg_lower):
                needs.append('생활')

            if ('창업' in msg_lower or '사업' in msg_lower) and \
               ('쪽' in msg_lower or '관련' in msg_lower or '추천' in msg_lower or '알아봐' in msg_lower):
                needs.append('창업')

            if needs:
                user_info['needs'] = needs
        
        return user_info
    
    except json.JSONDecodeError:
        return {}
    except Exception:
        return {}


# ======================
# 응답 생성
# ======================


SYSTEM_PROMPT = """당신은 서울시 복지 상담사 '나침반'입니다.

사용자에게 추가 정보는 복지를 추천해주기 전까지만 물어보고 그 이후론 자연스럽게 다른 거 궁금하면 도와주겠다고만 하세요.

## ⚠️ 모든 모드에서 FAQ 우선
- [관련 FAQ 정보]가 제공되면 **반드시 해당 내용을 있는 그대로** 답변하세요.
- FAQ 내용과 다른 정보를 지어내지 마세요. 반대로도 말하지 마세요. 그대로 말하세요.
- [관련 FAQ 정보]가 제공되면 **어떤 모드든** 해당 내용을 우선 참고하세요.
- FAQ에 정확한 답변이 있으면 그 내용을 기반으로 답변하세요.
- [관련 FAQ 정보]가 제공되면 **반드시 해당 내용을 그대로 인용하여** 답변하세요.
- FAQ에 있는 내용을 임의로 요약하거나 생략하지 마세요.
- FAQ가 없는 질문에만 일반적인 안내를 하세요.
- ⚠️ 다시 한번 강조: 이 모드에서는 절대 질문하지 마세요
  - 현재 파악된 정보로만 판단
  - 부족한 정보는 "확인 필요" 표시만

## ⚠️ 중요: 서울시 전용 서비스
- 이 서비스는 **서울시 복지 전용** 챗봇입니다.
- 서울 외 지역(부산, 인천, 대구, 경기도 등) 복지는 **절대 추천하지 마세요**.
- 부산형 긴급복지, 인천 청년지원 등 다른 지역 복지 프로그램을 언급하지 마세요.
- 서울 외 지역 사용자에게는 복지로(bokjiro.go.kr) 안내만 하세요.

절대 "해당되는 복지 프로그램이 없습니다", "매칭되는 프로그램이 없습니다"라는 표현을 사용하지 마세요.
"현재 정보로는 정확하지 않지만, 이런 프로그램들이 가능해 보여요. 더 정확한 추천을 위해 추가 정보 알려주세요!"

프로그램 후보를 제시할 때는:
1. 프로그램 이름 (굵게)
2. 왜 추천 가능한지 (조건 일부 일치 설명)
3. 더 정확한 판단을 위해 어떤 정보가 필요한지 질문

이 흐름을 항상 유지하세요.

## 당신의 역할
- 사용자의 상황에 공감해주고,
- "지금 조건으로 왜 복지 혜택 가능성이 있는지"를 설명해주고,
- 아래에 표시될 **복지 카드**를 자연스럽게 보도록 유도하는 역할입니다.
- 복지 카드(혜택/대상/신청방법 요약)는 **파이썬 코드에서 따로 렌더링**되므로,
  당신이 직접 "📋 맞춤 복지 카드" 섹션이나 "자세히보기" 버튼 텍스트를 만들 필요는 없습니다.

---

## 답변 모드

당신은 다중 모드로 동작합니다.

### 0) [MATCH_CARD_MODE] (카드가 이미 표시되는 매칭 모드)
- 새로운 질문을 던지지 마세요.
- 이 모드는 `[MATCH_CARD_MODE]` 태그가 붙었을 때만 사용됩니다.
- 이미 사용자의 정보가 충분히 모여 있고, 화면 아래에 **맞춤 복지 카드**가 표시될 예정입니다.
- 이미 복지 카드가 한 번이라도 노출된 상태라면
  (즉, [카드 노출 여부]가 "예"인 경우)
  → 사용자의 정보를 묻는 질문을 하지 않는다.

따라서:

- ❌ 새로운 질문을 던지지 마세요.
  - 특히 "월세는 본인 명의인가요?", "건강보험은 어떻게 가입되어 있나요?" 같은
    추가 정보 수집 질문을 다시 하지 마세요.
- ✅ 대신 아래만 해 주세요.
  - 지금 조건에서 왜 이런 복지들이 추천되었는지 간단히 설명
  - "아래 카드에서 자세한 내용 확인해 보세요."라고 안내
  - "궁금한 복지가 있으면 '○○ 자세히 알려줘'라고 말씀해 주세요."처럼
    다음 행동만 제안

---

### 1) [MATCH_MODE] (= 대부분의 초기 턴)

- 조건:
  - 기본 모드
  - 아직 카드가 나오지 않았거나, 정보 수집이 더 필요한 상태
  

- 해야 할 일:
  1. 공감
     - 예) "취준 중에 월세까지 부담하시면 정말 빠듯하실 것 같아요 😢"
  2. 부족한 정보 2~3개 질문 (나이 / 거주지 / 고용 상태 / 주거 형태 / 소득)
  3. 너무 구체적인 제도 이름 남발은 피하고, 먼저 "어떤 방향의 복지"가 가능한지 정도만 설명
  4. 정보가 충분해지면, 이후 턴에서 카드가 표시될 수 있음을 자연스럽게 안내
  
- 주의:매칭모드에서만 상대방의 정보에 대한 추가 질문을 하세요.

---

### 2) [DETAIL_MODE] (= 사용자가 특정 복지에 대해 "자세히 알려줘"라고 할 때)

- 조건:
  - 사용자 메시지에 `[DETAIL_MODE]`가 붙어 있으면 이 모드로 동작합니다.

- 해야 할 일:
  - 사용자가 묻는 **특정 복지 프로그램 1개**에 대해 아래 내용을 사람 말처럼 설명합니다:
    - 어떤 사람을 위한 제도인지 (대상, 연령, 소득, 거주지 등)
    - 어떤 혜택을 주는지 (금액, 횟수, 기간 등)
    - 신청 시 유의사항 & 조건 (예: 중위소득 %, 재직/구직 여부, 1회만 가능 등)
    - 대략적인 신청 방법 흐름

- 이 모드에서의 규칙:
  - "카드에서 확인하세요." 라고 떠넘기지 말고, 핵심 내용을 직접 설명하세요.
  - 마지막에는 항상 다음 행동을 제안하세요.
  - 사용자의 정보는 물어보지 마세요.

---

### 3) [ELIGIBILITY_MODE] (= 사용자가 "나 신청 가능해?" 등 물을 때)

- 조건:
  - 사용자 메시지 앞부분에 `[ELIGIBILITY_MODE]` 태그가 붙어 있으면 이 모드로 동작합니다.

- ⚠️ 최우선 규칙: 
- 절대 추가 질문 금지
  - "월세는 본인 명의인가요?" ❌
  - "월세는 얼마인가요?" ❌
  - "보증금은 얼마인가요?" ❌
  - 이런 질문들을 절대 하지 마세요.

- 해야 할 일:
  1. **결론 먼저 말하기** (현재 정보 기준)
     - "현재 정보로 판단하면 신청 가능해 보입니다" 또는
     - "현재 정보로는 일부 조건이 불확실하지만, 기본 조건은 충족됩니다"
  
  2. **조건 비교** (✅ 충족 / ⚠️ 확인 필요 형태로)
     예시:
     - ✅ 나이: 27세 (19-39세 대상 충족)
     - ✅ 거주지: 서울 (서울시 거주자 대상 충족)
     - ⚠️ 월세 금액: 70만원 이하 기준 (현재 정보 없음)
  
  3. **추가 확인 필요한 조건 안내** (질문 형태 ❌, 나열만 ⭕)
     - "정확한 판단을 위해서는 다음 사항을 확인하세요"
     - 리스트로만 나열
  
  4. **다음 행동 제안**
     - "자세한 신청 방법은 '어떻게 신청해?'라고 물어보세요"

- ⚠️ 다시 한번 강조: 이 모드에서는 절대 질문하지 마세요
  - 현재 파악된 정보로만 판단
  - 부족한 정보는 "확인 필요" 표시만

### 4) [APPLY_MODE] (= 사용자가 "신청 방법 알려줘" 등 물을 때)

- 조건:
  - 사용자 메시지 앞에 `[APPLY_MODE]` 태그가 붙어 있으면 이 모드로 동작합니다.

- 해야 할 일:
  1. 준비 서류
  2. 신청 경로
  3. 신청 절차
  4. 처리 기간
  5. 주의사항
  -주의: 다음 행동 제안만 하고 추가 정보 수집 질문은 하지 마세요.

---

### 5) [FAQ_MODE]

- 조건: [FAQ_MODE] 태그
- 해야 할 일:
  1. [관련 FAQ 정보]에서 답변을 그대로 인용
  2. FAQ 내용을 요약하거나 생략하지 않기
- 규칙:
  - 복지 카드를 표시하지 마세요
  - FAQ 답변만 깔끔하게 전달
  -주의: 다음 행동 제안만 하고 추가 정보 수집 질문은 하지 마세요.

---

## 🚨 필수 정보 수집 (복지 추천 전 반드시!) — [MATCH_MODE]에서만 적용

※ 이 규칙은 **[MATCH_MODE]일 때만** 강하게 적용됩니다.

### 수집해야 할 정보:
1. 나이
2. 거주지
3. 고용 상태
4. 주거 형태
5. 소득 수준
"""


def generate_response(
    user_message: str,
    user_info: dict,
    matched_programs: pd.DataFrame,
    conversation_history: list,
    intent: str = "match",
    is_other_request: bool = False,
    already_programs: list | None = None,
    must_ask_couple_income: bool = False,
) -> str:
    """GPT를 사용해 친근한 응답 생성"""

    # 🆕 0. DETAIL_MODE 전용: CSV/DF 기반으로만 상세 안내
    if intent == "detail" and matched_programs is not None and not matched_programs.empty:
        row = matched_programs.iloc[0]

        name = (row.get("program_name") or "").strip()
        cat1 = (row.get("category_primary") or "").strip()
        cat2 = (row.get("category_secondary") or "").strip()
        desc = (row.get("description") or "").strip()
        url = (row.get("url_pdf") or row.get("url") or "").strip()

        age_min = row.get("age_min")
        age_max = row.get("age_max")
        residence = row.get("residence_required") or row.get("residence")
        emp = row.get("employment_status")
        special = row.get("special_conditions")

        support_type = row.get("support_type")
        amount = row.get("support_amount")
        duration = row.get("support_duration")
        details = row.get("support_details")

        apply_period = row.get("application_period")
        apply_how = row.get("how_to_apply")
        contact = row.get("contact")

        lines: list[str] = []
        title = name or "해당 복지 프로그램"
        lines.append(f"**{title}**에 대해 자세히 안내드릴게요.\n")

        # 카테고리
        if cat1 or cat2:
            cat_text = " / ".join([c for c in [cat1, cat2] if c])
            lines.append(f"- 카테고리: {cat_text}")

        # 개요
        if desc:
            lines.append(f"- 개요: {desc}")

        # 지원 대상
        target_lines = []
        if age_min or age_max:
            if pd.notna(age_min) and pd.notna(age_max):
                target_lines.append(f"나이: 만 {int(age_min)}~{int(age_max)}세")
            elif pd.notna(age_min):
                target_lines.append(f"나이: 만 {int(age_min)}세 이상")
            elif pd.notna(age_max):
                target_lines.append(f"나이: 만 {int(age_max)}세 이하")

        if residence and isinstance(residence, str):
            target_lines.append(f"거주: {residence}")

        if emp and isinstance(emp, str):
            target_lines.append(f"고용 상태: {emp}")

        if special:
            # 리스트/문자열 모두 처리
            if isinstance(special, (list, tuple)):
                target_lines.append("특이사항: " + ", ".join(map(str, special)))
            else:
                target_lines.append(f"특이사항: {special}")

        if target_lines:
            lines.append("\n**✅ 지원 대상**")
            for t in target_lines:
                lines.append(f"- {t}")

        # 지원 내용
        support_lines = []
        if support_type:
            support_lines.append(f"지원 유형: {support_type}")
        if amount not in [None, "", 0, "0"]:
            support_lines.append(f"지원 금액: {amount}")
        if duration:
            support_lines.append(f"지원 기간: {duration}")
        if details:
            support_lines.append(f"세부 내용: {details}")

        if support_lines:
            lines.append("\n**💡 지원 내용**")
            for s in support_lines:
                lines.append(f"- {s}")

        # 신청 안내
        apply_lines = []
        if apply_period:
            apply_lines.append(f"신청/모집 기간: {apply_period}")
        if apply_how:
            apply_lines.append(f"신청 방법: {apply_how}")
        if contact:
            apply_lines.append(f"문의: {contact}")
        if url:
            apply_lines.append(f"자세히 보기: {url}")

        if apply_lines:
            lines.append("\n**📝 신청 안내**")
            for a in apply_lines:
                lines.append(f"- {a}")

        lines.append(
            "\n추가로 `신청 가능한지`, `다른 비슷한 교육 프로그램`도 궁금하시면 편하게 물어보세요. 😊"
        )

        return "\n".join(lines)

    # === 여기서부터는 기존 MATCH/FAQ/ELIGIBILITY/APPLY 모드 로직 ===

    # 1. 매칭된 프로그램 정보 정리
    programs_text = ""
    if matched_programs is not None and not matched_programs.empty:
        # ✅ 카테고리별로 그룹핑
        category_order = ["교육", "주거", "일자리", "생활·금융", "창업", "마음건강", "기타"]

        for cat in category_order:
            cat_programs = matched_programs[matched_programs["category_primary"] == cat]
            for _, row in cat_programs.head(2).iterrows():  # 카테고리당 최대 2개
                difficulty = row.get("difficulty_level", 3)
                if pd.notna(difficulty):
                    try:
                        difficulty_stars = "⭐" * int(difficulty)
                    except Exception:
                        difficulty_stars = "보통"
                else:
                    difficulty_stars = "보통"

                programs_text += f"""
- **{row['program_name']}** ({row.get('category_primary', '기타')})
  - 지원내용: {row.get('support_amount', '상세 내용 확인 필요')}
  - 신청방법: {row.get('how_to_apply', '홈페이지 확인')}
  - 난이도: {difficulty_stars}
"""

    # 2. FAQ 검색  👉 intent == "faq"일 때만 사용
    faq_context = ""
    if intent == "faq":
        try:
            current_program = ""
            if matched_programs is not None and not matched_programs.empty:
                current_program = matched_programs.iloc[0]["program_name"]

            enhanced_query = f"{current_program} {user_message}" if current_program else user_message
            faq_results = search_faq(enhanced_query, top_k=3)
            faq_context = format_faq_context(faq_results)
        except Exception:
            pass

    # 3. 카드 매칭 여부
    has_matches = matched_programs is not None and not matched_programs.empty

    # 4. 핵심 정보 개수
    info_count = sum(
        [
            1 if user_info.get("age") else 0,
            1 if user_info.get("residence") else 0,
            1 if user_info.get("employment_status") else 0,
            1 if user_info.get("housing_type") else 0,
            1 if user_info.get("income") is not None else 0,
        ]
    )

    # 5. 모드 태그
    if intent == "faq":
        mode_tag = "[FAQ_MODE]"
    elif intent == "apply":
        mode_tag = "[APPLY_MODE]"
    elif intent == "detail":
        mode_tag = "[DETAIL_MODE]"
    elif intent == "eligibility":
        mode_tag = "[ELIGIBILITY_MODE]"
    else:
        mode_tag = "[MATCH_CARD_MODE]" if has_matches else "[MATCH_MODE]"

    # 6. 신혼부부 소득 질문
    couple_income_instruction = ""
    if must_ask_couple_income:
        couple_income_instruction = """
[중요: 신혼부부 소득 관련 필수 질문]

이 소득이 개인 기준인지, 부부 합산 기준인지가 불명확합니다.
이 질문을 꼭 포함하세요:
"지금 말씀해 주신 소득은 ○○님 개인 소득인지, 아니면 배우자분 소득까지 포함한 **부부 합산 소득**인지 알려주실 수 있을까요?
신혼부부 지원은 대부분 부부 합산 소득을 기준으로 해서 이 부분이 중요합니다."
"""

    # 7. 카드 매칭 여부에 따른 추가 안내
    if has_matches and intent == "match":
        extra_instruction = """
(⚠️ 이미 복지 카드가 제공되었으므로,
추가 질문을 하지 말고, 보여준 카드 중심으로 안내만 이어가세요.)
"""
    else:
        extra_instruction = """
(ℹ️ 아직 복지 카드가 제공되지 않았거나, 정보 수집 단계입니다.
지원 가능성 판단을 위해 2~3가지 질문을 자연스럽게 던지세요.)
"""

    # 8. 정보 부족 시 제동
    if intent == "match" and info_count < 3:
        detail_level_instruction = """
지금은 핵심 정보(나이, 거주지, 고용 상태, 주거 형태, 소득) 중에서 3개 미만만 파악된 상태입니다.

- 이 단계에서는 **구체적인 복지 이름을 언급하지 마세요.**
- 대신, "주거비를 도와주는 청년 지원", "자립을 돕는 청년 복지"처럼
  아주 포괄적인 방향만 이야기해 주세요.
- 부족한 정보를 2~3가지 질문 형태로 자연스럽게 물어보세요.
"""
    else:
        detail_level_instruction = """
핵심 정보가 어느 정도 수집된 상태이므로,
상황에 맞는 복지 프로그램 이름을 1~3개까지 구체적으로 언급해도 좋습니다.
"""

    # 9. user prompt
    user_prompt = f"""{mode_tag}
{extra_instruction}
{detail_level_instruction}
{couple_income_instruction}

사용자 메시지: {user_message}
추출된 사용자 정보: {json.dumps(user_info, ensure_ascii=False)}
매칭된 복지 프로그램:
{programs_text if programs_text else "추천을 드리기 위해서는 추가 정보가 필요합니다."}

{faq_context}

위 정보를 바탕으로, 현재 모드에 맞게 답변하세요.
"""

    # 10. OpenAI 호출
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in conversation_history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_prompt})

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=RESPONSE_TEMPERATURE,
            max_tokens=MAX_TOKENS_RESPONSE,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"죄송해요, 응답 생성 중 오류가 발생했어요: {e}"