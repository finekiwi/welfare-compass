"""
복지나침반 🧭 - 서울시 AI 복지 매칭 서비스
메인 Streamlit 앱
"""

import streamlit as st
import pandas as pd
import re

from config import LOGO_PATH, COUPLE_INCOME_KEYWORDS, OTHER_REQUEST_KEYWORDS
from utils import (
    detect_intent,
    estimate_median_percent_2025,
    normalize_residence,
    load_welfare_data,
)
from services import extract_user_info, generate_response, match_welfare_programs
from ui import render_welfare_card, render_sidebar, inject_custom_css
from utils.user_needs import extract_requested_category

# ======================
# 페이지 설정
# ======================

st.set_page_config(
    page_title="복지나침반 🧭",
    page_icon="🧭",
    layout="centered"
)

inject_custom_css()


# ======================
# 초기화 함수
# ======================

def init_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.user_info = {}
        st.session_state.last_matched = pd.DataFrame()
        st.session_state.last_intent = "match"
        
        # 초기 인사 메시지
        welcome_msg = """안녕하세요! 저는 복지나침반이에요 🧭

서울시에서 받을 수 있는 복지 혜택을 찾아드릴게요.
복잡한 조건? 걱정 마세요. 대화만 하면 제가 알아서 찾아드려요!

**간단히 상황을 말씀해주세요.** 예를 들면:
- "27살이고 월세 살고 있어요"
- "취준생인데 지원받을 수 있는 게 있을까요?"
- "소득이 적어서 생활이 어려워요"

어떤 상황이신가요? 😊"""
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    
    if "user_info" not in st.session_state:
        st.session_state.user_info = {}
    if "last_matched" not in st.session_state:
        st.session_state.last_matched = pd.DataFrame()
    if "last_match_index" not in st.session_state:
        st.session_state.last_match_index = None


def render_user_message(content: str):
    """사용자 메시지 렌더링 (커스텀 HTML)"""
    st.markdown(f"""
<div class="user-msg-row">
    <div class="user-bubble">{content}</div>
    <div class="user-avatar">나</div>
</div>
    """, unsafe_allow_html=True)


def render_chat_history(df: pd.DataFrame):
    """대화 히스토리 렌더링"""
    for idx, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            render_user_message(message["content"])
        else:
            with st.chat_message("assistant", avatar=LOGO_PATH):
                st.markdown(message["content"])
                
                if message.get("show_card") and message.get("matched_programs") is not None:
                    matched_df = message["matched_programs"]

                    if isinstance(matched_df, pd.DataFrame) and not matched_df.empty:
                        st.markdown("---")
                        st.markdown("### 📋 맞춤 복지 카드")

                        # ✅ GPT / 매칭 점수 순서 그대로 상위 3개 사용
                        top_programs = matched_df.head(3)

                        first_rendered_program = None
                        for i, (_, row) in enumerate(top_programs.iterrows()):
                            if i == 0:
                                first_rendered_program = row.get("program_name", "")
                            render_welfare_card(row)

                        if first_rendered_program:
                            st.markdown(
                                "---\n\n"
                                f"💬 **궁금한 복지가 있으시면** `'{first_rendered_program} 자세히 알려줘'` 라고 말씀해주세요!\n\n"
                                "📝 신청 방법이나 필요 서류도 안내해드릴 수 있어요.\n\n"
                                "🔍 다른 상황(가족, 건강보험, 부채 등)이 있으시면 추가 복지도 찾아드릴게요!"
                            )


def process_user_input(prompt: str, df: pd.DataFrame):
    """사용자 입력 처리"""
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_user_message(prompt)
    
    # '다른 복지' follow-up 여부 감지
    is_other_request = any(k in prompt for k in OTHER_REQUEST_KEYWORDS)
    st.session_state.is_other_request = is_other_request
    
    # intent 계산
    last_intent = st.session_state.get("last_intent", "match")
    intent = detect_intent(prompt, last_intent)
    
    # 이미 추천했던 프로그램들 이름 리스트
    prev_matched = st.session_state.get("last_matched", pd.DataFrame())
    already_programs = []
    if prev_matched is not None and not prev_matched.empty:
        already_programs = (
            prev_matched.head(3)["program_name"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    
    # 처리 중 표시
    with st.chat_message("assistant", avatar=LOGO_PATH):
        with st.spinner("생각 중이에요... ⏳"):
            # 1. 사용자 정보 추출
            new_info = extract_user_info(prompt, st.session_state.messages)
            
            # 기존 정보와 병합 (덮어쓰기)
            for key, value in new_info.items():
                if value is not None and value != [] and value != "":
                    st.session_state.user_info[key] = value
            
            # ✅ 서울 지역 판별 로직
            user_info = st.session_state.user_info
            residence = user_info.get("residence")
            is_seoul = user_info.get("is_seoul_resident")
            
            # 1️⃣ LLM이 is_seoul_resident를 명시적으로 줬다면 최우선
            if is_seoul is True:
                if residence and "서울" not in residence:
                    user_info["residence"] = f"서울 {residence.strip()}"
                user_info["residence"] = normalize_residence(user_info["residence"])
                
            elif is_seoul is False:
                # 서울이 아니라고 LLM이 판단
                response = """죄송해요, 저는 **서울시 복지 전용 챗봇**이라 서울시 복지 정보만 안내해드릴 수 있습니다.
다른 지역 복지 정보는 **[복지로(bokjiro.go.kr)](https://www.bokjiro.go.kr)**에서 확인해 주시면 좋겠습니다.
혹시 서울 거주 가족분이나, 서울로 이주 계획 관련 복지가 궁금하시면 말씀해 주세요."""
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "show_card": False,
                    "matched_programs": None,
                })
                st.rerun()
            else:
                # 2️⃣ is_seoul_resident가 null인 경우 정규식 보조 판별
                seoul_keywords = r'(서울|종로구|중구|용산구|성동구|광진구|동대문구|중랑구|성북구|강북구|도봉구|노원구|은평구|서대문구|마포구|양천구|강서구|구로구|금천구|영등포구|동작구|관악구|서초구|강남구|송파구|강동구)'
                other_regions = r'(부산|인천|대구|대전|광주|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)'
                if residence:
                    if re.search(other_regions, residence):
                        response = """죄송해요, 저는 **서울시 복지 전용 챗봇**이라 서울시 복지 정보만 안내해드릴 수 있습니다.
다른 지역 복지 정보는 **[복지로(bokjiro.go.kr)](https://www.bokjiro.go.kr)**를 이용해 주세요."""
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "show_card": False,
                            "matched_programs": None,
                        })
                        st.rerun()
                    elif re.search(seoul_keywords, residence):
                        user_info["is_seoul_resident"] = True
                        if "서울" not in residence:
                            user_info["residence"] = f"서울 {residence.strip()}"
                        user_info["residence"] = normalize_residence(user_info["residence"])

            # 중위소득 계산
            percent, bracket = estimate_median_percent_2025(
                income=user_info.get("income"),
                income_type=user_info.get("income_type"),
                household_size=user_info.get("household_size")
            )
            user_info["median_percent"] = percent
            user_info["median_bracket"] = bracket
            
            # 신혼부부 소득 확인 로직
            special = user_info.get("special_conditions", []) or []
            is_newlywed = any("신혼" in s for s in special)
            income = user_info.get("income")
            income_scope = user_info.get("income_scope")

            # 부부 합산 표현 감지
            if income_scope is None:
                if any(kw in prompt for kw in COUPLE_INCOME_KEYWORDS):
                    user_info["income_scope"] = "부부합산"
                    income_scope = "부부합산"

            must_ask_couple_income = False
            if is_newlywed and income is not None and not income_scope:
                must_ask_couple_income = True
                    
            # 2. 매칭 로직
            info_count = sum([
                1 if user_info.get('age') else 0,
                1 if user_info.get('residence') else 0,
                1 if user_info.get('employment_status') else 0,
                1 if user_info.get('housing_type') else 0,
                1 if user_info.get('income') is not None else 0,
            ])
            
            # 3. 매칭 결과 결정
            matched = None
            new_match = False

            # 매칭모드 + 정보 3개 이상일 때 매칭
            if intent == "match" and info_count >= 3:
                # ✅ 첫 매칭에서는 카테고리 필터링 안 함
                requested_category = None
                if already_programs:  # 이미 추천한 프로그램이 있을 때만
                    requested_category = extract_requested_category(prompt)

                matched = match_welfare_programs(
                    user_info,
                    df,
                    include_category=requested_category,
                    exclude_programs=already_programs
                )
                
                # ✅ 매칭 결과 저장
                if matched is not None and not matched.empty:
                    st.session_state["last_matched"] = matched
                else:
                    st.session_state["last_matched"] = pd.DataFrame()
                
                new_match = True

            else:
                # match가 아니면 이번 턴에는 추천 안 함
                matched = pd.DataFrame()

            # 4. 응답 생성
            response = generate_response(
                prompt,
                user_info,
                matched,
                st.session_state.messages,
                intent=intent,
                is_other_request=is_other_request,
                already_programs=already_programs,
                must_ask_couple_income=must_ask_couple_income,
            )
    
    # 메시지 저장
    show_card = bool(new_match and matched is not None and not matched.empty)
    card_programs = matched.copy() if show_card else None
        
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "show_card": show_card,
        "matched_programs": card_programs
    })
    
    # 매칭 모드였고 결과 있으면 인덱스 저장
    assistant_index = len(st.session_state.messages) - 1
    if intent == "match" and matched is not None and not matched.empty:
        st.session_state.last_match_index = assistant_index
    
    # Last intent 업데이트
    st.session_state.last_intent = intent
    
    st.rerun()


# ======================
# 메인 함수
# ======================

def main():
    # 로고 + 타이틀
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(LOGO_PATH, width=130)
    with col2:
        st.title("복지나침반")
        st.caption("서울시 AI 복지 매칭 서비스")
    
    # 데이터 로드
    df = load_welfare_data()
    
    if df.empty:
        st.error("복지 데이터를 불러올 수 없습니다.")
        return
    
    # 세션 상태 초기화
    init_session_state()
    
    # 대화 히스토리 표시
    render_chat_history(df)
    
    # 사용자 입력
    if prompt := st.chat_input("상황을 말씀해주세요..."):
        process_user_input(prompt, df)
    
    # 사이드바
    render_sidebar()


if __name__ == "__main__":
    main()