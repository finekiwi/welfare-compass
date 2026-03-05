"""
사이드바 UI 컴포넌트
"""

import streamlit as st


def render_sidebar() -> None:
    """사이드바 렌더링"""
    with st.sidebar:
        # 디버깅 정보 표시
        if st.session_state.get("debug_info"):
            st.write("🔍 DEBUG:", st.session_state.debug_info)
        
        st.header("📋 파악된 정보")
        #st.write("🔍 전체 정보:", st.session_state.get("user_info", {}))
        
        info = st.session_state.get('user_info', {})
        if info:
            if info.get('age'):
                st.write(f"👤 나이: {info['age']}세")
            if info.get('income'):
                income_type = info.get('income_type', '월')
                st.write(f"💰 소득: {income_type} {info['income']}만원")
            if info.get('residence'):
                st.write(f"📍 거주지: {info['residence']}")
            if info.get('employment_status'):
                st.write(f"💼 고용상태: {info['employment_status']}")
            if info.get('housing_type'):
                st.write(f"🏠 주거형태: {info['housing_type']}")
            if info.get('special_conditions'):
                st.write(f"⭐ 특수조건: {', '.join(info['special_conditions'])}")
            
            if info.get("median_percent"):
                st.write(f"📊 중위소득 대비: 약 {info['median_percent']}% ({info['median_bracket']})")
            elif info.get("median_bracket"):
                st.write(f"📊 중위소득 구간: {info['median_bracket']}")
        else:
            st.write("아직 파악된 정보가 없어요")
        
        st.divider()
        
        if st.button("🔄 대화 초기화"):
            # 세션 전체 삭제
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
