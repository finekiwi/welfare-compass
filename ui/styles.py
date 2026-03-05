"""
CSS 스타일 정의
"""

import streamlit as st


CUSTOM_CSS = """
<style>
    /* 어시스턴트 메시지 아바타 정렬 */
    [data-testid="stChatMessageAssistant"] {
        align-items: flex-start !important;
    }

    [data-testid="stChatMessageAssistant"] img {
        margin-top: -10px !important;
    }
    
    /* 기본 Streamlit 채팅 숨기기 */
    [data-testid="stChatMessage"] {
        background: transparent !important;
    }
    
    /* 사용자 메시지 컨테이너 */
    .user-msg-row {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 8px;
        margin: 16px 0;
    }
    
    /* 사용자 말풍선 */
    .user-bubble {
        background: #f0f0f0;
        color: #333;
        padding: 12px 16px;
        border-radius: 20px;
        max-width: 70%;
        font-size: 15px;
        line-height: 1.5;
    }
    
    /* 사용자 아바타 */
    .user-avatar {
        width: 36px;
        height: 36px;
        background: #ff6b6b;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 14px;
        font-weight: bold;
    }
    
    /* 복지 카드 스타일 */
    .welfare-card {
        background: #ffffff;
        border: 1px solid #e8e8e8;
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .welfare-card-badge {
        display: inline-block;
        background: #fff0f0;
        color: #e74c3c;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .welfare-card-title {
        font-size: 17px;
        font-weight: 600;
        color: #1a1a1a;
        margin: 8px 0 12px 0;
    }
    
    .welfare-card-content {
        color: #666;
        font-size: 14px;
        line-height: 1.7;
    }
    
    .welfare-card-content p {
        margin: 6px 0;
    }
    
    .welfare-card-button {
        display: inline-block;
        margin-top: 12px;
        padding: 8px 16px;
        border: 1px solid #ddd;
        border-radius: 6px;
        font-size: 13px;
        color: #333;
        background: #fff;
        cursor: pointer;
    }
</style>
"""


def inject_custom_css() -> None:
    """커스텀 CSS 주입"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
