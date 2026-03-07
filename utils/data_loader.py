"""
복지 데이터 로드 유틸리티
"""

import pandas as pd
import streamlit as st
from config import DATA_PATH


@st.cache_data
def load_welfare_data() -> pd.DataFrame:
    """통합된 welfare_data.csv 파일 로드"""
    try:
        df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
        # 컬럼명 공백/특수문자 제거
        df.columns = df.columns.str.strip()
        # 빈 이름 컬럼(trailing comma) 제거
        df = df.loc[:, df.columns != '']
        df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

        # 필수 컬럼 리스트
        required_cols = [
            'id', 'program_name', 'category_primary', 'category_secondary', 'description',
            'age_min', 'age_max', 'income_type', 'income_max',
            'residence_required', 'employment_status', 'special_conditions',
            'support_type', 'support_amount', 'support_duration',
            'how_to_apply', 'contact', 'difficulty_level', 'source'
        ]

        # 없는 컬럼 자동 생성 (값은 None)
        for col in required_cols:
            if col not in df.columns:
                df[col] = None

        # 숫자 컬럼 변환 (강제 숫자화, NaN 허용)
        df['age_min'] = pd.to_numeric(df['age_min'], errors='coerce')
        df['age_max'] = pd.to_numeric(df['age_max'], errors='coerce')
        df['income_max'] = pd.to_numeric(df['income_max'], errors='coerce')

        return df

    except Exception as e:
        st.error(f"복지 데이터 로드 실패: {e}")
        return pd.DataFrame()
