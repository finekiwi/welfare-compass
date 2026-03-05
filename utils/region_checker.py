"""
지역 판별 유틸리티
"""

import re
from config import SEOUL_DISTRICTS_PATTERN, OTHER_REGIONS_PATTERN


def is_seoul_region(residence: str | None) -> bool:
    """서울 지역인지 판별"""
    if not residence:
        return True  # 기본값은 서울로 가정
    return bool(re.search(SEOUL_DISTRICTS_PATTERN, residence, re.IGNORECASE))


def is_other_region(residence: str | None) -> bool:
    """서울 외 지역인지 판별"""
    if not residence:
        return False
    return bool(re.search(OTHER_REGIONS_PATTERN, residence, re.IGNORECASE))


def normalize_residence(residence: str | None) -> str | None:
    """
    거주지 정규화
    - 서울 구 이름만 있으면 "서울 {구이름}" 형태로 변환
    - 다른 지역이면 "서울" 제거
    """
    if not residence:
        return None
    
    # 다른 지역이면 "서울" 제거
    if is_other_region(residence):
        clean_residence = re.sub(r'서울\s*', '', residence).strip()
        return clean_residence if clean_residence else residence
    
    # 서울 지역인데 "서울"이 없으면 추가
    if is_seoul_region(residence) and '서울' not in residence:
        return f"서울 {residence}"
    
    return residence
