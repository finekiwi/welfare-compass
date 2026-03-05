"""
중위소득 계산 유틸리티
"""

from config import MEDIAN_INCOME_2025


def get_median_base_2025(household_size: int | None) -> float | None:
    """
    가구원 수별 2025년 기준중위소득 (월, 만원)
    8인 이상 가구는 7인가구 기준 + (7인-6인 차액 * 추가 인원 수)
    """
    if not household_size or household_size <= 0:
        return None

    if household_size <= 7:
        return MEDIAN_INCOME_2025.get(household_size)

    # 8인 이상
    diff = MEDIAN_INCOME_2025[7] - MEDIAN_INCOME_2025[6]
    extra = household_size - 7
    return MEDIAN_INCOME_2025[7] + diff * extra


def estimate_median_percent_2025(
    income: float | None,
    income_type: str | None,
    household_size: int | None
) -> tuple[int | None, str | None]:
    """
    중위소득 대비 퍼센트 추정
    
    Args:
        income: 소득 (만원 단위)
        income_type: "월" 또는 "연"
        household_size: 가구원 수 (없으면 1로 가정)
    
    Returns:
        (대략적인 중위소득 %, 구간 라벨) 또는 (None, None)
    """
    if income is None:
        return None, None

    # 연봉이면 월 소득으로 변환
    monthly_income = income
    if income_type == "연":
        monthly_income = income / 12.0

    base = get_median_base_2025(household_size or 1)
    if not base:
        return None, None

    percent = monthly_income / base * 100

    # 구간 라벨
    if percent <= 50:
        bracket = "중위소득 50% 이하 추정"
    elif percent <= 60:
        bracket = "중위소득 60% 이하 추정"
    elif percent <= 100:
        bracket = "중위소득 100% 이하 추정"
    else:
        bracket = "중위소득 100% 초과 추정"

    return round(percent), bracket
