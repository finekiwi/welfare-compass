"""
복지 프로그램 매칭 서비스
"""

import re
import pandas as pd


def match_welfare_programs(
    user_info: dict,
    df: pd.DataFrame,
    include_category=None,
    exclude_programs=None
) -> pd.DataFrame:
    """사용자 정보에 맞는 복지 프로그램 매칭 - 다양한 카테고리에서 추천

    Args:
        user_info: 사용자 정보 딕셔너리
        df: 복지 프로그램 데이터프레임
        include_category: 특정 카테고리만 포함 (예: '주거', '일자리')
        exclude_programs: 제외할 프로그램 이름 리스트
    """
    if df.empty or 'program_name' not in df.columns:
        return pd.DataFrame()

    matched = df.copy()

    # 이전에 추천한 프로그램 제외
    if exclude_programs:
        matched = matched[~matched['program_name'].isin(exclude_programs)]

    # 특정 카테고리만 필터링
    if include_category:
        matched = matched[
            matched['category_primary'].str.contains(include_category, case=False, na=False) |
            matched['category_secondary'].str.contains(include_category, case=False, na=False)
        ]

    # 1. 나이 필터링
    if user_info.get('age'):
        age = user_info['age']
        mask = (
            (matched['age_min'].isna() | (matched['age_min'] <= age)) &
            (matched['age_max'].isna() | (matched['age_max'] >= age))
        )
        matched = matched[mask]

    # 2. 특수조건 필터링 (신혼부부, 한부모 등은 해당자만)
    matched = matched[matched['special_conditions'].apply(
        lambda x: _check_special_conditions(x, user_info)
    )]

    # 프로그램 이름으로도 이중 체크
    user_special = [s.lower() for s in user_info.get('special_conditions', [])]
    is_newlywed = any('신혼' in s for s in user_special)

    if not is_newlywed:
        matched = matched[
            ~matched['program_name'].str.contains('신혼|혼인', case=False, na=False)
        ]

    # 3. 거주지 필터링
    if user_info.get('residence'):
        residence = user_info.get('residence', '')
        seoul_keywords = r'(서울|종로|중구|용산|성동|광진|동대문|중랑|성북|강북|도봉|노원|은평|서대문|마포|양천|강서|구로|금천|영등포|동작|관악|서초|강남|송파|강동|왕십리|신촌|홍대|성수|잠실)'
        is_seoul = bool(re.search(seoul_keywords, residence, re.IGNORECASE))

        if residence and not is_seoul:
            matched = matched[
                matched['residence_required'].isna() |
                ~matched['residence_required'].str.contains('서울', na=False)
            ]

    # 4. 고용상태 필터링
    if user_info.get('employment_status'):
        emp_status = user_info['employment_status']
        matched = matched[matched['employment_status'].apply(
            lambda x: _check_employment(x, emp_status)
        )]

    # 5. 관련 카테고리 도출
    relevant_categories = _get_relevant_categories(user_info)

    # 6. 우선순위 점수 계산
    matched['priority'] = matched.apply(
        lambda row: _calc_priority(row, user_info, relevant_categories),
        axis=1
    )

    # 점수가 30점 미만인 프로그램 제외
    matched = matched[matched['priority'] >= 30]

    # 7. 정렬
    if 'difficulty_level' in matched.columns:
        matched = matched.sort_values(['priority', 'difficulty_level'], ascending=[False, True])
    else:
        matched = matched.sort_values('priority', ascending=False)

    # 8. 카테고리별로 골고루 선택
    final_results = []
    categories_selected = {}

    for _, row in matched.iterrows():
        cat = row.get('category_primary', '기타')

        if include_category:
            # 사용자가 특정 카테고리 요청 시 해당 카테고리는 제한 없음
            if include_category.lower() in cat.lower():
                final_results.append(row)
                categories_selected[cat] = categories_selected.get(cat, 0) + 1
            else:
                if categories_selected.get(cat, 0) < 1:
                    final_results.append(row)
                    categories_selected[cat] = categories_selected.get(cat, 0) + 1
        else:
            max_per_category = 2
            if cat == '주거':
                max_per_category = 2
            elif cat in ['일자리', '교육', '생활·금융']:
                max_per_category = 2

            if categories_selected.get(cat, 0) < max_per_category:
                final_results.append(row)
                categories_selected[cat] = categories_selected.get(cat, 0) + 1

        if len(final_results) >= 10:
            break

    if final_results:
        return pd.DataFrame(final_results)
    return pd.DataFrame()


def _check_special_conditions(row_conditions, user_info: dict) -> bool:
    """특수조건 체크"""
    if pd.isna(row_conditions) or row_conditions == '' or row_conditions == '없음':
        return True

    row_conds = str(row_conditions).lower()
    user_special = [s.lower() for s in user_info.get('special_conditions', [])]

    # "제외 대상" 표현이 있으면 통과
    if '제외' in row_conds and '대상' in row_conds:
        return True

    # 기초생활수급자 복지는 수급자만
    if '기초생활' in row_conds or '수급자' in row_conds:
        if not any('기초생활' in s or '수급자' in s for s in user_special):
            return False

    # 차상위 복지는 차상위만
    if '차상위' in row_conds:
        if not any('차상위' in s for s in user_special):
            return False

    # 중증장애인 복지는 중증장애인만
    if '중증장애' in row_conds:
        if not any('중증장애' in s or '장애' in s for s in user_special):
            return False

    # 자녀 관련 복지는 자녀 있는 사람만
    children_keywords = ['자녀', '초등', '중등', '고등', '미취학', '유치원']
    if any(keyword in row_conds for keyword in children_keywords):
        if not user_info.get('has_children', False):
            return False

    # 신혼부부 복지는 신혼부부만
    if '신혼' in row_conds and '제외' not in row_conds:
        if not any('신혼' in s for s in user_special):
            return False

    # 한부모 복지는 한부모만
    if '한부모' in row_conds and '제외' not in row_conds:
        if not any('한부모' in s for s in user_special):
            return False

    # 장애인 복지는 장애인만
    if '장애' in row_conds and '제외' not in row_conds:
        if not any('장애' in s for s in user_special):
            return False

    # 다자녀 복지는 다자녀만
    if '다자녀' in row_conds and '제외' not in row_conds:
        if not any('다자녀' in s for s in user_special):
            return False

    return True


def _check_employment(row_status, emp_status: str) -> bool:
    """고용상태 체크"""
    if pd.isna(row_status) or row_status == '제한없음':
        return True
    if emp_status == '구직중' and '구직중' in str(row_status):
        return True
    if emp_status == '재직' and ('재직' in str(row_status) or '근로' in str(row_status)):
        return True
    if emp_status == '학생' and '학생' in str(row_status):
        return True
    return True


def _get_relevant_categories(user_info: dict) -> list:
    """사용자 맥락에서 관련 카테고리 도출"""
    relevant_categories = []

    # 주거 맥락
    housing = user_info.get('housing_type', '').strip()
    if housing:
        relevant_categories.append('주거')
        if housing == '전세':
            relevant_categories.append('전세')
        elif housing == '월세':
            relevant_categories.append('월세')
        elif housing == '고시원':
            relevant_categories.append('고시원')

    # 취업 맥락
    emp = user_info.get('employment_status', '')
    if emp in ['구직중', '무직']:
        relevant_categories.append('일자리')

    # 소득 맥락
    income = user_info.get('income')
    if income is not None and income < 300:
        relevant_categories.append('생활')
        relevant_categories.append('금융')

    # 특수조건 맥락
    special = user_info.get('special_conditions', [])
    if '한부모' in special or '장애인' in special:
        relevant_categories.append('생활')

    # 교육 맥락 (자녀 있을 때만)
    has_children = user_info.get('has_children', False)
    children_ages = user_info.get('children_ages', [])
    if has_children or children_ages:
        relevant_categories.append('교육')

    # 필요 분야 직접 추가
    needs = user_info.get('needs', [])
    for need in needs:
        if need not in relevant_categories:
            relevant_categories.append(need)

    # 기본: 청년이면 일자리/주거 기본 추천
    if not relevant_categories and user_info.get('age'):
        age = user_info['age']
        if 19 <= age <= 39:
            relevant_categories = ['주거', '일자리', '생활']

    return relevant_categories


def _calc_priority(row, user_info: dict, relevant_categories: list) -> int:
    """우선순위 점수 계산"""
    score = 0
    category = str(row.get('category_primary', '')).lower()
    description = str(row.get('description', '')).lower()
    program_name = str(row.get('program_name', '')).lower()
    support_amount = str(row.get('support_amount', '')).lower()
    row_special = str(row.get('special_conditions', '')).lower()

    subcat = str(row.get('category_secondary', '')).strip()
    housing = user_info.get('housing_type', '').strip()

    # 사용자 특수조건
    user_special = [s.lower() for s in user_info.get('special_conditions', [])]
    is_newlywed = any('신혼' in s for s in user_special)

    # 1. 청년 특화 복지
    if '청년' in program_name:
        score += 10 if is_newlywed else 50

    # 2. 신혼부부 우선
    if is_newlywed:
        if '신혼' in program_name or '신혼' in description or '신혼' in row_special:
            score += 60
        if '청년' in program_name and '신혼' not in program_name and '신혼' not in description:
            score -= 10

    # 3. 실질적 금전 혜택 우선
    amounts = re.findall(r'(\d+)만원', support_amount)
    if amounts:
        max_amount = max([int(a) for a in amounts])
        if max_amount >= 100:
            score += 15
        elif max_amount >= 50:
            score += 15
        elif max_amount >= 10:
            score += 5

    # 4. 관련 카테고리 매칭
    for cat in relevant_categories:
        if cat in category:
            score += 20
        if cat in description or cat in program_name:
            score += 10

    # 5. 주거형태 세부 매칭
    if housing:
        if housing == '월세':
            if subcat == '월세':
                score += 40
            elif subcat == '전월세':
                score += 25
            elif subcat == '전세':
                score -= 50
            elif subcat in ['임대']:
                score += 10
        elif housing == '전세':
            if subcat == '전세':
                score += 40
            elif subcat == '전월세':
                score += 25
            elif subcat == '월세':
                score -= 50
            elif subcat in ['임대']:
                score += 10

    # 6. 고용상태 세부 매칭
    emp = user_info.get('employment_status', '')
    if emp in ['구직중', '무직', '구직']:
        if '취업' in program_name or '일자리' in program_name or '자립' in program_name or '교육' in program_name:
            score += 40
        if '청년통장' in program_name or '저축' in program_name:
            score += 20

    # 7. 핵심 키워드 보너스
    핵심_키워드 = ['자립', '통장', '지원금', '수당', '월세']
    for kw in 핵심_키워드:
        if kw in program_name:
            score += 10

    # 8. 이사비/중개보수 프로그램 패널티
    이사_키워드 = ['이사비', '이사', '중개보수', '중개수수료']
    if any(kw in program_name for kw in 이사_키워드):
        score -= 50

    # 9. 위기 긴급 지원 프로그램 패널티
    위기_키워드 = ['위기', '긴급', '희망온돌']
    if any(kw in program_name for kw in 위기_키워드):
        income = user_info.get('income', 0)
        needs = user_info.get('needs', [])
        is_urgent = any(kw in str(needs).lower() for kw in ['긴급', '위기', '급해'])

        if not is_urgent and income is not None and income > 0:
            score -= 80
        elif income == 0:
            score -= 60

    return score
