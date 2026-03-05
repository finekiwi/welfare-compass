"""
FAQ RAG 모듈 (CSV 버전)
- faq.csv 로드 → Document 생성 → 임베딩 → FAISS 저장 → 유사도 검색
"""

from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# ===== 경로 설정 =====
BASE_DIR = Path(__file__).parent
FAQ_CSV_PATH = BASE_DIR / "data" / "faq.csv"   # csv 위치
FAISS_INDEX_PATH = BASE_DIR / "faiss_index"    # 인덱스 저장 폴더


# ===== 컬럼 매핑 유틸 =====
def _get_column_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """CSV 컬럼명을 표준화"""
    col_map = {}
    for col in df.columns:
        c = col.strip().lower()
        if "program" in c:
            col_map["program_name"] = col
        elif c == "q" or "question" in c:
            col_map["q"] = col
        elif c == "a" or "answer" in c:
            col_map["a"] = col
    
    missing = [k for k in ["program_name", "q", "a"] if k not in col_map]
    if missing:
        raise ValueError(
            f"FAQ CSV에 필요한 컬럼이 없습니다: {missing} / 실제 컬럼: {list(df.columns)}"
        )
    
    return col_map


# ===== CSV → Document 변환 =====
def load_faq_csv(path: Path = FAQ_CSV_PATH) -> List[Document]:
    """질문만 임베딩, 답변은 메타데이터에"""
    if not path.exists():
        print(f"❌ FAQ CSV 파일이 없습니다: {path}")
        return []

    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = [c.strip() for c in df.columns]
    col_map = _get_column_mapping(df)

    docs: List[Document] = []

    for idx, row in df.iterrows():
        program_name = str(row[col_map["program_name"]]).strip()
        question = str(row[col_map["q"]]).strip()
        answer = str(row[col_map["a"]]).strip()

        # ✅ 질문만 임베딩 대상
        content = f"{program_name}: {question}"

        metadata = {
            "source_file": path.name,
            "program_name": program_name,
            "row_index": int(idx),
            "question": question,
            "answer": answer,
        }

        docs.append(Document(page_content=content, metadata=metadata))

    print(f"📄 FAQ CSV에서 {len(docs)}개 QA 로드 완료")
    return docs


# ===== FAISS 인덱스 생성 & 로드 =====
def build_faiss_index(
    docs: List[Document],
    save_path: Path = FAISS_INDEX_PATH,
) -> FAISS:
    """FAISS 벡터 인덱스 생성 및 저장 (CSV 기반)"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("🔄 FAQ 임베딩 생성 중...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 로컬 저장
    save_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(save_path))
    print(f"💾 FAISS 인덱스 저장 완료: {save_path}")

    return vectorstore


def load_faiss_index(index_path: Path = FAISS_INDEX_PATH) -> Optional[FAISS]:
    """저장된 FAISS 인덱스 로드 (없으면 None)"""
    if not index_path.exists():
        print(f"❌ 인덱스 폴더가 없습니다: {index_path}")
        return None

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,  # 로컬 파일이니까 OK
    )
    print("✅ FAQ FAISS 인덱스 로드 완료")
    return vectorstore


# ===== 검색 & 컨텍스트 변환 =====
def search_faq(
    query: str,
    vectorstore: Optional[FAISS] = None,
    top_k: int = 3,
) -> List[Dict]:
    """
    CSV 기반 FAQ 검색 + (있으면) RAG 기반 검색.

    Returns:
        [
          {
            "question": str,
            "answer": str,
            "program_name": str,
            "source_file": str,
            "score": float,
          },
          ...
        ]
    """
    faq_results: List[Dict] = []

    # 1️⃣ 먼저 인덱스 없으면 로드 시도
    if vectorstore is None:
        vectorstore = load_faiss_index()

    # 2️⃣ RAG 기반 검색 (인덱스가 있을 때)
    if vectorstore is not None:
        results = vectorstore.similarity_search_with_score(query, k=top_k)

        print("\n🔍 [RAG 검색 결과]")
        for doc, score in results:
            sim = round(1 - score, 3)
            metadata = doc.metadata or {}

            q = str(metadata.get("question", "질문 정보 없음"))
            a = str(metadata.get("answer") or doc.page_content)
            program_name = str(metadata.get("program_name", ""))
            source_file = str(metadata.get("source_file", "unknown"))

            print(f"📄 Source: {source_file}")
            print(f"🏷 Program: {program_name}")
            print(f"❓ Q: {q[:80]}...")
            print(f"💬 A: {a[:100]}...")
            print(f"📈 Score: {sim}\n")

            faq_results.append(
                {
                    "question": q,
                    "answer": a,
                    "program_name": program_name,
                    "source_file": source_file,
                    "score": float(sim),
                }
            )

        return faq_results

    # 3️⃣ 인덱스 없으면 CSV 키워드 검색 fallback
    print("\n⚠️ FAISS 인덱스가 없어 CSV 기반 검색으로 fallback 합니다.")

    if not FAQ_CSV_PATH.exists():
        print(f"❌ FAQ CSV 파일도 없습니다: {FAQ_CSV_PATH}")
        return []

    df = pd.read_csv(FAQ_CSV_PATH, encoding='utf-8-sig')
    df.columns = [c.strip() for c in df.columns]

    try:
        col_map = _get_column_mapping(df)
    except ValueError as e:
        print(f"❌ {e}")
        return []

    # 질문 + 답변 둘 다 검색
    query_lower = str(query).lower()
    mask = (
        df[col_map["q"]].astype(str).str.lower().str.contains(query_lower, na=False) |
        df[col_map["a"]].astype(str).str.lower().str.contains(query_lower, na=False)
    )
    filtered = df[mask].head(top_k)

    print("\n🔎 [CSV 기반 검색 결과]")
    for _, row in filtered.iterrows():
        program_name = str(row[col_map["program_name"]]).strip()
        question = str(row[col_map["q"]]).strip()
        answer = str(row[col_map["a"]]).strip()

        print(f"🏷 Program: {program_name}")
        print(f"❓ Q: {question}")
        print(f"💬 A: {answer[:100]}...\n")

        faq_results.append(
            {
                "question": question,
                "answer": answer,
                "program_name": program_name,
                "source_file": FAQ_CSV_PATH.name,
                "score": 0.8,  # 임시 고정 점수
            }
        )

    return faq_results


def format_faq_context(
    faq_results: List[Dict],
    score_threshold: float = 0.3,
) -> str:
    """
    검색 결과를 프롬프트에 넣을 컨텍스트 문자열로 변환.
    """
    if not faq_results:
        return ""

    context_parts: List[str] = ["[관련 FAQ 정보]"]

    for r in faq_results:
        if not r:
            continue

        score = float(r.get("score", 0.0))
        if score < score_threshold:
            continue

        q = str(r.get("question", "질문 정보 없음"))
        a = str(r.get("answer", "답변 정보 없음"))
        source = str(r.get("source_file", "unknown"))
        program_name = str(r.get("program_name", ""))

        if program_name:
            context_parts.append(f"- [{program_name}] (출처: {source})")
        else:
            context_parts.append(f"- 출처: {source}")
        context_parts.append(f"  Q: {q}")
        context_parts.append(f"  A: {a}")
        context_parts.append("")

    return "\n".join(context_parts)


# ===== 인덱스 초기화 스크립트 =====
def init_faq_index():
    """
    FAQ CSV를 읽어서 FAISS 인덱스 생성 (최초 1회 실행)
    터미널에서:
        python faq.py
    """
    print("=" * 50)
    print("FAQ RAG 인덱스 초기화 시작 (CSV 기반)")
    print("=" * 50)

    docs = load_faq_csv()
    if not docs:
        print("❌ 로드된 FAQ가 없습니다.")
        return

    build_faiss_index(docs)

    print("=" * 50)
    print("✅ FAQ RAG 인덱스 초기화 완료!")
    print("=" * 50)


if __name__ == "__main__":
    init_faq_index()