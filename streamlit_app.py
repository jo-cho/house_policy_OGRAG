"""한국 부동산 정책 Hybrid Graph RAG용 Streamlit 웹 앱."""

from __future__ import annotations

import json
from collections import Counter

import pandas as pd
import streamlit as st
from openai import OpenAI

from rag import (
    EMBED_MODEL,
    answer_with_openai,
    embed_corpus,
    embed_question,
    graph_document_edges,
    load_bundle,
    retrieve,
)


st.set_page_config(
    page_title="부동산 정책 Graph RAG",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """<style>
    .block-container {max-width: 1260px; padding-top: 1.5rem;}
    [data-testid="stMetric"] {padding: 0.6rem; border: 1px solid #d9dfe6; border-radius: 0.65rem;}
    .stChatMessage {border-radius: .7rem;}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="정책 원문·RDF 그래프를 읽는 중입니다...")
def get_data():
    # 공개 정책 데이터만 서버 전체에서 공유하고 사용자 키와 대화는 캐시하지 않는다.
    return load_bundle()


try:
    bundle = get_data()
except Exception as exc:
    st.error("필수 데이터 파일을 읽을 수 없습니다. GitHub의 data/ 디렉터리를 확인하세요.")
    st.exception(exc)
    st.stop()


def show_evidence(result: dict, key_prefix: str) -> None:
    """검색 출처를 문서 ID와 관계 검증 상태 중심으로 보여준다."""
    chunks = result.get("chunks", [])
    if not chunks:
        st.warning("검색된 원문이 없습니다.")
        return
    st.caption("검색 방식: " + result.get("mode", ""))
    for i, item in enumerate(chunks, 1):
        with st.expander(f"T{i} · [{item['doc_id']}] {item['title']} · 검색점수 {item['score']:.3f}"):
            st.caption(f"등록일 {item['reg_date']} · {item['publisher']} · {item['source_column']}")
            st.text(item["text"])

    relations = result.get("relations", [])
    if relations:
        st.markdown("**RDF 관계 주장 (검증되지 않은 후보 포함)**")
        rows = [{
            "문서 ID": r["doc_id"], "주어": r["subject"],
            "관계": r["predicate_label"], "목적어": r["object"],
            "검증 상태": r["status"], "신뢰도": r["confidence"],
            "원문 근거": r["evidence"],
        } for r in relations]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    followups = result.get("followups", [])
    if followups:
        with st.expander(f"후속대책 연결 후보 {len(followups)}건 (확정 관계 아님)"):
            st.dataframe(pd.DataFrame(followups), use_container_width=True, hide_index=True)


def graph_dot(edges: list[dict]) -> str:
    """문서 중심 관계를 간결한 방향 그래프로 렌더링할 DOT 코드를 생성한다."""
    lines = ['digraph G {', 'rankdir=LR;', 'graph [bgcolor="transparent",pad="0.15"];',
             'node [shape="box", style="rounded,filled", fillcolor="#f1f5f9", color="#94a3b8", '
             'fontname="Noto Sans CJK KR", fontsize=10];',
             'edge [color="#64748b", fontname="Noto Sans CJK KR", fontsize=9];']
    ids: dict[str, str] = {}
    for edge in edges:
        for field in ("subject_id", "object_id"):
            uri = edge[field]
            if uri not in ids:
                node = f"n{len(ids)}"
                ids[uri] = node
                label = str(edge["subject_label" if field == "subject_id" else "object_label"])
                if len(label) > 38:
                    label = label[:35] + "…"
                lines.append(f'{node} [label={json.dumps(label, ensure_ascii=False)}];')
        label = edge["predicate_label"]
        lines.append(
            f'{ids[edge["subject_id"]]} -> {ids[edge["object_id"]]} '
            f'[label={json.dumps(label, ensure_ascii=False)}];'
        )
    lines.append('}')
    return "\n".join(lines)


with st.sidebar:
    st.header("⚙️ 검색 및 모델 설정")
    api_key = st.text_input(
        "OpenAI API Key", type="password", placeholder="sk-...",
        help="입력한 키는 현재 Streamlit 사용자 세션에서만 사용하며 소스 코드·CSV에 저장하지 않습니다.",
    )
    model = st.selectbox("답변 GPT 모델", ["gpt-4.1-mini", "gpt-4.1", "gpt-5-mini"], index=0)
    search_mode = st.radio(
        "검색 방식",
        ["TF-IDF + RDF (즉시 검색)", "OpenAI 임베딩 + RDF (의미 검색)"],
        help="무료 모드는 문자 기반 TF-IDF와 RDF 관계 검색을 사용합니다. 임베딩 모드는 추가 API 호출 비용이 발생합니다.",
    )
    top_k = st.slider("원문 검색 건수", min_value=3, max_value=12, value=7)

    if "embedding_matrix" not in st.session_state:
        st.session_state.embedding_matrix = None
    if search_mode.startswith("OpenAI"):
        st.caption(f"임베딩 모델: `{EMBED_MODEL}` · 512차원")
        if st.session_state.embedding_matrix is not None:
            st.success(f"이 세션에 임베딩 {len(st.session_state.embedding_matrix)}개 준비됨")
        if st.button("임베딩 인덱스 생성 / 재생성", disabled=not bool(api_key), use_container_width=True):
            progress_bar = st.progress(0, text="OpenAI로 원문 임베딩 생성 중")
            try:
                client = OpenAI(api_key=api_key, timeout=60.0, max_retries=2)
                def update_progress(done: int, total: int) -> None:
                    progress_bar.progress(done / total, text=f"임베딩 생성 {done:,}/{total:,} 청크")
                st.session_state.embedding_matrix = embed_corpus(client, bundle.chunks, update_progress)
                st.success("현재 세션의 의미 검색 인덱스를 생성했습니다.")
            except Exception:
                st.session_state.embedding_matrix = None
                st.error("임베딩 생성에 실패했습니다. API 키 권한, 잔액, 네트워크를 확인하세요.")
            finally:
                progress_bar.empty()
        if st.session_state.embedding_matrix is None:
            st.info("임베딩 인덱스가 없으면 TF-IDF + RDF 검색으로 자동 전환합니다.")

    st.divider()
    if st.button("대화 내용 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("사용자 API 키는 GitHub에 포함되지 않습니다. 인덱스는 세션 메모리에만 보관됩니다.")
    st.caption("원문은 정책 문서 기록이며 법령 현행 여부를 보증하지 않습니다.")


st.title("🏘️ 부동산 정책 Graph RAG")
st.caption("정책 문서 262건 · RDF 정책/법령/세부정책 관계 · 출처 기반 GPT 질의응답")
metric_cols = st.columns(4)
metric_cols[0].metric("정책 문서", f"{len(bundle.docs):,}건")
metric_cols[1].metric("원문 청크", f"{len(bundle.chunks):,}개")
metric_cols[2].metric("통합 개체", f"{len(bundle.entities):,}개")
metric_cols[3].metric("관계 주장", f"{len(bundle.relations):,}건")

chat_tab, explore_tab, info_tab = st.tabs(["💬 정책 질문", "🕸️ RDF 관계 탐색", "📊 데이터 및 사용안내"])

with chat_tab:
    st.info("예시: ‘2023년 전세사기 피해자 지원 보완방안의 세부정책과 법령 언급을 정리해줘.’")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("search"):
                with st.expander("검색된 정책 원문·RDF 관계 보기"):
                    show_evidence(message["search"], f"past{index}")

    user_prompt = st.chat_input("부동산 정책, 세부정책, 법령, 후속대책에 관해 질문하세요")
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
        with st.chat_message("assistant"):
            with st.spinner("정책 원문과 RDF 그래프를 검색하고 있습니다..."):
                embedding_matrix = None
                question_vector = None
                if search_mode.startswith("OpenAI") and st.session_state.embedding_matrix is not None and api_key:
                    try:
                        client = OpenAI(api_key=api_key, timeout=60.0, max_retries=2)
                        question_vector = embed_question(client, user_prompt)
                        embedding_matrix = st.session_state.embedding_matrix
                    except Exception:
                        st.warning("질문 임베딩 생성에 실패하여 TF-IDF 검색으로 전환했습니다.")
                search = retrieve(bundle, user_prompt, top_k=top_k,
                                  embeddings=embedding_matrix, question_embedding=question_vector)
            if not search["chunks"]:
                reply = "관련 정책 원문을 찾지 못했습니다. 질문의 정책명 또는 연도를 다르게 입력해 주세요."
            elif not api_key:
                reply = "관련 정책 문서를 검색했습니다. **좌측에 OpenAI API Key를 입력하면 검색된 근거를 바탕으로 GPT 답변을 생성합니다.** 아래 근거를 먼저 확인할 수 있습니다."
            else:
                with st.spinner(f"{model} 모델이 근거를 검토하고 있습니다..."):
                    try:
                        client = OpenAI(api_key=api_key, timeout=90.0, max_retries=2)
                        history = st.session_state.messages[:-1]
                        reply = answer_with_openai(client, user_prompt, search, model=model, conversation=history)
                        if not reply:
                            reply = "모델이 빈 답변을 반환했습니다. 다시 질문해 주세요."
                    except Exception:
                        reply = "OpenAI API 호출에 실패했습니다. 키·권한·모델 사용 가능 여부와 크레딧을 확인하세요. 검색 근거는 아래에서 볼 수 있습니다."
            st.markdown(reply)
            with st.expander("검색된 정책 원문·RDF 관계 보기", expanded=not bool(api_key)):
                show_evidence(search, "new")
            st.session_state.messages.append({"role": "assistant", "content": reply, "search": search})

with explore_tab:
    st.subheader("문서별 정책 관계 탐색")
    st.caption("CSV의 문서 ID로 선택한 후 실제 RDF 그래프에서 문서→정책→세부정책 연결을 조회합니다.")
    documents = bundle.docs.to_dict("records")
    selection = st.selectbox(
        "정책 문서 선택",
        options=list(range(len(documents))),
        format_func=lambda i: f"[{documents[i]['ID']}] {documents[i]['TITLE_CLEAN']} ({documents[i]['REG_DATE']})",
        index=next((i for i, row in enumerate(documents) if row["ID"] == "75748"), 0),
    )
    doc = documents[selection]
    st.markdown(f"**{doc['TITLE_CLEAN']}**")
    st.caption(f"문서 ID {doc['ID']} · 등록일 {doc['REG_DATE']} · 발행기관 {doc['PUBLISHER']}")
    edges = graph_document_edges(bundle, doc["ID"])
    if edges:
        st.graphviz_chart(graph_dot(edges), use_container_width=True)
        st.dataframe(pd.DataFrame([{
            "주어": e["subject_label"], "관계": e["predicate_label"], "목적어": e["object_label"]
        } for e in edges]), use_container_width=True, hide_index=True)
    else:
        st.warning("이 문서에서 연결된 RDF 관계를 찾지 못했습니다.")
    with st.expander("원문 보기"):
        st.markdown("**정책 배경**")
        st.text(doc["POLICY_CONTEXT_CLEAN"])
        st.markdown("**정책 본문**")
        st.text(doc["ABSTRACT_CLEAN"])
    st.caption("원문에 등장한 법령을 자동 추출한 경우 법적 근거로 확정하지 않습니다.")

with info_tab:
    st.subheader("현재 앱의 구조")
    st.markdown(
        """- **그래프 검색**: 2차 중복 통합 RDF(약 12만 트리플)에서 문서·정책·세부정책·법령의 관계를 탐색합니다.
- **벡터 검색**: 기본 한국어 문자 TF-IDF. 선택적으로 `text-embedding-3-small`의 의미 벡터를 생성해 함께 사용합니다.
- **청크 검색**: 정제된 정책 원문을 의미/줄 단위로 나누며 문서 ID와 등록일을 보존합니다.
- **답변 생성**: 앱에서 직접 입력한 API 키로 OpenAI Responses API를 호출합니다.
- **근거 추적**: 검색 원문, 추출 관계 및 후보 상태를 함께 표시합니다."""
    )
    st.warning("정책 및 법령 관계의 상당수는 규칙 기반 자동 추출 후보입니다. 실제 법령 적용·시행 여부는 국가법령정보 등 공신력 있는 자료와 대조해야 합니다.")
    counts = bundle.relations["predicate"].value_counts().rename_axis("관계 유형").reset_index(name="건수")
    st.dataframe(counts, use_container_width=True, hide_index=True)
    st.download_button(
        "관계 유형 통계 CSV 다운로드",
        data=counts.to_csv(index=False).encode("utf-8-sig"),
        file_name="relationship_counts.csv", mime="text/csv",
    )
    st.caption("GitHub에 공개 저장소로 배포하면 포함된 정책 원문과 RDF도 공개됩니다. 공개 가능 여부를 먼저 확인하세요.")
