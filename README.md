# 🏘️ 부동산 정책 Hybrid Graph RAG — Streamlit

**정책 문서 262건의 원문 + 2차 중복 통합 RDF 그래프 + OpenAI GPT**를 결합한 Streamlit 웹 앱입니다. API 키는 앱 좌측에서 방문자가 직접 입력합니다.

## 기능

- 원문 기반 한국어 질문 검색: 659개 청크, 문자 TF-IDF 벡터 검색
- RDF 그래프 관계 확장: 종합대책, 세부정책, 법령 언급, 정책 버전, 변경 이벤트
- 사용자가 선택하면 OpenAI 임베딩(`text-embedding-3-small`)으로 의미 검색 추가
- OpenAI Responses API(`gpt-4.1-mini` 기본)로 검색 근거 기반 답변 생성
- 답변에 `[문서 75748]` 등 출처 ID를 표시하고 원문·관계·검증상태 확인
- 문서별 RDF 정책 네트워크 그래프 탐색
- **API 키를 저장·공유 캐싱하지 않음** (각 이용자가 앱에서 입력)

> 중요: 데이터의 법령·후속관계 상당수는 규칙 기반 **자동 추출 후보**입니다. `mentionsLegalNorm`은 법적 근거가 아닙니다. 등록일도 반드시 시행일과 일치하지 않습니다.

## 1. 로컬 실행

Python **3.11 ~ 3.12** 권장.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

웹 브라우저에서 `http://localhost:8501` 접속 후 사이드바의 **OpenAI API Key**를 입력합니다. 키를 입력하지 않아도 원문 검색과 RDF 탐색은 작동합니다. 모델 호출은 입력한 키의 OpenAI 계정에 청구됩니다.

## 2. GitHub 업로드 및 Streamlit Community Cloud 배포

1. GitHub에서 새 저장소(예: `realestate-policy-rag`)를 만듭니다.
2. **이 ZIP 파일의 압축을 풀고**, 그 안의 `streamlit_app.py`, `rag.py`, `requirements.txt`, `build_chunks.py`, `.streamlit/config.toml`, `data/` 등의 파일을 저장소 **최상위 디렉터리**에 올립니다. `data/graph.ttl`과 `data/policies.csv`도 반드시 포함합니다.
3. [Streamlit Community Cloud](https://share.streamlit.io/)에 GitHub 계정으로 로그인합니다.
4. **Create app** → 저장소·브랜치 선택 → Main file path에 `streamlit_app.py`를 입력하고 배포합니다.
5. 발급된 `https://...streamlit.app` 주소로 사이트에 접속할 수 있습니다.

**GitHub Secrets나 `secrets.toml`에 API 키를 넣을 필요가 없습니다.** 방문자가 직접 키를 입력하는 BYOK(Bring Your Own Key) 방식입니다. `data/` 안의 정책 원문도 GitHub 저장소에 함께 업로드되므로, **공개 저장소라면 데이터도 공개**됩니다. 저장소 공개 여부와 원본 데이터의 이용 조건을 확인하세요.

## 3. 사용 방법

1. 좌측 사이드바에 `sk-...` 형식의 OpenAI API 키를 입력합니다.
2. 처음에는 **TF-IDF + RDF** 모드로 질의하고 답변 품질을 확인하세요. (검색 무료, GPT 답변은 유료)
3. 의미 검색을 원하면 **OpenAI 임베딩 + RDF** 모드를 선택하고 **임베딩 인덱스 생성** 버튼을 누릅니다. 659개 청크에 대해 OpenAI API를 호출하므로 비용이 발생합니다.
4. 질문을 입력하면 검색한 정책 원문과 RDF 관계를 바탕으로 GPT가 답변합니다.
5. `검색된 정책 원문·RDF 관계 보기`에서 원문과 자동 추출 신뢰도를 확인합니다.
6. `RDF 관계 탐색` 탭에서 특정 문서의 정책·세부정책 구조를 살펴봅니다.

### 질문 예시

- `2023년 전세사기 피해자 지원 보완방안에 포함된 세부정책은?`
- `2024년 주택공급 확대방안에서 언급하는 법령은? 법적 근거가 확정된 것만 구분해줘.`
- `전세사기 피해자 지원과 관련된 2023년 및 2024년 정책을 비교해줘.`
- `주택공급 정책에서 정부가 제시한 정량적 목표를 원문 출처와 함께 정리해줘.`

## 4. 프로젝트 구조

```text
realestate-policy-rag/
├── streamlit_app.py          # Streamlit UI, API 키 입력, 질의/시각화
├── rag.py                    # RDF 검색, TF-IDF, 임베딩, GPT 프롬프트
├── build_chunks.py           # 원문 → 검색 청크 생성
├── requirements.txt
├── .streamlit/config.toml
├── .gitignore
├── tests/test_rag.py
└── data/
    ├── policies.csv         # 정제된 262건 원문
    ├── chunks.csv           # 원문 청크 659개
    ├── graph.ttl            # 중복 통합된 RDF 그래프
    ├── ontology_schema.ttl  # 재설계한 OWL/Turtle 스키마
    ├── entities.csv         # 5,691개 통합 개체
    ├── relations.csv        # 4,381건 관계 주장 및 검증상태
    ├── evidence.csv         # 관계별 추출 근거
    └── followup_candidates.csv
```

## 5. 갱신 및 테스트

원문 정책 CSV를 바꾸면 아래 명령으로 청크를 다시 생성합니다.

```bash
python build_chunks.py
python -m pytest tests -q
```

원천 파일을 갱신했을 경우에는 새 문서 ID가 RDF·관계 CSV에도 반영됐는지 별도 검사해야 합니다. `chunks.csv`만 갱신해도 RDF가 자동 갱신되는 것은 아닙니다.

## 6. 개인정보·비용·운영 주의사항

- OpenAI API 키는 `st.text_input(type="password")`로 받고, 서버의 영구 파일·GitHub 저장소·공유 캐시에 저장하지 않습니다. **세션이 지속되는 동안에는 앱 서버의 세션 상태에서 처리될 수 있으므로 신뢰할 수 있는 서버에만 입력하세요.**
- 이용자가 입력한 질문과 검색된 정책 원문·관계는 OpenAI API로 전송됩니다. API 요청에는 `store=False`를 설정했습니다. 이는 API의 저장 옵션이며 모든 데이터 처리·보존이 0이라는 의미는 아닙니다.
- `data/graph.ttl` 약 23MB, RDF 약 12만 트리플로, 최초 로딩 시 수 초와 수백 MB 메모리가 사용될 수 있습니다.
- 임베딩 생성은 **요청한 브라우저 세션의 메모리에만** 유지되므로 새 세션에서는 다시 생성해야 하며 API 비용이 다시 발생할 수 있습니다.
- 모든 RDF 관계가 사실로 검증된 것은 아닙니다. 특히 **법률 조항·정책 시행 여부·후속정책 인과관계**는 공식 자료와 별도 교차 검증하세요.
- 서비스 공개 시 이용자 키 남용·과금 문제에 대비해 요청량 제한, 서비스 고지문, 접근 제어를 추가하는 것이 좋습니다.

## 참고 문서

- [OpenAI Responses API](https://developers.openai.com/api/docs/guides/text)
- [OpenAI Embeddings](https://developers.openai.com/api/docs/guides/embeddings)
- [Streamlit Community Cloud 배포](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
