import os
import re
import time
import pandas as pd
import altair as alt
import streamlit as st
from dotenv import load_dotenv

#########################
# 1) Pinecone (v2) Import
#########################
from pinecone import Pinecone, ServerlessSpec

#########################
# 2) LangChain & OpenAI
#########################
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

########################################
# === 환경변수 로드 ===
########################################
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY or not PINECONE_ENV:
    raise ValueError("Pinecone API Key(또는 ENV)가 .env에 설정되지 않았습니다.")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key가 .env에 설정되지 않았습니다.")

########################################
# === Pinecone 인스턴스 & 인덱스 세팅 ===
########################################
# 1) Pinecone 초기화
pc = Pinecone(api_key=PINECONE_API_KEY)

# 2) 인덱스 이름 및 파라미터
index_name = "quickstart"       # 원하는 인덱스 이름
dimension = 1536               # OpenAIEmbeddings(ada-002) 차원
metric = "cosine"              # 보통 cosine 또는 dotproduct

# 3) 인덱스가 없으면 생성
all_indexes = pc.list_indexes().names()
if index_name not in all_indexes:
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(
            cloud="aws",       # 예: Starter(무료) 플랜이면 gcp us-central1 권장
            region="us-east-1"  
            # AWS us-west-2 등은 무료플랜에서 안 될 수 있음 -> 플랜 상황에 맞게 수정
        )
    )

# 4) 인덱스 핸들
my_index = pc.Index(index_name)

########################################
# === CSV -> 임베딩 -> Pinecone Upsert ===
########################################
# (실제로는 '처음 한 번'만 업로드하고, 데이터가 변동될 때만 업데이트 권장)
loader = CSVLoader(file_path="tasks.csv")
documents = loader.load()   # List[Document], doc.page_content 등 포함

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=OPENAI_API_KEY
)

# 1) texts, metadata 구성
texts = [doc.page_content for doc in documents]
metadatas = [{"source": f"row_{i}"} for i in range(len(texts))]

# 2) OpenAIEmbeddings로 각 문서 임베딩
vectors = []
for i, (text, meta) in enumerate(zip(texts, metadatas)):
    vector = embeddings.embed_query(text)   # 혹은 embed_documents([text])[0]
    # Pinecone 업서트용 record
    # 'id', 'values', 'metadata'
    record = {
        "id": f"task_{i}",
        "values": vector,
        "metadata": meta
    }
    vectors.append(record)

# 3) Upsert
my_index.upsert(vectors=vectors, namespace="tasks_namespace")

# Pinecone는 eventually consistent
# -> upsert 후 일정 시간 기다릴 수 있음
time.sleep(5)  # 데이터가 인덱싱될 시간

########################################
# === Pinecone 검색 함수 ===
########################################
def retrieve_tasks(query: str, k: int = 5):
    """
    1) query 임베딩 -> Pinecone my_index.query()
    2) 유사 top-k 문서 -> doc.page_content (저장 시에는 없지만, 예시로 metadata에 담거나 별도DB 참조)
    이 코드에서는 tasks.csv text를 그대로 upsert했으므로, metadata에 text를 따로 안 넣었음.
    => "값" (values)은 hidden이므로, rank만 받음
    """
    # Query를 OpenAIEmbeddings로 임베딩
    query_vector = embeddings.embed_query(query)

    result = my_index.query(
        namespace="tasks_namespace",
        vector=query_vector,
        top_k=k,
        include_values=False,
        include_metadata=True
    )

    # result.matches -> [{id, score, metadata}, ...]
    # 여기서는 원본 text가 없으므로, 'id'나 'metadata'만 확인 가능
    # tasks.csv의 구체 content(문자열)가 필요하면, doc.page_content를 DB나 별도 구조로 관리해야 함.
    # 예: 여기선 for문으로 vectors에 (id->text) 맵핑을 따로 보관할 수도 있음.

    # 편의상, "id"에 매칭되는 original text를 역참조하자 (vectors 리스트 사용)
    # => 실서비스에서는 DB or dict 등으로 해시맵을 구성하는 게 낫습니다.
    hits = []
    for match in result.matches:
        match_id = match.id
        # "task_3" -> integer index
        idx = int(match_id.replace("task_", ""))
        original_text = texts[idx]   # CSV의 page_content
        hits.append(original_text)

    return hits

########################################
# === parse_fields: 태스크 세부추출 ===
########################################
def parse_fields(text: str) -> dict:
    fields = [
        "TaskID", "TaskName", "ParentTaskName", "AssigneeList", "Reviewer",
        "Status", "StartDate", "DueDate", "Priority", "Urgency", "Space",
        "ApprovalStatus", "TaskDetail", "EvaluationCriteria", "AssigneeFeedback",
        "AttachedFiles", "RelatedTaskList"
    ]
    parsed = {}
    for field in fields:
        pattern = rf"{field}:\s*(.*?)(?=\s+[A-Z][a-zA-Z]+:|$)"
        match = re.search(pattern, text)
        if match:
            parsed[field] = match.group(1).strip()
        else:
            parsed[field] = ""
    return parsed

import yaml
from langchain.prompts import PromptTemplate

# 1) prompts.yaml 파일 로드
with open("prompts.yaml", "r", encoding="utf-8") as f:
    prompt_data = yaml.safe_load(f)
    # prompt_data = {
    #   "summary_prompt": {
    #       "description": "...",
    #       "input_variables": [...],
    #       "template": "..."
    #   },
    #   "decision_prompt": {...}
    # }

# 2) Summaries PromptTemplate 생성
summary_info = prompt_data["summary_prompt"]
summary_prompt = PromptTemplate(
    input_variables=summary_info["input_variables"],
    template=summary_info["template"]
)

# 3) OX Decision PromptTemplate 생성
decision_info = prompt_data["decision_prompt"]
decision_prompt = PromptTemplate(
    input_variables=decision_info["input_variables"],
    template=decision_info["template"]
)

# 이제 summary_prompt, decision_prompt를
# LLMChain(llm=..., prompt=summary_prompt) 등으로 활용하면 됩니다.


summaries_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
summaries_chain = LLMChain(llm=summaries_llm, prompt=summary_prompt)

def summarize_tasks(task_texts: list, message: str) -> str:
    joined = "\n\n".join(task_texts)
    summary = summaries_chain.run(message=message, relevant_tasks=joined)
    return summary

decision_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
decision_chain = LLMChain(llm=decision_llm, prompt=decision_prompt)

def generate_decision(message: str):
    relevant_list = retrieve_tasks(message, k=5)
    joined_tasks = "\n\n".join(relevant_list)
    decision_result = decision_chain.run(
        message=message,
        relevant_tasks=joined_tasks
    )
    return decision_result, relevant_list

########################################
# === 시각화 (Altair) ===
########################################
def visualize_tasks(task_dicts: list):
    if not task_dicts:
        st.write("검색 결과가 없습니다.")
        return

    df = pd.DataFrame(task_dicts)

    st.subheader("검색 결과 테이블")
    st.dataframe(df)

    # O/X 분포 파이 차트
    if "ApprovalStatus" in df.columns:
        status_counts = df["ApprovalStatus"].value_counts().reset_index()
        status_counts.columns = ["ApprovalStatus", "Count"]
        chart = alt.Chart(status_counts).mark_arc().encode(
            theta="Count",
            color="ApprovalStatus",
            tooltip=["ApprovalStatus", "Count"]
        ).properties(title="검색된 태스크들의 O/X 분포")
        st.altair_chart(chart, use_container_width=True)

########################################
# === Streamlit 메인 앱 ===
########################################
def main():
    st.set_page_config(
        page_title="OX llm prototype (Pinecone, Official Style)",
        page_icon=":globe_with_meridians:"
    )
    st.title("OX llm prototype (Pinecone DB, Official) | Task 검색 / OX Decision")

    tabs_list = st.tabs(["Task Search & Visualization", "OX Decision Helper"])

    # [탭1] 검색 & 시각화
    with tabs_list[0]:
        st.subheader("Task 검색 & 시각화 (Pinecone)")

        query = st.text_input("검색어를 입력하세요 (예: '마케팅 기획', '신메뉴 개발' 등)")
        if query:
            st.write(f"'{query}' 관련 유사 태스크 검색...")
            results = retrieve_tasks(query, k=5)  # texts

            # parse fields
            parsed_list = [parse_fields(r) for r in results]
            visualize_tasks(parsed_list)

            if results:
                st.subheader("검색 결과 요약 (LLM)")
                summary_text = summarize_tasks(results, query)
                st.info(summary_text)

    # [탭2] OX Decision Helper
    with tabs_list[1]:
        st.subheader("OX Decision Helper (Pinecone)")

        new_task = st.text_area("새로운 Task (O/X 판단 대상)")
        if new_task:
            st.write("유사 테스크 검색 중...")
            decision, relevant_tasks = generate_decision(new_task)

            st.subheader("의사결정 결과 (LLM)")
            st.info(decision)

            with st.expander("참조된 유사 테스크 보기"):
                for i, rt in enumerate(relevant_tasks, start=1):
                    st.write(f"**유사테스크 {i}**\n{rt}\n")

if __name__ == "__main__":
    main()
