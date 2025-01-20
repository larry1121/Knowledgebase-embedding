import os
import re
import time
import pandas as pd
import altair as alt
import streamlit as st
from dotenv import load_dotenv

##########################
# 1) Pinecone (v2)
##########################
from pinecone import Pinecone, ServerlessSpec

##########################
# 2) LangChain & OpenAI
##########################
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

##########################
# 3) YAML for Prompts
##########################
import yaml

##########################
# 4) Moderation Chain
##########################
from langchain.chains.moderation import OpenAIModerationChain

##########################
# 전역 설정(상수)
##########################
INDEX_NAME = "quickstart"
DIMENSION = 1536
METRIC = "cosine"
NAMESPACE = "tasks_namespace"
CSV_FILE = "tasks.csv"
PROMPTS_FILE = "prompts.yaml"

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY or not PINECONE_ENV:
    raise ValueError("Pinecone API Key 또는 ENV가 .env에 설정되지 않았습니다.")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key가 .env에 설정되지 않았습니다.")


########################################################################
# [Safety Module] OpenAIModerationChain
########################################################################

moderation_chain = OpenAIModerationChain(
    openai_api_key=OPENAI_API_KEY,
    error=False  # error=True시 flagged되면 Exception 발생
)

def safe_input(user_text: str) -> str:
    """
    사용자 입력을 ModerationChain으로 검사.
    - flagged=True면 경고(st.warning) + ""(빈문자) 반환
    """
    mod_result = moderation_chain.invoke({"input": user_text})
    print("70 : "+ mod_result["output"])
    if mod_result["output"] == "Text was found that violates OpenAI's content policy.":
        # 유해/부적절하다고 판단되면 경고 + 입력값 비움
        st.warning("경고: 입력 내용에서 부적절한 표현이 감지되었습니다. 해당 입력은 무시됩니다.")
        st.stop()
        return ""
    return user_text

def safe_output(llm_text: str) -> str:
    """
    LLM 출력에 대해 ModerationChain으로 검사.
    - flagged=True면 [REDACTED] 처리
    """
    mod_result = moderation_chain.invoke({"input": llm_text})
    if mod_result["output"] is True:
        return "[REDACTED: 부적절한 출력이 감지되어 숨김 처리]"
    return llm_text

########################################################################
# [UTILITY FUNCTIONS]
########################################################################

def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    all_indexes = pc.list_indexes().names()
    if INDEX_NAME not in all_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    my_index = pc.Index(INDEX_NAME)
    return pc, my_index

def load_csv_and_embed(csv_file: str, openai_api_key: str):
    loader = CSVLoader(file_path=csv_file)
    documents = loader.load()

    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=openai_api_key
    )

    texts = [doc.page_content for doc in documents]
    metadatas = [{"source": f"row_{i}"} for i in range(len(texts))]

    vectors = []
    for i, (text, meta) in enumerate(zip(texts, metadatas)):
        vec = embeddings.embed_query(text)
        record = {
            "id": f"task_{i}",
            "values": vec,
            "metadata": meta
        }
        vectors.append(record)
    return texts, vectors

def upsert_documents(my_index, vectors, namespace=NAMESPACE):
    my_index.upsert(vectors=vectors, namespace=namespace)
    time.sleep(5)

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
        parsed[field] = match.group(1).strip() if match else ""
    return parsed

def visualize_tasks(task_dicts: list):
    if not task_dicts:
        st.write("검색 결과가 없습니다.")
        return
    df = pd.DataFrame(task_dicts)
    st.subheader("검색 결과 테이블")
    st.dataframe(df)
    if "ApprovalStatus" in df.columns:
        status_counts = df["ApprovalStatus"].value_counts().reset_index()
        status_counts.columns = ["ApprovalStatus", "Count"]
        chart = alt.Chart(status_counts).mark_arc().encode(
            theta="Count",
            color="ApprovalStatus",
            tooltip=["ApprovalStatus", "Count"]
        ).properties(title="검색된 태스크들의 O/X 분포")
        st.altair_chart(chart, use_container_width=True)

########################################################################
# [PROMPT & LLM CHAIN]
########################################################################

def load_prompts_yaml(prompts_file: str):
    with open(prompts_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data

def create_prompt_template(info: dict) -> PromptTemplate:
    return PromptTemplate(
        input_variables=info["input_variables"],
        template=info["template"]
    )

def build_llmchain_for_summary(summary_prompt: PromptTemplate):
    summaries_llm = ChatOpenAI(temperature=0, model="gpt-4o-mini-2024-07-18")
    return LLMChain(llm=summaries_llm, prompt=summary_prompt)

def build_llmchain_for_decision(decision_prompt: PromptTemplate):
    decision_llm = ChatOpenAI(temperature=0, model="gpt-4o-mini-2024-07-18")
    return LLMChain(llm=decision_llm, prompt=decision_prompt)

########################################################################
# [BUSINESS LOGIC]
########################################################################

def retrieve_tasks(query: str, embeddings_obj: OpenAIEmbeddings, my_index, texts, k: int = 5):
    query_vector = embeddings_obj.embed_query(query)
    result = my_index.query(
        namespace=NAMESPACE,
        vector=query_vector,
        top_k=k,
        include_values=False,
        include_metadata=True
    )
    hits = []
    for match in result.matches:
        match_id = match.id
        idx = int(match_id.replace("task_", ""))
        original_text = texts[idx]
        hits.append(original_text)
    return hits

def summarize_tasks(task_chain: LLMChain, task_texts: list, message: str) -> str:
    # 사용자 입력 + 유사 문서들 모두 safe_input 처리
    safe_msg = safe_input(message)
    safe_joined = safe_input("\n\n".join(task_texts))

    # 체인 실행
    raw_output = task_chain.run(message=safe_msg, relevant_tasks=safe_joined)
    return safe_output(raw_output)

def generate_decision(task_chain: LLMChain, message: str, relevant_texts: list):
    safe_msg = safe_input(message)
    safe_joined = safe_input("\n\n".join(relevant_texts))

    raw_output = task_chain.run(message=safe_msg, relevant_tasks=safe_joined)
    return safe_output(raw_output)

########################################################################
# [STREAMLIT MAIN APP]
########################################################################

def main():
    st.set_page_config(
        page_title="OX llm prototype (ModerationChain with Warning)",
        page_icon=":globe_with_meridians:"
    )
    st.title("OX llm prototype (ModerationChain) | Task 검색 / OX Decision")
    
    # 1) Pinecone 초기화
    pc, my_index_obj = init_pinecone()

    # 2) CSV 로드 & 임베딩
    texts_list, vectors_data = load_csv_and_embed(CSV_FILE, OPENAI_API_KEY)

    # 3) Pinecone Upsert
    upsert_documents(my_index_obj, vectors_data, namespace=NAMESPACE)

    # 4) Prompts 로드 & LLMChain
    prompt_data = load_prompts_yaml(PROMPTS_FILE)
    summary_prompt_obj = create_prompt_template(prompt_data["summary_prompt"])
    summary_chain = build_llmchain_for_summary(summary_prompt_obj)

    decision_prompt_obj = create_prompt_template(prompt_data["decision_prompt"])
    decision_chain_obj = build_llmchain_for_decision(decision_prompt_obj)

    # 5) 검색용 임베딩
    embed_for_query = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=OPENAI_API_KEY
    )

    tabs_list = st.tabs(["Task Search & Visualization", "OX Decision Helper"])

    # [TAB1] 검색
    with tabs_list[0]:
        st.subheader("Task 검색 & 시각화")
        query = st.text_input("검색어를 입력하세요...")

        if query:
            # 안전 검사
            checked_query = safe_input(query)
            if not checked_query:
                st.warning("검색어가 무효화되었습니다.")
                return

            st.write(f"'{checked_query}' 관련 유사 태스크 검색 중...")
            results = retrieve_tasks(checked_query, embed_for_query, my_index_obj, texts_list)

            parsed_list = [parse_fields(r) for r in results]
            visualize_tasks(parsed_list)

            if results:
                st.subheader("검색 결과 요약 (LLM)")
                summary_text = summarize_tasks(summary_chain, results, checked_query)
                st.info(summary_text)

    # [TAB2] OX Decision
    with tabs_list[1]:
        st.subheader("OX Decision Helper")
        new_task = st.text_area("새로운 Task (O/X 판단 대상)")

        if new_task:
            checked_task = safe_input(new_task)
            if not checked_task:
                st.warning("입력이 무효화되었습니다.")
                return

            st.write("유사 테스크 검색 중...")
            relevant_texts = retrieve_tasks(checked_task, embed_for_query, my_index_obj, texts_list)

            decision_result = generate_decision(decision_chain_obj, checked_task, relevant_texts)

            st.subheader("의사결정 결과 (LLM)")
            st.info(decision_result)

            with st.expander("참조된 유사 테스크 보기"):
                for i, rt in enumerate(relevant_texts, start=1):
                    st.write(f"**유사테스크 {i}**\n{rt}\n")


if __name__ == "__main__":
    main()
