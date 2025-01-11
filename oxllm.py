import streamlit as st
import pandas as pd
import re
import altair as alt

from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

############################################################
# 0. 공통: CSV -> FAISS VectorStore 로드
############################################################
#   하나의 앱에서 사용하므로, 글로벌로 한번만 로드
loader = CSVLoader(file_path="tasks.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)


############################################################
# A. 공통 함수들
############################################################

def parse_fields(text: str) -> dict:
    """
    "TaskID: ... TaskName: ..." 식 문자열을 필드별 dict로 변환
    """
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


def retrieve_tasks(query: str, k: int = 5):
    """
    입력 쿼리에 대한 유사 태스크 검색 (FAISS)
    """
    similar_docs = db.similarity_search(query, k=k)
    return [doc.page_content for doc in similar_docs]


############################################################
# B. "Search & Visualization" 탭에서 사용할 Summaries, 차트 함수
############################################################

# Summaries
summary_template = """
You are a highly knowledgeable Task Searcher in the OX Human Resource Platform in KOREAN.

Below is message need to answer:
{message}

We also have a historical record of tasks that were decided O or X, which might be relevant about message:
{relevant_tasks}


Please provide a concise summary of the main points or patterns you see across these tasks.
Keep it short, in bullet points or a short paragraph.
"""
summary_prompt = PromptTemplate(input_variables=["message","relevant_tasks"], template=summary_template)
summaries_llm = ChatOpenAI(temperature=0, model="gpt-4o-mini-2024-07-18")
summaries_chain = LLMChain(llm=summaries_llm, prompt=summary_prompt)

def summarize_tasks(task_texts: list,message:str) -> str:
    joined = "\n\n".join(task_texts)
    summary = summaries_chain.run(relevant_tasks=joined,message=message)
    return summary

# 시각화
def visualize_tasks(task_dicts: list):
    """
    검색된 Task들을 DataFrame으로 만들어
    간단한 차트(예: ApprovalStatus 분포)와 표로 보여준다.
    """
    if not task_dicts:
        st.write("검색 결과가 없습니다.")
        return

    df = pd.DataFrame(task_dicts)

    # 표로 표시
    st.subheader("검색 결과 테이블")
    st.dataframe(df)

    # 예: ApprovalStatus 분포 파이 차트
    if "ApprovalStatus" in df.columns:
        status_counts = df["ApprovalStatus"].value_counts().reset_index()
        status_counts.columns = ["ApprovalStatus", "Count"]

        pie_chart = alt.Chart(status_counts).mark_arc().encode(
            theta="Count",
            color="ApprovalStatus",
            tooltip=["ApprovalStatus", "Count"]
        ).properties(
            title="검색된 태스크들의 O/X 분포"
        )
        st.altair_chart(pie_chart, use_container_width=True)


############################################################
# C. "OX Decision Helper" 탭에서 사용할 Prompt/Chain
############################################################
decision_template = """
You are a highly knowledgeable Task Reviewer in the OX Human Resource Platform.

Below is a new task that needs an O or X decision:
{message}

We also have a historical record of tasks that were decided O or X, which might be relevant to this new task:
{relevant_tasks}

Please follow these instructions:

1. Carefully analyze the new task details, including all context such as task name, assignees, priorities, and evaluation criteria.
2. Compare the new task to any relevant historical tasks found in the database (represented by relevant_tasks). Identify patterns or precedents.
3. Determine whether this new task is more likely to be approved (O) or rejected (X).
4. Provide a numerical probability of how likely it is to be approved based on the historical data (for example, “There is a 75% chance this task will receive O”).
5. Give a concise explanation/rationale for your decision, referencing any similar tasks from the historical records.
6. Maintain friendliness and professionalism in your interactions.

Return only the final decision (O or X), the probability, and your short reasoning in KOREAN.

Example format:
Decision: O (85%)
Reasoning: 이 테스크는 과거 O 결정을 받은 유사 사례와 매우 흡사하므로...
"""
decision_prompt = PromptTemplate(
    input_variables=["message", "relevant_tasks"],
    template=decision_template
)

decision_llm = ChatOpenAI(temperature=0, model="gpt-4o-mini-2024-07-18")
decision_chain = LLMChain(llm=decision_llm, prompt=decision_prompt)


def generate_decision(message: str):
    relevant = retrieve_tasks(message, k=5)
    joined_tasks = "\n\n".join(relevant)
    result = decision_chain.run(message=message, relevant_tasks=joined_tasks)
    return result, relevant


############################################################
# D. Streamlit Main App with Tabs
############################################################
def main():
    st.set_page_config(page_title="OX llm prototype", page_icon=":globe_with_meridians:")
    st.title("OX llm prototype | Task 검색 / OX Decision Helper")

    # 탭 구성
    tabs = st.tabs(["Task Search & Visualization", "OX Decision Helper"])

    # 1) 탭: Search & Visualization
    with tabs[0]:
        st.subheader("LLM 기반 Task 검색 & 시각화")

        query = st.text_input("검색어를 입력하세요 (예: '마케팅 기획', '신메뉴 개발' 등)")
        if query:
            st.write(f"'{query}' 관련 유사 태스크 검색 중...")
            results = retrieve_tasks(query, k=5)

            # 파싱
            parsed_list = [parse_fields(r) for r in results]

            # 시각화
            visualize_tasks(parsed_list)

            # 요약
            if results:
                st.subheader("검색 결과 요약 (LLM)")
                summary_text = summarize_tasks(results,query)
                st.info(summary_text)

    # 2) 탭: OX Decision Helper
    with tabs[1]:
        st.subheader("LLM 기반 OX Decision Helper")

        new_task = st.text_area("새로운 Task 내용을 입력하세요 (O/X 판단 대상)")
        if new_task:
            st.write("유사 테스크를 참고하여 O/X 결정 중...")
            decision, relevant_tasks = generate_decision(new_task)

            # 결과 표시
            st.subheader("의사결정 결과 (LLM)")
            st.info(decision)

            # 어떤 유사 테스크가 참조되었는지
            with st.expander("참조된 유사 테스크 보기"):
                for i, rt in enumerate(relevant_tasks, start=1):
                    st.write(f"**유사테스크 {i}**\n{rt}\n")


if __name__ == "__main__":
    main()
