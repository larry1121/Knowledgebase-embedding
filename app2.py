import streamlit as st
import pandas as pd
import re

from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# 1. CSV 불러와서 FAISS VectorStore 구성
loader = CSVLoader(file_path="tasks.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)


##########################
# (A) 유사도 검색 함수
##########################
def retrieve_info(query, k=7):
    """
    사용자가 입력한 query(새로운 Task)에 대해
    tasks.csv에 저장된 과거 Task 중 가장 유사한 k개를 검색하여 반환.
    """
    similar_response = db.similarity_search(query, k=k)
    # page_content만 추출
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array


##########################
# (B) 문서(텍스트) 파싱용 정규식 함수
##########################
def parse_fields(text: str) -> dict:
    """
    "TaskID: T032 TaskName: ... ParentTaskName: ..." 식으로 이어진 문자열에서
    주요 필드 값을 추출해 dict로 반환.
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
            # 불필요한 공백 제거
            parsed[field] = match.group(1).strip()
        else:
            parsed[field] = ""
    return parsed


##########################
# (C) O/X 의사결정 프롬프트
##########################
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

##########################
# (D) 자동 요약(Auto Summaries) 프롬프트
##########################
summaries_template = """
You are an assistant that provides short summaries of tasks.

Below is a list of relevant tasks from the past:
{relevant_tasks}

Please provide a concise bullet-point summary in KOREAN,
highlighting the essential points from these tasks (common goals, issues, or notable outcomes).

Output format example:
- 요약 1...
- 요약 2...
- ...
"""

summaries_prompt = PromptTemplate(
    input_variables=["relevant_tasks"],
    template=summaries_template
)

##########################
# (E) 비교 분석(Comparative Analysis) 프롬프트
##########################
compare_template = """
You are an assistant that compares tasks from the past and highlights any key differences or similarities.

Below is a table of tasks in CSV-like format:
{task_table}

Please provide a short commentary in KOREAN that compares these tasks:
- Focus on their ApprovalStatus, Priority, or any unique points.
- Mention any patterns (e.g., high priority tasks often received O, etc.).

Return a short analysis (1~3 sentences).
"""

compare_prompt = PromptTemplate(
    input_variables=["task_table"],
    template=compare_template
)


# 모델 세팅
llm = ChatOpenAI(temperature=0, model="gpt-4o-2024-08-06")

# 체인 구성
decision_chain = LLMChain(llm=llm, prompt=decision_prompt)
summaries_chain = LLMChain(llm=llm, prompt=summaries_prompt)
compare_chain = LLMChain(llm=llm, prompt=compare_prompt)


##########################
# (F) 한 번의 질문 -> 세 가지 결과 (O/X 결정, 요약, 비교분석)
##########################
def generate_full_response(message):
    # 1. 유사 테스크 검색
    relevant_tasks = retrieve_info(message, k=7)

    # 2. OX 결정 체인
    #  - relevant_tasks(리스트[str])를 적당한 형식으로 합쳐서 LLM에 전달
    joined_tasks_for_decision = "\n\n".join(relevant_tasks)
    decision_result = decision_chain.run(
        message=message,
        relevant_tasks=joined_tasks_for_decision
    )

    # 3. 자동 요약(Auto Summaries)
    #  - relevant_tasks를 SummariesChain에 주어 요약
    joined_tasks_for_summaries = "\n\n".join(relevant_tasks)
    summary_result = summaries_chain.run(
        relevant_tasks=joined_tasks_for_summaries
    )

    # 4. 비교 분석(Comparative Analysis)
    #  (a) 유사 테스크를 parse_fields로 dict화 -> DataFrame -> CSV-like string
    rows = []
    for doc_text in relevant_tasks:
        row_dict = parse_fields(doc_text)
        rows.append(row_dict)
    df = pd.DataFrame(rows)
    # (b) df를 CSV-like string으로 만든 뒤 compare_chain에 전달
    #     (간단히 df.to_csv(index=False) 사용)
    table_csv_str = df.to_csv(index=False)
    compare_result = compare_chain.run(
        task_table=table_csv_str
    )

    return decision_result, summary_result, df, compare_result


##########################
# (G) Streamlit UI
##########################
def main():
    st.set_page_config(page_title="OX Decision + Summaries + Compare", page_icon=":bar_chart:")
    st.header("OX Decision Helper (with Auto Summaries & Compare) :bar_chart:")

    message = st.text_area("Paste the new task details here...")

    if message:
        st.write("Generating results based on historical tasks...")

        decision_res, summary_res, compare_df, compare_comment = generate_full_response(message)

        # 1) OX 결정 결과
        st.subheader("1. O/X 결정 결과")
        st.info(decision_res)

        # 2) 자동 요약
        st.subheader("2. 자동 요약(Auto Summaries)")
        st.write(summary_res)

        # 3) 비교 분석
        st.subheader("3. 비교 분석(Comparative Analysis)")
        st.write("아래는 유사도가 높은 7개 Task의 데이터입니다:")
        st.dataframe(compare_df)

        # LLM의 비교 코멘트
        st.write("LLM 비교 코멘트:")
        st.info(compare_comment)


if __name__ == '__main__':
    main()
