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

# 1. Vectorise the tasks.csv data
loader = CSVLoader(file_path="tasks.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# Regex parsing function
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

# 2. Function for similarity search
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=7)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-4o-2024-08-06")

template = """
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

Return only the final decision (O or X), the probability, and your short reasoning.

Example format:
Decision: O (85%) 
Reasoning: This task is very similar to past campaign tasks that received O due to well-defined goals...
Now, please produce your final O or X decision based on the new task and the historical records,
answer in KOREAN:

"""

prompt = PromptTemplate(
    input_variables=["message", "relevant_tasks"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# 4. Retrieval augmented generation
def generate_response(message):
    relevant_tasks = retrieve_info(message)
    response = chain.run(message=message, relevant_tasks=relevant_tasks)
    return response, relevant_tasks

# 5. Build an app with streamlit
def main():
    st.set_page_config(page_title="OX Decision Helper", page_icon=":bird:")
    st.header("OX Decision Helper :bird:")

    message = st.text_area("Paste the new task details here...")

    if message:
        st.write("Generating O/X decision based on historical tasks...")
        decision, relevant_tasks = generate_response(message)

        # 결과(LLM 응답) 표시
        st.info(decision)

        # 아래부터 relevant_tasks(유사도 높은 7개)를 표 형태로 노출
        st.subheader("관련 이력 Tasks (유사도 높은 7개)")

        # 각각의 doc_text를 parse_fields() 함수로 Dictionary로 바꾸고,
        # rows 배열에 담은 뒤 DataFrame으로 만들기
        rows = []
        for doc_text in relevant_tasks:
            row_dict = parse_fields(doc_text)
            rows.append(row_dict)

        # rows로부터 데이터프레임 생성
        df = pd.DataFrame(rows)

        # Streamlit의 표 출력
        st.table(df)  # 또는 st.dataframe(df)

if __name__ == '__main__':
    main()
