import streamlit as st
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="tasks.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    print(page_contents_array)

    return page_contents_array


# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-4o-2024-08-06")

template = """
You are a highly knowledgeable Task Reviewer in the OX Human Resource Platform.

Below is a new task that needs an O or X decision:
{message}

We also have a historical record of tasks that were decided O or X, which might be relevant to this new task:
{best_practice}

Please follow these instructions:

1. Carefully analyze the new task details, including all context such as task name, assignees, priorities, and evaluation criteria.
2. Compare the new task to any relevant historical tasks found in the database (represented by best_practice). Identify patterns or precedents.
3. Determine whether this new task is more likely to be approved (O) or rejected (X).
4. Provide a numerical probability of how likely it is to be approved based on the historical data (for example, “There is a 75% chance this task will receive O”).
5. Give a concise explanation/rationale for your decision, referencing any similar tasks from the historical records.

Return only the final decision (O or X), the probability, and your short reasoning.

Example format:
Decision: O Probability: 85% Reasoning: "This task is very similar to past campaign tasks that received O due to well-defined goals..."
Now, please produce your final O or X decision based on the new task and the historical records, answer in KOREAN:

"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval augmented generation
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response


# 5. Build an app with streamlit
def main():
    st.set_page_config(page_title="OX Decision Generator", page_icon=":bird:")
    st.header("OX Decision Generator :bird:")

    message = st.text_area("Paste the new task details here...")

    if message:
        st.write("Generating O/X decision based on historical tasks...")
        result = generate_response(message)
        st.info(result)


if __name__ == '__main__':
    main()
