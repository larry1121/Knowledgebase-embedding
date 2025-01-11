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

###########################
# (A) 기존: tasks.csv 로드, FAISS 등
###########################
loader = CSVLoader(file_path="tasks.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# (기존 parse_fields, retrieve_info 등 함수도 여기 있을 것으로 가정)
# ... (생략)
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

###########################
# (A) 기존: tasks.csv 로드, FAISS 등
###########################
loader = CSVLoader(file_path="tasks.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# (기존 parse_fields, retrieve_info 등 함수도 여기 있을 것으로 가정)
# ... (생략)

###########################
# (B) 시각화와 LLM 분석을 위한 함수들
###########################

def load_tasks_dataframe(csv_path: str = "tasks.csv") -> pd.DataFrame:
    """
    tasks.csv 파일을 DataFrame으로 로드하여 반환.
    """
    df = pd.read_csv(csv_path)
    return df

def preprocess_tasks(df: pd.DataFrame) -> pd.DataFrame:
    """
    차트/통계용으로 필요한 컬럼들을 전처리하거나,
    날짜형 변환, 공백 제거 등을 수행.
    """
    # 예: StartDate, DueDate를 실제 날짜 형으로 변환 시도
    if 'StartDate' in df.columns:
        df['StartDate'] = pd.to_datetime(df['StartDate'], errors='coerce')
    if 'DueDate' in df.columns:
        df['DueDate'] = pd.to_datetime(df['DueDate'], errors='coerce')
    return df

###########################
# (C) 간단한 통계 + 차트 예시
###########################

def create_dashboard_charts(df: pd.DataFrame):
    """
    주어진 df를 바탕으로,
    1) O/X 분포 차트
    2) Space(마케팅, 개발 등)별 O/X 현황
    3) 월별(또는 주차별) O/X 추이
    등 몇 가지 Altair 차트를 만들어 Streamlit에 표시
    """

    st.subheader("시각화 대시보드 (Charts & Graphs)")

    # 1) O/X 분포 파이 차트
    if "ApprovalStatus" in df.columns:
        ox_counts = df["ApprovalStatus"].value_counts().reset_index()
        ox_counts.columns = ["ApprovalStatus", "Count"]

        pie_chart = alt.Chart(ox_counts).mark_arc().encode(
            theta="Count",
            color="ApprovalStatus",
            tooltip=["ApprovalStatus", "Count"]
        ).properties(
            title="O/X 분포"
        )
        st.altair_chart(pie_chart, use_container_width=True)

    # 2) Space별 O/X 분포 (Stacked Bar)
    if "Space" in df.columns and "ApprovalStatus" in df.columns:
        space_ox = df.groupby(["Space", "ApprovalStatus"]).size().reset_index(name="Count")

        bar_chart = alt.Chart(space_ox).mark_bar().encode(
            x="Space:N",
            y="Count:Q",
            color="ApprovalStatus:N",
            tooltip=["Space", "ApprovalStatus", "Count"]
        ).properties(
            title="스페이스별 O/X 현황"
        )
        st.altair_chart(bar_chart, use_container_width=True)

    # 3) 월별 O/X 추이 (예: StartDate 기준)
    #    StartDate, ApprovalStatus 가 존재할 때
    if "StartDate" in df.columns and "ApprovalStatus" in df.columns:
        df["YearMonth"] = df["StartDate"].dt.to_period("M").astype(str)  # YYYY-MM 형태
        monthly_ox = df.groupby(["YearMonth", "ApprovalStatus"]).size().reset_index(name="Count")

        line_chart = alt.Chart(monthly_ox).mark_line(point=True).encode(
            x="YearMonth:N",
            y="Count:Q",
            color="ApprovalStatus:N",
            tooltip=["YearMonth", "ApprovalStatus", "Count"]
        ).properties(
            title="월별 O/X 추이"
        )
        st.altair_chart(line_chart, use_container_width=True)

###########################
# (D) LLM 분석 프롬프트
###########################
analysis_template = """
You are a data analyst. I have some summarized statistics or a description of charts 
about an OX-based task system.

Below is the data or summary:
{stats_summary}

Please analyze it in KOREAN, 
focus on any interesting trends or anomalies you see (e.g. certain month has higher X, 
some space has lower O, etc.). 
Return 2~3 sentences of insights.
"""

analysis_prompt = PromptTemplate(
    input_variables=["stats_summary"],
    template=analysis_template
)

analysis_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

analysis_chain = LLMChain(llm=analysis_llm, prompt=analysis_prompt)


def generate_analysis_comment(stats_summary: str) -> str:
    """
    통계나 차트 데이터를 요약해 LLM에 넘겨, 추가 인사이트를 2~3줄로 받아옴.
    """
    comment = analysis_chain.run(stats_summary=stats_summary)
    return comment


###########################
# (E) Streamlit App
###########################
def main():
    st.set_page_config(page_title="OX Decision Dashboard", page_icon=":bar_chart:")
    st.header("OX Decision Helper with Dashboard :bar_chart:")

    # 1) 기본 텍스트 입력 (의사결정용)
    message = st.text_area("Paste the new task details here... (optional)")

    # (기존) 만약 message가 있으면, OX 결정/ Summaries / Compare 등 실행
    # 여기서는 예시이므로, 대시보드에 집중
    # ------------------------------------------------
    
    # 2) 시각화 대시보드
    st.subheader("대시보드 데이터 로드")
    #   tasks.csv를 로드 & 전처리
    df_tasks = load_tasks_dataframe("tasks.csv")
    df_tasks = preprocess_tasks(df_tasks)

    if not df_tasks.empty:
        # 차트 표시
        create_dashboard_charts(df_tasks)

        # 3) LLM 분석 코멘트
        #    차트에 쓰인 통계를 간단히 텍스트화하거나,
        #    여기서는 예시로 "Space별 승인/거절 수"를 요약
        if "Space" in df_tasks.columns and "ApprovalStatus" in df_tasks.columns:
            # 예: space별 O/X count
            space_approval = df_tasks.groupby(["Space", "ApprovalStatus"]).size().reset_index(name="Count")
            # dataframe -> string
            stats_str = space_approval.to_csv(index=False)
        else:
            stats_str = "No Space or ApprovalStatus data available."

        analysis_comment = generate_analysis_comment(stats_str)
        
        st.subheader("데이터 분석 인사이트 (LLM)")
        st.info(analysis_comment)
    else:
        st.warning("tasks.csv 가 비어있거나 데이터를 로드할 수 없습니다.")


if __name__ == '__main__':
    main()

###########################
# (B) 시각화와 LLM 분석을 위한 함수들
###########################

def load_tasks_dataframe(csv_path: str = "tasks.csv") -> pd.DataFrame:
    """
    tasks.csv 파일을 DataFrame으로 로드하여 반환.
    """
    df = pd.read_csv(csv_path)
    return df

def preprocess_tasks(df: pd.DataFrame) -> pd.DataFrame:
    """
    차트/통계용으로 필요한 컬럼들을 전처리하거나,
    날짜형 변환, 공백 제거 등을 수행.
    """
    # 예: StartDate, DueDate를 실제 날짜 형으로 변환 시도
    if 'StartDate' in df.columns:
        df['StartDate'] = pd.to_datetime(df['StartDate'], errors='coerce')
    if 'DueDate' in df.columns:
        df['DueDate'] = pd.to_datetime(df['DueDate'], errors='coerce')
    return df

###########################
# (C) 간단한 통계 + 차트 예시
###########################

def create_dashboard_charts(df: pd.DataFrame):
    """
    주어진 df를 바탕으로,
    1) O/X 분포 차트
    2) Space(마케팅, 개발 등)별 O/X 현황
    3) 월별(또는 주차별) O/X 추이
    등 몇 가지 Altair 차트를 만들어 Streamlit에 표시
    """

    st.subheader("시각화 대시보드 (Charts & Graphs)")

    # 1) O/X 분포 파이 차트
    if "ApprovalStatus" in df.columns:
        ox_counts = df["ApprovalStatus"].value_counts().reset_index()
        ox_counts.columns = ["ApprovalStatus", "Count"]

        pie_chart = alt.Chart(ox_counts).mark_arc().encode(
            theta="Count",
            color="ApprovalStatus",
            tooltip=["ApprovalStatus", "Count"]
        ).properties(
            title="O/X 분포"
        )
        st.altair_chart(pie_chart, use_container_width=True)

    # 2) Space별 O/X 분포 (Stacked Bar)
    if "Space" in df.columns and "ApprovalStatus" in df.columns:
        space_ox = df.groupby(["Space", "ApprovalStatus"]).size().reset_index(name="Count")

        bar_chart = alt.Chart(space_ox).mark_bar().encode(
            x="Space:N",
            y="Count:Q",
            color="ApprovalStatus:N",
            tooltip=["Space", "ApprovalStatus", "Count"]
        ).properties(
            title="스페이스별 O/X 현황"
        )
        st.altair_chart(bar_chart, use_container_width=True)

    # 3) 월별 O/X 추이 (예: StartDate 기준)
    #    StartDate, ApprovalStatus 가 존재할 때
    if "StartDate" in df.columns and "ApprovalStatus" in df.columns:
        df["YearMonth"] = df["StartDate"].dt.to_period("M").astype(str)  # YYYY-MM 형태
        monthly_ox = df.groupby(["YearMonth", "ApprovalStatus"]).size().reset_index(name="Count")

        line_chart = alt.Chart(monthly_ox).mark_line(point=True).encode(
            x="YearMonth:N",
            y="Count:Q",
            color="ApprovalStatus:N",
            tooltip=["YearMonth", "ApprovalStatus", "Count"]
        ).properties(
            title="월별 O/X 추이"
        )
        st.altair_chart(line_chart, use_container_width=True)

###########################
# (D) LLM 분석 프롬프트
###########################
analysis_template = """
You are a data analyst. I have some summarized statistics or a description of charts 
about an OX-based task system.

Below is the data or summary:
{stats_summary}

Please analyze it in KOREAN, 
focus on any interesting trends or anomalies you see (e.g. certain month has higher X, 
some space has lower O, etc.). 
Return 2~3 sentences of insights.
"""

analysis_prompt = PromptTemplate(
    input_variables=["stats_summary"],
    template=analysis_template
)

analysis_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

analysis_chain = LLMChain(llm=analysis_llm, prompt=analysis_prompt)


def generate_analysis_comment(stats_summary: str) -> str:
    """
    통계나 차트 데이터를 요약해 LLM에 넘겨, 추가 인사이트를 2~3줄로 받아옴.
    """
    comment = analysis_chain.run(stats_summary=stats_summary)
    return comment


###########################
# (E) Streamlit App
###########################
def main():
    st.set_page_config(page_title="OX Decision Dashboard", page_icon=":bar_chart:")
    st.header("OX Decision Helper with Dashboard :bar_chart:")

    # 1) 기본 텍스트 입력 (의사결정용)
    message = st.text_area("Paste the new task details here... (optional)")

    # (기존) 만약 message가 있으면, OX 결정/ Summaries / Compare 등 실행
    # 여기서는 예시이므로, 대시보드에 집중
    # ------------------------------------------------
    
    # 2) 시각화 대시보드
    st.subheader("대시보드 데이터 로드")
    #   tasks.csv를 로드 & 전처리
    df_tasks = load_tasks_dataframe("tasks.csv")
    df_tasks = preprocess_tasks(df_tasks)

    if not df_tasks.empty:
        # 차트 표시
        create_dashboard_charts(df_tasks)

        # 3) LLM 분석 코멘트
        #    차트에 쓰인 통계를 간단히 텍스트화하거나,
        #    여기서는 예시로 "Space별 승인/거절 수"를 요약
        if "Space" in df_tasks.columns and "ApprovalStatus" in df_tasks.columns:
            # 예: space별 O/X count
            space_approval = df_tasks.groupby(["Space", "ApprovalStatus"]).size().reset_index(name="Count")
            # dataframe -> string
            stats_str = space_approval.to_csv(index=False)
        else:
            stats_str = "No Space or ApprovalStatus data available."

        analysis_comment = generate_analysis_comment(stats_str)
        
        st.subheader("데이터 분석 인사이트 (LLM)")
        st.info(analysis_comment)
    else:
        st.warning("tasks.csv 가 비어있거나 데이터를 로드할 수 없습니다.")


if __name__ == '__main__':
    main()
