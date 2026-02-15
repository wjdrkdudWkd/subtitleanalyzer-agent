# app/agent/nodes.py
from pydantic import Field

from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from .state import AgentState

def extract_words(state: AgentState):
    """
    자막 원문에서 학습자가 공부하기 좋은 핵심 단어 5개를 추출합니다.
    """
    print("---단어 추출 중---")

    # 1. 모델 설정 (많이 쓰는 gpt-4o 추천)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 2. 프롬프트 설계 (이게 에이전트의 핵심 로직입니다)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 언어 학습 콘텐츠 제작자야. 제공된 자막에서 학습자가 익히면 좋은 중요한 단어 5개를 선정해줘."),
        ("user", "다음 자막에서 핵심 단어 5개만 뽑아서 쉼표로 구분해줘: {subtitle_raw}")
    ])

    # 3. 체인 생성 및 실행
    chain = prompt | llm
    response = chain.invoke({"subtitle_raw": state["subtitle_raw"]})

    # 4. 결과 파싱 (단순 쉼표 구분 -> 리스트)
    words = [w.strip() for w in response.content.split(",")]

    # 5. 상태 업데이트 (Java의 setter 느낌)
    return {"selected_words": words, "retry_count": 0}


# 1. AI 응답 형식을 강제하기 위한 스키마 (Java의 DTO와 유사)
class WordDetail(BaseModel):
    word: str = Field(description="단어")
    meaning: str = Field(description="자막 문맥에 맞는 한국어 뜻")
    example: str = Field(description="해당 단어를 사용한 새로운 예문")


def generate_content(state: AgentState):
    """
    추출된 단어들의 문맥상 의미를 파악하고 학습 콘텐츠를 생성합니다.
    """
    print("---단어장 콘텐츠 생성 중---")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)  # 생성 시에는 약간의 창의성 허용
    parser = JsonOutputParser(pydantic_object=WordDetail)

    # 2. 문맥 중심 프롬프트 설계
    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 전문 언어 튜터야. 주어진 자막 문맥을 바탕으로 단어의 뜻을 풀이해줘. 반드시 JSON 형식으로 답해야 해.\n{format_instructions}"),
        ("user", "자막 내용: {subtitle_raw}\n\n추출된 단어들: {selected_words}\n\n위 단어들에 대해 자막의 흐름에 맞는 뜻과 새로운 예문을 만들어줘.")
    ]).partial(format_instructions=parser.get_format_instructions())

    # 3. 체인 실행
    chain = prompt | llm | parser
    results = chain.invoke({
        "subtitle_raw": state["subtitle_raw"],
        "selected_words": state["selected_words"]
    })

    # 4. 상태 업데이트
    return {"word_entries": results if isinstance(results, list) else [results]}


def validate_result(state: AgentState):
    """
    생성된 단어장 콘텐츠의 품질을 검사합니다.
    """
    print("---품질 검증 중---")

    entries = state.get("word_entries", [])
    retry_count = state.get("retry_count", 0)

    # 1. 기본적인 유효성 체크
    if not entries:
        return {
            "error": "단어장이 생성되지 않았습니다.",
            "retry_count": retry_count + 1
        }

    # 2. 실무적인 검증 로직 (예: 특정 필드 누락 여부)
    for entry in entries:
        if not entry.get("meaning") or not entry.get("example"):
            return {
                "error": "필수 필드가 누락되었습니다.",
                "retry_count": retry_count + 1
            }

    # 3. 모든 검증 통과 시 error를 비움
    print("---검증 통과!---")
    return {"error": None, "retry_count": retry_count}
