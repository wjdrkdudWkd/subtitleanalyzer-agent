# app/agent/nodes.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
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