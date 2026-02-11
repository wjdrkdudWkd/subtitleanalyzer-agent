from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    """
        이 에이전트의 상태를 관리하는 객체입니다.
        노드들 사이를 흘러다니며 데이터를 전달합니다.
    """
    subtitle_raw: str  # 입력받은 자막 텍스트
    selected_words: List[str]  # AI가 추출한 핵심 단어 리스트
    word_entries: List[dict]  # 최종 생성된 단어 데이터 (단어, 뜻, 예문)
    retry_count: int  # 검증 실패 시 재시도 횟수
    error: Optional[str]  # 에러 메시지