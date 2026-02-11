# app/agent/graph.py
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import extract_words, generate_content, validate_result

def create_graph():
    # 1. 그래프 초기화
    workflow = StateGraph(AgentState)

    # 2. 노드 등록
    workflow.add_node("extract", extract_words)
    workflow.add_node("generate", generate_content)
    workflow.add_node("validate", validate_result)

    # 3. 경로 설정
    workflow.set_entry_point("extract")
    workflow.add_edge("extract", "generate")
    workflow.add_edge("generate", "validate")

    # 4. 조건부 분기 (이게 에이전트의 핵심!)
    def decide_next_node(state: AgentState):
        if state["error"] is not None and state["retry_count"] < 3:
            print(f"---검증 실패 (재시도 {state['retry_count']}회)---")
            return "retry"
        return "end"

    workflow.add_conditional_edges(
        "validate",
        decide_next_node,
        {
            "retry": "generate", # 실패 시 생성 단계로 루프
            "end": END           # 성공 시 종료
        }
    )

    return workflow.compile()