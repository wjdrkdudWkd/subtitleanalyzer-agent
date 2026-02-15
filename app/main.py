import asyncio

import load_dotenv
from dotenv import load_dotenv
from app.agent.graph import create_graph

load_dotenv()

async def main():
    # 1. 그래프(에이전트) 생성
    app = create_graph()

    # 2. 테스트용 입력 데이터 (자막의 일부)
    initial_state = {
        "subtitle_raw": (
            "すごいわ　カルシファー！ あなたは一流よ！"
            "心臓！　心臓があるのかい？"
        ),
        "retry_count": 0
    }

    # 3. 에이전트 실행
    print("🚀 자막 분석 에이전트 시작...")
    final_result = await app.ainvoke(initial_state)

    # 4. 결과 출력
    print("\n✅ 최종 추출된 단어:")
    print(final_result.get("selected_words"))

    print("\n📚 생성된 단어장:")
    for entry in final_result.get("word_entries", []):
        print(f"- {entry['word']}: {entry['meaning']}")
        print(f"  예문: {entry['example']}")


if __name__ == "__main__":
    # LangGraph의 비동기 실행을 위해 asyncio 사용
    asyncio.run(main())