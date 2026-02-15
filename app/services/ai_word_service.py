import asyncio
import logging
import time
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from sqlmodel import Session, select
from app.models import WordEntry, JapaneseWordMetadata, WordLearningContent, SubtitleSentence
from app.schemas import WordEnhanceResponse

load_dotenv()

# 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AIWordService:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=WordEnhanceResponse)
        self.semaphore = asyncio.Semaphore(3)  # 동시 실행 제한

    def _chunk_list(self, lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    async def _process_batch(self, session: Session, chunk: List[tuple], batch_idx: int, total_batches: int):
        """개별 배치를 처리하고 진행 현황을 로깅함"""
        async with self.semaphore:
            batch_start_time = time.time()
            context_list = [{"word": we.base_form, "context": text} for we, text in chunk]
            chunk_size = len(chunk)

            logger.info(f"🚀 [Batch {batch_idx}/{total_batches}] 단어 {chunk_size}개 처리 시작...")

            prompt = ChatPromptTemplate.from_template(
                "너는 일본어 교육 전문가야. 아래 단어 리스트와 각 문맥을 바탕으로 정보를 추출해줘.\n"
                "규칙: 모든 읽기는 히라가나로만, JLPT 등급은 N1~N5로.\n\n"
                "{format_instructions}\n"
                "데이터 리스트: {context_list}"
            )

            input_data = prompt.format_messages(
                context_list=context_list,
                format_instructions=self.parser.get_format_instructions()
            )

            try:
                response = await self.llm.ainvoke(input_data)
                batch_result = self.parser.parse(response.content)

                # DB 반영 로직
                result_map = {item.base_form: item for item in batch_result.words}
                success_count = 0

                for we, _ in chunk:
                    if we.base_form in result_map:
                        data = result_map[we.base_form]
                        we.japanese_metadata.reading = data.reading
                        we.japanese_metadata.jlpt_level = data.jlpt_level

                        content = WordLearningContent(
                            word_entry_id=we.id,
                            meaning=data.meaning,
                            usage_tip=f"JLPT {data.jlpt_level} 수준",
                            generated_example={"ja": data.usage_example, "ko": data.usage_meaning}
                        )
                        session.add(content)
                        success_count += 1

                duration = time.time() - batch_start_time
                logger.info(
                    f"✅ [Batch {batch_idx}/{total_batches}] 완료 ({success_count}/{chunk_size} 단어) - 소요시간: {duration:.2f}s")
                return success_count

            except Exception as e:
                logger.error(f"❌ [Batch {batch_idx}/{total_batches}] 에러 발생: {str(e)}")
                return 0

    async def enhance_words_hybrid(self, session: Session, subtitle_id: int, batch_size: int = 15):
        start_time = time.time()

        # 1. 대상 단어 조회
        statement = (
            select(WordEntry, SubtitleSentence.sentence_text)
            .join(JapaneseWordMetadata)
            .join(SubtitleSentence, WordEntry.first_occurrence_id == SubtitleSentence.id)
            .where(WordEntry.subtitle_id == subtitle_id)
            .where(WordEntry.is_valid == True)
            .where(JapaneseWordMetadata.jlpt_level == "WAIT")
        )
        results = session.exec(statement).all()
        total_words = len(results)

        if total_words == 0:
            logger.info(f"ℹ️ 자막 ID {subtitle_id}: 강화할 단어가 없습니다.")
            return 0

        # 2. 청킹 및 배치 준비
        chunks = list(self._chunk_list(results, batch_size))
        total_batches = len(chunks)

        logger.info(f"🔥 자막 ID {subtitle_id}: 총 {total_words}개 단어 강화 시작 (총 {total_batches}개 배치)")

        # 3. 병렬 처리 실행
        tasks = [
            self._process_batch(session, chunk, i + 1, total_batches)
            for i, chunk in enumerate(chunks)
        ]

        success_counts = await asyncio.gather(*tasks)

        # 4. 최종 저장 및 로그
        session.commit()

        total_success = sum(success_counts)
        total_duration = time.time() - start_time
        logger.info(f"🎉 전체 작업 완료: {total_success}/{total_words} 단어 성공 - 총 소요시간: {total_duration:.2f}s")

        return total_success