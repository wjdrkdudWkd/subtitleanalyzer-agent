import asyncio
import logging
import time
import json
from collections import defaultdict
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from sqlmodel import Session, select
from app.models import WordEntry, JapaneseWordMetadata, WordLearningContent, SubtitleSentence
from app.schemas import AIWordEnhanceResponseV2

load_dotenv()

# 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROMPT_V2 = """
너는 일본어 교육 전문가이자 창의적인 작가야. 제공된 데이터를 바탕으로 학습 콘텐츠를 만들어줘.

[수행 과제]
1. contexts를 참조하여 각 단어의 정확한 의미와 읽기를 파악해.
2. 각 word_groups에 대해, 해당 그룹의 단어 3개를 모두 포함한 '새로운' 예문을 하나씩 창조해.

[절대 규칙]
- 제공된 contexts의 문장을 그대로 리턴하거나 복사하지 마라. (저작권 준수)
- 반드시 자막 상황과 다른 고유하고 자연스러운 문장을 만들어라.
- 모든 읽기는 히라가나로만 작성해라.

{format_instructions}

[입력 데이터]
{payload}
"""


class AIWordService:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=AIWordEnhanceResponseV2)
        self.prompt = ChatPromptTemplate.from_template(PROMPT_V2)
        self.semaphore = asyncio.Semaphore(3)  # 동시 실행 제한

    def _chunk_list(self, lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def _prepare_payload(self, results: List[tuple]) -> Dict[str, Any]:
        """
        DB 조회 결과(WordEntry, SubtitleSentence)를 AI 전달용 구조로 매핑
        문장 간 인터리빙 전략으로 단어 그룹화
        """
        # 1. 문장 중복 제거 및 단어 분류 (Sentence-based Grouping)
        # 유니크한 문장(Context) 추출 및 짧은 ID 부여
        unique_contexts = {}  # {db_sentence_id: {"id": simple_id, "text": text}}
        sentence_buckets = defaultdict(list)  # {sentence_id: [word_info, ...]}
        context_counter = 1

        # 2. 단어별로 컨텍스트 참조 정보 정리
        for word, sentence in results:
            if sentence.id not in unique_contexts:
                unique_contexts[sentence.id] = {
                    "id": context_counter,
                    "text": sentence.sentence_text
                }
                context_counter += 1

            # 문장 ID별로 단어를 담아둡니다 (바구니 채우기)
            sentence_buckets[sentence.id].append({
                "base_form": word.base_form,
                "context_id": unique_contexts[sentence.id]["id"]
            })

        # 2. 인터리빙(Interleaving) 전략: 각 바구니에서 하나씩 골고루 뽑기
        interleaved_words = []
        buckets = list(sentence_buckets.values())

        # 모든 바구니가 빌 때까지 돌아가며 하나씩 추출
        max_len = max(len(b) for b in buckets)
        for i in range(max_len):
            for bucket in buckets:
                if i < len(bucket):
                    interleaved_words.append(bucket[i])

        # 3. 섞인 리스트를 3개씩 소그룹화
        group_size = 3
        word_groups = []
        for i in range(0, len(interleaved_words), group_size):
            group_id = (i // group_size) + 1
            word_groups.append({
                "group_id": group_id,
                "words": interleaved_words[i: i + group_size]
            })

        return {
            "contexts": list(unique_contexts.values()),
            "word_groups": word_groups
        }

    async def _process_batch(self, session: Session, chunk: List[tuple], batch_idx: int, total_batches: int):
        """개별 배치를 처리하고 진행 현황을 로깅함"""
        async with self.semaphore:
            batch_start_time = time.time()
            logger.info(f"🚀 [Batch {batch_idx}/{total_batches}] v0.0.2 처리 시작...")

            # 1. 데이터 매핑 (인터리빙)
            payload = self._create_interleaved_payload(chunk)

            # 2. AI 호출
            input_data = self.prompt.format_messages(
                payload=json.dumps(payload, ensure_ascii=False),
                format_instructions=self.parser.get_format_instructions()
            )

            try:
                response = await self.llm.ainvoke(input_data)
                ai_data = self.parser.parse(response.content)

                # 3. DB 저장
                success_count = self._save_results(session, chunk, ai_data)

                duration = time.time() - batch_start_time
                logger.info(f"✅ [Batch {batch_idx}/{total_batches}] 완료 ({success_count}개) - {duration:.2f}s")
                return success_count
            except Exception as e:
                logger.error(f"❌ [Batch {batch_idx}/{total_batches}] 에러: {str(e)}")
                return 0

    def _create_interleaved_payload(self, chunk: List[tuple]) -> Dict[str, Any]:
        """chunk 내부의 단어들을 문장별로 섞어 그룹화"""
        unique_contexts = {}
        sentence_buckets = defaultdict(list)
        context_counter = 1

        for word, sentence_id, sentence_text in chunk:
            if sentence_id not in unique_contexts:
                unique_contexts[sentence_id] = {"id": context_counter, "text": sentence_text}
                context_counter += 1

            sentence_buckets[sentence_id].append({
                "word_id": word.id,
                "base_form": word.base_form,
                "context_id": unique_contexts[sentence_id]["id"]
            })

        interleaved = []
        buckets = list(sentence_buckets.values())
        max_len = max(len(b) for b in buckets)
        for i in range(max_len):
            for bucket in buckets:
                if i < len(bucket): interleaved.append(bucket[i])

        word_groups = []
        for i in range(0, len(interleaved), 3):
            word_groups.append({
                "group_id": (i // 3) + 1,
                "words": interleaved[i: i + 3]
            })

        return {"contexts": list(unique_contexts.values()), "word_groups": word_groups}


    async def enhance_words_v2(self, session: Session, results: List[tuple]):
        # 1. 데이터 매핑 (아까 만든 로직)
        payload = self._prepare_payload(results)

        # 2. 프롬프트 주입
        input_data = self.prompt_template.format_messages(
            payload=json.dumps(payload, ensure_ascii=False),
            format_instructions=self.parser.get_format_instructions()
        )

        # 3. AI 호출
        try:
            response = await self.llm.ainvoke(input_data)
            ai_data = self.parser.parse(response.content)

            # 4. DB 저장 (이 부분이 다음 고비!)
            # ai_data.word_definitions -> JapaneseWordMetadata 업데이트
            # ai_data.examples -> WordLearningContent 생성 및 저장
            return ai_data
        except Exception as e:
            logger.error(f"AI 처리 중 오류 발생: {e}")
            return None

    def _save_results(self, session: Session, chunk: List[tuple], ai_data: AIWordEnhanceResponseV2):
        word_entry_map = {word.id: word for word, _, _ in chunk}

        # 1. 현재 배치에 포함된 WordEntry 객체들을 ID 기반 Map으로 변환
        word_entry_map = {word.id: word for word, _, _ in chunk}

        # 2. AI가 응답한 예문들을 word_id 기반으로 역매핑 (중요!)
        # 어떤 word_id가 어떤 예문을 가져야 하는지 Map 생성
        # {word_id: GroupedExample}
        word_to_example_map = {}
        for ex in ai_data.examples:
            for w_id in ex.word_ids:
                word_to_example_map[w_id] = ex

        success_count = 0

        # 3. AI 응답 데이터 순회 및 DB 반영
        for def_res in ai_data.word_definitions:
            word_id = def_res.word_id
            word_entry = word_entry_map.get(word_id)

            if not word_entry:
                logger.warning(f"⚠️ AI가 보낸 word_id {word_id}를 현재 배치에서 찾을 수 없습니다.")
                continue

            # [A] 메타데이터 업데이트 (JapaneseWordMetadata)
            word_entry.japanese_metadata.reading = def_res.reading
            word_entry.japanese_metadata.jlpt_level = def_res.jlpt_level

            # [B] 학습 콘텐츠 저장 (WordLearningContent)
            # 역매핑된 맵에서 해당 단어의 예문을 찾아옴
            matching_ex = word_to_example_map.get(word_id)

            if matching_ex:
                content = WordLearningContent(
                    word_entry_id=word_entry.id,
                    meaning=def_res.meaning,
                    usage_tip=f"JLPT {def_res.jlpt_level} 수준",
                    generated_example={
                        "ja": matching_ex.new_sentence_ja,
                        "ko": matching_ex.new_sentence_ko
                    }
                )
                session.add(content)
                success_count += 1

        return success_count

    async def enhance_words_hybrid(self, session: Session, subtitle_id: int, batch_size: int = 15):
        start_time = time.time()
        # 쿼리 수정: sentence_id를 함께 가져와서 인터리빙에 활용
        statement = (
            select(WordEntry, SubtitleSentence.id, SubtitleSentence.sentence_text)
            .join(JapaneseWordMetadata)
            .join(SubtitleSentence, WordEntry.first_occurrence_id == SubtitleSentence.id)
            .where(WordEntry.subtitle_id == subtitle_id, WordEntry.is_valid == True,
                   JapaneseWordMetadata.jlpt_level == "WAIT")
        )
        results = session.exec(statement).all()
        if not results: return 0

        chunks = list(self._chunk_list(results, batch_size))
        tasks = [self._process_batch(session, chunk, i + 1, len(chunks)) for i, chunk in enumerate(chunks)]

        success_counts = await asyncio.gather(*tasks)
        session.commit()

        logger.info(f"🎉 작업 완료: {sum(success_counts)}개 단어 성공 ({time.time() - start_time:.2f}s)")
        return sum(success_counts)