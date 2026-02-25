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
from app.models import WordEntry, JapaneseWordMetadata, WordLearningContent, SubtitleSentence, ExampleSentence, \
    ExampleTranslation, SubtitleTranslation
from app.schemas import AIWordEnhanceResponseV2, AIWordEnhanceResponseV4

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

PROMPT_V3 = """
너는 글로벌 일본어 교육 전문가이자 전문 번역가야. 
제공된 데이터를 바탕으로 다국어 학습 콘텐츠를 생성해줘.

[수행 과제]
1. Context Translation: 제공된 `contexts` 리스트의 모든 문장을 한국어로 정확하게 번역해.
2. Word Definition: 각 단어의 의미를 contexts 문맥에 맞게 풀이해.
3. Creative Example: `word_groups`의 단어들을 활용해 저작권 없는 고유한 새 문장을 창작하고 해석을 달아줘.

[절대 규칙]
- 원본 자막을 그대로 예문으로 쓰지 마라.
- 모든 `word_id`와 `context_id`를 정확히 매칭하여 리턴해라.

{format_instructions}

[입력 데이터]
{payload}
"""


class AIWordService:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=AIWordEnhanceResponseV4)
        self.prompt = ChatPromptTemplate.from_template(PROMPT_V3)
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
        # {simple_id: db_id} 역매핑용 딕셔너리
        id_mapping = {}
        context_counter = 1

        for word, sentence_id, sentence_text in chunk:
            if sentence_id not in unique_contexts:
                unique_contexts[sentence_id] = {"id": context_counter, "text": sentence_text}
                # 역매핑 정보 저장
                id_mapping[context_counter] = sentence_id
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

        return {
            "payload": {"contexts": list(unique_contexts.values()), "word_groups": word_groups},
            "context_mapping": id_mapping  # 이 정보가 저장 시 필요함
        }


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

    def _save_results(self, session: Session, chunk: List[tuple], ai_data: Any, context_mapping: Dict[int, int]):
        word_entry_map = {word.id: word for word, _, _ in chunk}
        success_count = 0

        # 1. 원본 자막 번역 저장 (SubtitleTranslation)
        for trans in ai_data.subtitle_translations:
            db_sentence_id = context_mapping.get(trans.context_id)
            if db_sentence_id:
                new_sub_trans = SubtitleTranslation(
                    subtitle_sentence_id=db_sentence_id,
                    language_code="ko",
                    translated_text=trans.translation_ko
                )
                session.add(new_sub_trans)

        # 2. 예문 데이터 처리 (ExampleSentence & Translation)
        created_example_ids = {}
        for ex in ai_data.examples:
            new_example = ExampleSentence(group_id=ex.group_id, sentence_text=ex.new_sentence_ja)
            session.add(new_example)
            session.flush()  # ID 확보

            created_example_ids[ex.group_id] = new_example.id
            session.add(ExampleTranslation(
                example_id=new_example.id,
                language_code="ko",
                translated_text=ex.new_sentence_ko
            ))

        # 3. 단어별 메타데이터 및 학습 콘텐츠 매핑
        for def_res in ai_data.word_definitions:
            word_entry = word_entry_map.get(def_res.word_id)
            if not word_entry: continue

            word_entry.japanese_metadata.reading = def_res.reading
            word_entry.japanese_metadata.jlpt_level = def_res.jlpt_level

            target_group_id = next((ex.group_id for ex in ai_data.examples if def_res.word_id in ex.word_ids), None)
            if target_group_id in created_example_ids:
                session.add(WordLearningContent(
                    word_entry_id=word_entry.id,
                    example_id=created_example_ids[target_group_id],
                    meaning=def_res.meaning,
                    usage_tip=f"JLPT {def_res.jlpt_level} 수준",
                    language_code="ko"
                ))
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