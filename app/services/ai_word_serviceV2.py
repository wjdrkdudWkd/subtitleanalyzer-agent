import asyncio
import logging
import time
import json
from collections import defaultdict
from dataclasses import field
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic.dataclasses import dataclass
from sqlmodel import Session, select
from app.models import WordEntry, WordLearningContent, SubtitleSentence, ExampleSentence, \
    ExampleTranslation, SubtitleTranslation
from app.schemas import AIWordEnhanceResponseV4

load_dotenv()

# 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

PROMPT_V4 = """
당신은 일본어 교육 에이전트입니다. 제공된 `data`와 `groups`를 바탕으로 학습 콘텐츠를 생성하세요.

[수행 과제]
1. **Subtitle Translation**: `data`의 각 `text`를 문맥에 맞게 한국어로 번역하세요.
2. **Word Definition**: `data.words`의 각 단어를 분석하세요.
   - `mean`(뜻)은 부모 객체의 `text` 문맥을 최우선으로 반영합니다.
   - `lv`(JLPT)는 N1~N5 등급으로 판정하세요.
3. **Group Example Creation**: `groups`의 `wids`에 포함된 모든 단어를 사용하여 '하나의 새로운 일본어 문장'을 창작하세요.
   - **Constraint**: `groups.wids`는 `data.words.id`를 참조합니다. 반드시 해당 ID의 단어들을 모두 포함해야 합니다.

[주의 사항]
- **ID Integrity**: 입력받은 모든 `id`, `gid`는 결과 JSON에서 절대 변경하거나 누락하지 마세요. (매핑 정확도 100% 유지)
- **Efficiency**: 부연 설명이나 서론 없이, 지정된 JSON 포맷으로만 즉시 응답하세요.
- **Conciseness**: 예문 해석(`ko`)과 로직 설명(`logic`)은 학습자가 한눈에 읽기 좋게 짧고 간결하게 작성하세요.

{format_instructions}

[입력 데이터]
{payload}
"""

@dataclass(frozen=True)
class WordDTO:
    """단어 개별 정보"""
    word_id: int
    base_form: str

@dataclass(frozen=True)
class SentenceTaskDTO:
    """문장 단위 작업 그룹 (Orchestrator의 기본 단위)"""
    sentence_id: int
    sentence_text: str
    words: List[WordDTO] = field(default_factory=list)

@dataclass
class BatchProcessingResult:
    """한 배치의 작업 결과 보고서"""
    success_count: int
    batch_idx: int
    process_time: float


class AIWordService:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=AIWordEnhanceResponseV4)
        self.prompt = ChatPromptTemplate.from_template(PROMPT_V4)
        self.semaphore = asyncio.Semaphore(3)  # 동시 실행 제한

    def _fetch_analysis_targets(self, session: Session, subtitle_id: int) -> List[SentenceTaskDTO]:
        statement = (
            select(
                WordEntry,  # [0] WordEntry 객체 전체 (Entity)
                SubtitleSentence.id,  # [1] 문장 ID (Long/Integer)
                SubtitleSentence.sentence_text  # [2] 문장 내용 (String)
            )
            .join(
                SubtitleSentence,
                WordEntry.first_occurrence_id == SubtitleSentence.id
            )
            .where(
                WordEntry.subtitle_id == subtitle_id,
                WordEntry.is_valid == True,
                # 아직 AI 분석이 안 된 데이터만 필터링 (비즈니스 로직)
                # JapaneseWordMetadata 조인이 필요할 수 있음
            )
        )
        results = session.exec(statement).all()

        sentence_map = {}

        for word_obj, s_id, s_text in results:
            if s_id not in sentence_map:
                sentence_map[s_id] = SentenceTaskDTO(
                    sentence_id=s_id,
                    sentence_text=s_text,
                    words=[]
                )

            sentence_map[s_id].words.append(
                WordDTO(
                    word_id=word_obj.id,
                    base_form=word_obj.base_form,
                )
            )

        return list(sentence_map.values())



    async def enhance_words_hybrid(self, session: Session, subtitle_id: int, batch_size: int = 3):
        """
            [Orchestrator]
            계층형 데이터를 가져와 병렬로 분석을 실행하고 최종 결과를 반환합니다.
            batch_size = 문장 개수
        """
        start_time = time.time()

        # 1. [FETCH] 계층형 DTO 리스트 확보
        sentence_tasks = self._fetch_analysis_targets(session, subtitle_id)

        if not sentence_tasks:
            logger.info("✅ 분석할 새로운 단어가 없습니다.")
            return 0

        # 2. [CHUNKING] 문장 단위로 배치 분할
        # 단어 개수가 아니라 '문장(Context) 개수' 기준으로 쪼갭니다.
        chunks = [
            sentence_tasks[i: i + batch_size]
            for i in range(0, len(sentence_tasks), batch_size)
        ]

        total_batches = len(chunks)
        logger.info(f"🔥 분석 시작: 총 {len(sentence_tasks)}개 문장 그룹 / {total_batches}개 배치")

        # 3. [PARALLEL EXECUTION] asyncio.gather로 병렬 처리
        # 각 배치는 독립적인 비동기 태스크로 실행됩니다.
        batch_tasks = [
            self._process_batch_v3(session, chunk, i + 1, total_batches)
            for i, chunk in enumerate(chunks)
        ]

        # 자바의 join()처럼 모든 태스크가 끝날 때까지 비동기 대기합니다.
        # 결과값으로 각 배치의 성공 단어 개수 리스트가 돌아옵니다.
        # success_counts = await asyncio.gather(*batch_tasks)

        # 끝나는 순서대로 하나씩 반환받음 (as_completed)
        success_counts = []
        completed_count = 0
        for task in asyncio.as_completed(batch_tasks):
            count = await task  # 여기서 각 배치의 결과가 나옴
            success_counts.append(count)
            completed_count += 1

            # 오케스트레이터 레벨에서 전체 진행률 로깅 가능
            percent = (completed_count / total_batches) * 100
            logger.info(f"🚀 전체 진행률: {percent:.1f}% ({completed_count}/{total_batches} 배치 처리됨)")
            # 추후 실시간 진행률 표시 SSE

        # 4. [COMMIT & SUMMARY]
        total_success = sum(success_counts)
        session.commit()  # 트랜잭션 최종 커밋

        duration = time.time() - start_time
        logger.info(f"🎉 전체 작업 완료: {total_success}개 단어 강화 성공 ({duration:.2f}s)")

        return total_success


    async def _process_batch_v3(self, session: Session, chunk: List[SentenceTaskDTO], batch_idx: int, total_batches: int) -> int :
        """
            개별 배치를 처리하는 실제 작업 단위
        """
        async with self.semaphore: # 동시 호출 제한 (Rate Limit 관리)
            batch_start_time = time.time()

            # 1. AI에게 보낼 페이로드 조립 DTO => JSON
            payload = self._build_ai_payload(chunk)

            input_message = self.prompt.format_messages(
                payload=json.dumps(payload, ensure_ascii=False),
                format_instructions=self.parser.get_format_instructions()
            )

            try:
                response = await self.llm.ainvoke(input_message)
                ai_data = self.parser.parse(response.content)

                # 3. DB 저장
                # 성공한 단어 개수를 반환합니다.
                success_count = self._save_v3_results(session, chunk, ai_data)

                # session.flush()는 _save_v3_results 내부에서 수행됩니다.
                # 최종 commit은 오케스트레이터가 합니다.

                return success_count

            except Exception as e:
                logger.error(f"❌ [Batch {batch_idx}] 처리 중 예외 발생: {e}")
                return 0


    def _save_v3_results(self,session: Session, chunk: List[SentenceTaskDTO], ai_data: AIWordEnhanceResponseV4):
        """
            AI 응답 데이터를 정규화된 테이블들에 나눠서 저장 (Flush 포함)
        """

        # 1. 매핑 준비: AI의 가상 context_id -> 실제 DB sentence_id
        # SentenceTaskDTO 구조 덕분에 아주 쉽게 맵을 만듭니다.
        context_id_map = {task.sentence_id: task.sentence_id for task in chunk}

        # 2. 단어 객체 맵핑 (Update용)
        # 이번 배치에 포함된 모든 단어 ID 리스트를 추출해서 맵핑
        word_ids = [w.word_id for task in chunk for w in task.words]
        word_map = {w_id: session.get(WordEntry, w_id) for w_id in word_ids}

        success_count = 0

        # --- (1) 원본 자막 번역 저장 (SubtitleTranslation) ---
        # 1. 이번 배치에 포함된 문장 ID 리스트 확보
        sentence_ids = [task.sentence_id for task in chunk]

        # 2. [Select] 이미 번역이 존재하는 문장 ID들을 한 번에 조회 (N+1 방지)
        existing_translations = session.exec(
            select(SubtitleTranslation)
            .where(
                SubtitleTranslation.subtitle_sentence_id.in_(sentence_ids),
                SubtitleTranslation.language_code == "ko"
            )
        ).all()

        # 빠른 조회를 위해 맵으로 변환 {sentence_id: translation_obj}
        trans_map = {t.subtitle_sentence_id: t for t in existing_translations}

        # 3. [Insert or Update]
        for trans in ai_data.trans:
            db_sent_id = context_id_map.get(trans.s_id)
            if not db_sent_id: continue

            if db_sent_id in trans_map:
                # 💡 [Update] 이미 있으면 내용만 갱신 (Dirty Checking 활용)
                trans_map[db_sent_id].translated_text = trans.tr_ko
            else:
                # 💡 [Insert] 없으면 새로 추가
                new_sub_trans = SubtitleTranslation(
                    subtitle_sentence_id=db_sent_id,
                    language_code="ko",
                    translated_text=trans.tr_ko
                )
                session.add(new_sub_trans)

        # --- (2) 창의적 예문 및 해석 저장 (ExampleSentence) ---
        # AI가 준 예문 그룹별로 저장하고 생성된 PK를 보관
        example_id_map = {}  # {ai_group_id: db_example_id}

        for ex in ai_data.exs:
            new_example = ExampleSentence(
                sentence_text=ex.ex_ja
            )
            session.add(new_example)
            session.flush()  # 💡 [ID 확보] DB에 쿼리를 날려 자동 생성된 PK를 받아옵니다 (JPA의 saveAndFlush)

            example_id_map[ex.gid] = new_example.id

            # 예문의 번역도 세트로 저장
            session.add(ExampleTranslation(
                example_id=new_example.id,
                language_code="ko",
                translated_text=ex.ex_ko
            ))

        # --- (3) 단어 메타데이터 업데이트 및 학습 콘텐츠 생성 ---
        for def_res in ai_data.words:
            word_entry = word_map.get(def_res.w_id)
            if not word_entry: continue

            # 메타데이터 업데이트 (Dirty Checking처럼 작동)
            word_entry.japanese_metadata.reading = def_res.r
            word_entry.japanese_metadata.jlpt_level = def_res.lv

            # 해당 단어가 포함된 AI 예문 그룹 ID 찾기
            target_group_id = next((ex.gid for ex in ai_data.exs if def_res.w_id in ex.wids), None)

            if target_group_id in example_id_map:
                session.add(WordLearningContent(
                    word_entry_id=word_entry.id,
                    example_id=example_id_map[target_group_id],
                    meaning=def_res.m,
                    usage_tip=f"JLPT {def_res.lv} 수준의 단어입니다.",
                    language_code="ko"
                ))
                success_count += 1

        return success_count


    def _build_ai_payload(self, chunk: List[SentenceTaskDTO]) -> Dict[str, Any]:
        """
        기존 인터리빙 로직을 유지하면서 응집도 높은 JSON 구조로 페이로드 생성
        """
        # 1. 'data' 섹션 생성: 문장과 소속 단어들을 한 몸으로 묶음
        # AI가 문장(Context)을 보면서 단어를 바로 해석할 수 있게 응집도를 높입니다.
        data = [
            {
                "id": task.sentence_id,  # DB 실제 PK 사용
                "text": task.sentence_text,
                "words": [
                    {"id": w.word_id, "base": w.base_form}
                    for w in task.words
                ]
            }
            for task in chunk
        ]

        # 2. 'groups' 섹션 생성: 기존 인터리빙 로직 유지
        # 문장들을 순회하며 단어를 하나씩 뽑아 골고루 섞인 그룹을 만듭니다.
        all_word_ids = []
        max_words_in_sentence = max(len(task.words) for task in chunk) if chunk else 0

        for i in range(max_words_in_sentence):
            for task in chunk:
                if i < len(task.words):
                    # AI에게는 ID 리스트만 넘겨서 추론 비용(토큰)을 절약합니다.
                    all_word_ids.append(task.words[i].word_id)

        # 3. 추출된 단어 ID들을 3개씩 묶어서 그룹화
        groups = []
        for i in range(0, len(all_word_ids), 3):
            groups.append({
                "gid": (i // 3) + 1,
                "wids": all_word_ids[i: i + 3]
            })

        return {
            "data": data,
            "groups": groups
        }