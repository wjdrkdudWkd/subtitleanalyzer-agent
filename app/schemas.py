from pydantic import BaseModel, Field
from typing import List, Optional


class WordDefinition(BaseModel):
    word_id: int = Field(description="제공된 단어의 고유 ID")
    base_form: str = Field(description="단어 원형")
    reading: str = Field(description="히라가나 읽기")
    meaning: str = Field(description="제공된 context를 바탕으로 파악한 한국어 뜻")
    jlpt_level: str = Field(description="JLPT 등급 (N1~N5)")

class GroupedExample(BaseModel):
    group_id: int = Field(description="요청받은 word_groups의 group_id")
    new_sentence_ja: str = Field(description="그룹 내 단어들을 모두 활용해 창작한 새로운 일본어 문장")
    new_sentence_ko: str = Field(description="창작된 문장의 한국어 해석")
    # logic_reasoning: str = Field(description="왜 이런 문장을 만들었는지에 대한 짧은 설명 (디버깅용)")
    word_ids: List[int] = Field(description="이 예문에 포함된 단어 ID 리스트")

class AIWordEnhanceResponseV2(BaseModel):
    word_definitions: List[WordDefinition]
    examples: List[GroupedExample]

class WordDetail(BaseModel):
    base_form: str = Field(description="원문 단어")
    reading: str = Field(description="단어의 히라가나 읽기 (가타카나도 히라가나로 변환)")
    jlpt_level: str = Field(description="JLPT 등급 (N1, N2, N3, N4, N5 중 하나)")
    meaning: str = Field(description="단어의 대표적인 한국어 뜻")
    usage_example: str = Field(description="해당 단어가 사용된 짧은 일본어 예문")
    usage_meaning: str = Field(description="일본어 예문의 한국어 해석")

class WordEnhanceResponse(BaseModel):
    words: List[WordDetail]

class SubtitleTranslationResponse(BaseModel):
    context_id: int = Field(description="제공된 contexts의 id")
    translation_ko: str = Field(description="원본 자막의 한국어 해석")
    translation_en: Optional[str] = Field(default=None, description="원본 자막의 영어 해석 (선택 사항)")

class AIWordEnhanceResponseV3(BaseModel):
    # 🌍 원본 자막 번역 추가
    subtitle_translations: List[SubtitleTranslationResponse]
    word_definitions: List[WordDefinition]
    examples: List[GroupedExample]


class WordDefinitionV4(BaseModel):
    w_id: int  #word_id 💡 이제 실제 DB의 WordEntry.id를 받습니다.
    m: str      #meaning
    r: str      #reading
    lv: str     #jlpt_level

class SentenceTranslationV4(BaseModel):
    s_id: int  # 💡 이제 실제 DB의 SubtitleSentence.id를 받습니다.
    tr_ko: str  #한국어 해석


class ExampleV4(BaseModel):
    gid: int    #group_id
    ex_ja: str  #example_ja
    ex_ko: str  #example_ko
    wids: List[int] # 사용된 단어 ID들

class AIWordEnhanceResponseV4(BaseModel):
    """최종 AI 응답 규격"""
    trans: List[SentenceTranslationV4]
    words: List[WordDefinitionV4]
    exs: List[ExampleV4]

