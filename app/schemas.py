from pydantic import BaseModel, Field
from typing import List

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
    logic_reasoning: str = Field(description="왜 이런 문장을 만들었는지에 대한 짧은 설명 (디버깅용)")
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