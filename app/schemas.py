from pydantic import BaseModel, Field
from typing import List

class WordDetail(BaseModel):
    base_form: str = Field(description="원문 단어")
    reading: str = Field(description="단어의 히라가나 읽기 (가타카나도 히라가나로 변환)")
    jlpt_level: str = Field(description="JLPT 등급 (N1, N2, N3, N4, N5 중 하나)")
    meaning: str = Field(description="단어의 대표적인 한국어 뜻")
    usage_example: str = Field(description="해당 단어가 사용된 짧은 일본어 예문")
    usage_meaning: str = Field(description="일본어 예문의 한국어 해석")

class WordEnhanceResponse(BaseModel):
    words: List[WordDetail]