import json
from datetime import datetime
from typing import List, Optional, Dict

from sqlalchemy import Column, UniqueConstraint
from sqlmodel import SQLModel, Field, Relationship, JSON, create_engine


# 1. JSON 직렬화 시 ASCII 이스케이프 방지 (JSON 필드용)
def json_serializer(obj):
    return json.dumps(obj, ensure_ascii=False)


# --- [Domain Entities] ---

class Subtitle(SQLModel, table=True):
    __tablename__ = "subtitles"
    id: Optional[int] = Field(default=None, primary_key=True)
    source_type: str = Field(nullable=False)  # FILE, YOUTUBE, NETFLIX
    source_key: str = Field(nullable=False)
    language: str = Field(nullable=False, max_length=10)
    title: Optional[str] = None
    status: str = Field(default="PENDING")
    created_at: datetime = Field(default_factory=datetime.now)

    # Relationship
    sentences: List["SubtitleSentence"] = Relationship(back_populates="subtitle")
    word_entries: List["WordEntry"] = Relationship(back_populates="subtitle")


class SubtitleSentence(SQLModel, table=True):
    __tablename__ = "subtitle_sentences"

    id: Optional[int] = Field(default=None, primary_key=True)
    subtitle_id: int = Field(foreign_key="subtitles.id", nullable=False)

    # 🌍 글로벌 대응을 위한 언어 코드 추가 (예: 'ja', 'ko', 'en')
    language_code: str = Field(default="ja", index=True, max_length=10, nullable=False)

    sentence_text: str = Field(max_length=1000, nullable=False)
    sentence_order: int = Field(nullable=False)

    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # 🔍 벡터 검색을 위한 임베딩 데이터
    embedding: Optional[List[float]] = Field(default=None, sa_column=Column(JSON))

    # 관계 설정
    subtitle: "Subtitle" = Relationship(back_populates="sentences")

    # 번역 테이블과의 관계
    translations: List["SubtitleTranslation"] = Relationship(back_populates="sentence")


class WordEntry(SQLModel, table=True):
    __tablename__ = "word_entries"
    id: Optional[int] = Field(default=None, primary_key=True)
    subtitle_id: int = Field(foreign_key="subtitles.id", nullable=False)
    first_occurrence_id: Optional[int] = Field(foreign_key="subtitle_sentences.id")
    base_form: str = Field(nullable=False)
    language: str = Field(nullable=False)
    part_of_speech: Optional[str] = None
    frequency: int = Field(default=1)
    is_valid: bool = Field(default=True, index=True)  # 필터링용 인덱스 추가
    skip_reason: Optional[str] = None  # 왜 제외되었는지 기록 (디버깅용)

    subtitle: Subtitle = Relationship(back_populates="word_entries")
    learning_content: Optional["WordLearningContent"] = Relationship(back_populates="word_entry")

    # 신규 추가: 일본어 전용 메타데이터 (1:1 관계)
    japanese_metadata: Optional["JapaneseWordMetadata"] = Relationship(
        back_populates="word_entry",
        sa_relationship_kwargs={"uselist": False}  # Java의 @OneToOne 설정
    )


class JapaneseWordMetadata(SQLModel, table=True):
    __tablename__ = "japanese_word_metadata"
    id: Optional[int] = Field(default=None, primary_key=True)
    word_entry_id: int = Field(foreign_key="word_entries.id", unique=True, nullable=False)

    reading: Optional[str] = None  # 히라가나/가타카나 읽기
    jlpt_level: Optional[str] = Field(default=None, max_length=2)  # N1 ~ N5

    word_entry: WordEntry = Relationship(back_populates="japanese_metadata")


class WordLearningContent(SQLModel, table=True):
    __tablename__ = "word_learning_contents"
    id: Optional[int] = Field(default=None, primary_key=True)
    word_entry_id: int = Field(foreign_key="word_entries.id", unique=True, nullable=False)

    # 🆕 신규 연결: 생성된 예문 ID (정규화)
    example_id: Optional[int] = Field(foreign_key="example_sentences.id")

    meaning: str = Field(nullable=False)
    language_code: str = Field(default="ko", index=True)  # 뜻의 언어 (글로벌 대응)
    usage_tip: Optional[str] = None
    model_name: str = Field(default="gpt-4o-mini")

    word_entry: "WordEntry" = Relationship(back_populates="learning_content")
    example: Optional["ExampleSentence"] = Relationship(back_populates="word_contents")

# --- [v0.0.3 신규 추가 테이블] ---

class SubtitleTranslation(SQLModel, table=True):
    """원본 자막 문장의 다국어 해석본"""
    __tablename__ = "subtitle_translations"

    # 복합 유니크 제약조건 설정 (JPA의 @UniqueConstraint와 동일)
    __table_args__ = (
        UniqueConstraint("subtitle_sentence_id", "language_code", name="uq_sentence_lang"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    subtitle_sentence_id: int = Field(foreign_key="subtitle_sentences.id", nullable=False)

    language_code: str = Field(index=True, max_length=10)  # ko, en, zh 등
    translated_text: str = Field(max_length=1000, nullable=False)

    sentence: "SubtitleSentence" = Relationship(back_populates="translations")

class ExampleSentence(SQLModel, table=True):
    """AI가 창작한 고유 예문 (단어 그룹당 1개 생성)"""
    __tablename__ = "example_sentences"
    id: Optional[int] = Field(default=None, primary_key=True)
    sentence_text: str = Field(max_length=1000, nullable=False)  # 보통 일본어 원문
    created_at: datetime = Field(default_factory=datetime.now)

    # 관계 설정
    translations: List["ExampleTranslation"] = Relationship(back_populates="example")
    word_contents: List["WordLearningContent"] = Relationship(back_populates="example")

class ExampleTranslation(SQLModel, table=True):
    """생성된 예문의 다국어 해석본"""
    __tablename__ = "example_translations"
    id: Optional[int] = Field(default=None, primary_key=True)
    example_id: int = Field(foreign_key="example_sentences.id", nullable=False)

    language_code: str = Field(index=True, max_length=10)  # ko, en 등
    translated_text: str = Field(max_length=1000, nullable=False)

    example: ExampleSentence = Relationship(back_populates="translations")


# --- [DB Engine Setup] ---

sqlite_url = "sqlite:///database.db"
engine = create_engine(
    sqlite_url,
    echo=True,
    connect_args={"check_same_thread": False},
    json_serializer=json_serializer
)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)