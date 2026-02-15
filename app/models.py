import json
from datetime import datetime
from typing import List, Optional, Dict
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
    sentence_text: str = Field(max_length=1000, nullable=False)
    sentence_order: int = Field(nullable=False)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    embedding: Optional[List[float]] = Field(default=None, sa_type=JSON)

    subtitle: Subtitle = Relationship(back_populates="sentences")


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
    meaning: str = Field(nullable=False)
    usage_tip: Optional[str] = None
    generated_example: Dict[str, str] = Field(default_factory=dict, sa_type=JSON)
    model_name: str = Field(default="gpt-4o-mini")

    word_entry: WordEntry = Relationship(back_populates="learning_content")


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