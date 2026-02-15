from fastapi import FastAPI, UploadFile
from fastapi.params import Depends, Form, File
from sqlmodel import Session

from app.database import get_session
from app.services.ai_word_service import AIWordService
from app.services.subtitle_service import SubtitleService
from app.services.word_service import WordService
from app.models import create_db_and_tables

app = FastAPI(
    title="Subtitle Analyzer Agent API",
    description="자막 분석 및 단어장 생성 에이전트 서비스",
    version="0.1.0"
)

@app.on_event("startup")
def on_startup():
    # 서버 시작 시 DB 테이블 생성 (JPA의 ddl-auto: update와 유사)
    create_db_and_tables()


@app.post("/subtitles/ingest", tags=["Ingest"])
async def ingest_subtitle(
        file: UploadFile = File(...),
        language: str = Form("ja"),
        session: Session = Depends(get_session)  # 의존성 주입 (DI)
):
    content = (await file.read()).decode("utf-8")

    # 서비스 호출
    subtitle = SubtitleService.ingest_subtitle(
        session, content, file.filename, language
    )

    return {
        "id": subtitle.id,
        "title": subtitle.title,
        "sentence_count": len(subtitle.sentences)
    }

# 3. 헬스체크
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok"}

word_service = WordService()

@app.post("/subtitles/{subtitle_id}/analyze-words", tags=["Analysis"])
async def analyze_words(subtitle_id: int, session: Session = Depends(get_session)):
    """
    저장된 자막 문장들에서 단어를 추출하여 WordEntry 테이블을 채웁니다.
    """
    count = word_service.extract_words_from_subtitle(session, subtitle_id)

    return {
        "subtitle_id": subtitle_id,
        "extracted_word_count": count,
        "status": "COMPLETED"
    }


@app.post("/subtitles/{subtitle_id}/enhance")
async def enhance_subtitle_words(subtitle_id: int, session: Session = Depends(get_session)):
    ai_service = AIWordService()
    # 2. 비동기 함수이므로 앞에 await 필수 추가
    count = await ai_service.enhance_words_hybrid(session, subtitle_id)
    return {"message": f"{count} words enhanced successfully"}


# --reload 옵션을 주면 코드 수정 시 서버가 자동으로 재시작됩니다. (Hot Reload)
# uvicorn app.api.main:app --reload