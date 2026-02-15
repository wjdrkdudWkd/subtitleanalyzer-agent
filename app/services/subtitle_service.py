import pysubs2
from sqlmodel import Session

from app.models import SubtitleSentence, Subtitle


class SubtitleService:
    @staticmethod
    def ingest_subtitle(session: Session, file_content: str, filename: str, language: str):
        # 1. 자막 파싱 (라이브러리가 읽어주는 그대로 가져옵니다)
        subs = pysubs2.SSAFile.from_string(file_content)

        subtitle = Subtitle(
            source_type="FILE",
            source_key=filename,
            language=language,
            title=filename,
            status="PENDING"
        )

        for i, line in enumerate(subs):
            # 2. 인코딩 시도 안 함! 줄바꿈만 정리해서 원본 그대로 저장합니다.
            raw_text = line.plaintext.replace("\n", " ").strip()

            if not raw_text: continue

            sentence = SubtitleSentence(
                sentence_text=raw_text,  # \uXXXX 가 포함된 원본 그대로 저장
                sentence_order=i + 1,
                start_time=line.start / 1000.0,
                end_time=line.end / 1000.0,
                subtitle=subtitle
            )
            subtitle.sentences.append(sentence)

        # 3. DB 저장 (가장 안전한 ASCII 세이프 상태)
        session.add(subtitle)
        session.commit()
        session.refresh(subtitle)

        return subtitle