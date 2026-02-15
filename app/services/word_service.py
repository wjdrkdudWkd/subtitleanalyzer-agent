import re
import jaconv
from janome.tokenizer import Tokenizer
from sqlmodel import Session, select
from app.models import SubtitleSentence, WordEntry, JapaneseWordMetadata


class WordService:
    def __init__(self):
        self.tokenizer = Tokenizer()

    def _validate_word(self, base_form, pos):
        if not base_form:
            return False, "EMPTY"

            # 1. 특수문자/숫자 패턴 정의 (숫자, 문장부호, 기호 등)
            # \W는 문자가 아닌 것, [0-9]는 숫자, _는 언더바
        symbol_pattern = re.compile(r'[0-9\W_]')

        # 🔍 첫 번째 문자가 특수문자/숫자인지 확인
        if symbol_pattern.match(base_form[0]):
            # 첫 글자가 특수문자라면, '전체 문자'가 특수문자인지 검사
            # all()을 사용해 중간에 일반 문자가 하나라도 섞여 있으면 True(통과)가 됨
            if re.fullmatch(r'[\d\W_]+', base_form):
                return False, "ALL_SYMBOLS_OR_NUM"

            # 첫 글자는 특수문자지만 뒤에 일반 문자가 섞여 있다면?
            # (예: "!안녕", "1등") -> 유효한 단어로 보고 통과시킴

        # 2. 의미 없는 한 글자 가나 (조사 성격이나 단순 감탄사)
        if len(base_form) == 1:
            # 히라가나/가타카나 한 글자이면서 명사가 아닌 경우
            if re.match(r'^[ぁ-んァ-ヶー]$', base_form) and pos != '名詞':
                return False, "SINGLE_KANA_NOISE"

        return True, None

    def extract_words_from_subtitle(self, session: Session, subtitle_id: int):
        # 문장 데이터 가져오기
        statement = select(SubtitleSentence).where(SubtitleSentence.subtitle_id == subtitle_id)
        sentences = session.exec(statement).all()

        # 중복 단어 방지를 위한 맵 {base_form: WordEntry}
        word_map = {}

        for sent in sentences:
            # 텍스트 확보
            text = sent.sentence_text

            # 2. Janome으로 형태소 분석
            for token in self.tokenizer.tokenize(text):
                base_form = token.base_form
                pos = token.part_of_speech.split(',')[0]

                # 의미 있는 품사(명사, 동사, 형용사, 부사)만 추출
                if pos in ['名詞', '動詞', '形容詞', '副詞']:
                    if base_form not in word_map:

                        is_valid, skip_reason = self._validate_word(base_form, pos)

                        # 1. 가타카나 요미가나를 히라가나로 변환
                        # Janome 결과가 '*'인 경우(기호 등)는 원문(base_form)을 사용
                        raw_reading = token.reading if token.reading != "*" else base_form
                        hiragana_reading = jaconv.kata2hira(raw_reading)

                        # WordEntry 생성
                        word_entry = WordEntry(
                            subtitle_id=subtitle_id,
                            first_occurrence_id=sent.id,
                            base_form=base_form,
                            language="ja",
                            part_of_speech=pos,
                            frequency=1,
                            is_valid=is_valid,
                            skip_reason=skip_reason
                        )

                        # 오직 히라가나와 장음(ー)으로만 구성된 패턴
                        hiragana_only_pattern = re.compile(r'^[ぁ-んー]+$')

                        is_pure_hiragana = bool(hiragana_only_pattern.match(base_form))
                        metadata = JapaneseWordMetadata(
                            # 히라가나면 그대로 넣고, 아니면(한자/가타카나) AI가 채우도록 비워둠
                            reading=base_form if is_pure_hiragana else None,
                            # 히라가나여도 JLPT 등급은 모르니 일단 WAIT (혹은 별도 상태값)
                            jlpt_level="WAIT",
                            word_entry=word_entry
                        )
                        word_entry.japanese_metadata = metadata
                        word_map[base_form] = word_entry
                    else:
                        # 이미 등록된 단어면 빈도수만 증가
                        word_map[base_form].frequency += 1

        # 4. 벌크 저장
        for word in word_map.values():
            session.add(word)

        session.commit()
        return len(word_map)