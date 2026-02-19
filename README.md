# 🤖 Subtitle-Analyzer-Agent

> **LLM 에이전트 기반 문맥 최적화 자막 분석 및 학습 콘텐츠 생성 파이프라인**

본 프로젝트는 외국어 콘텐츠의 자막을 분석하여 학습자에게 최적화된 단어장과 창의적인 예문을 제공하는 AI 에이전트 시스템입니다. (아직 일본어만 지원) 기존 **[Spring-based Subtitle Analyzer](https://github.com/wjdrkdudWkd/subtitleanalyzer)** 의 비즈니스 로직을 계승하면서, LLM의 추론 능력을 극대화할 수 있는 **Python/LangChain** 기반의 에이전트 구조로 재설계되었습니다.

---

## 🚀 Key Features

### 1. Contextual Word Extraction (Implemented)

* **Janome NLP Integration**: 일본어 형태소 분석기를 통해 문맥상 유의미한 품사(명사, 동사, 형용사, 부사)를 정밀하게 추출합니다.
* **Regex Filtering**: 노이즈 데이터(특수문자, 숫자, 의미 없는 감탄사 등)를 정규표현식으로 정제하여 데이터 품질을 확보합니다.

### 2. Sentence Interleaving Strategy (Implemented)

* **Hallucination Control**: AI가 원본 자막을 그대로 복사하는 문제를 방지하기 위해, 서로 다른 문장에서 추출된 단어들을 교차 배치(Round-Robin)하여 새로운 문맥을 강제합니다.
* **Creative Reasoning**: 서로 이질적인 단어 조합을 제공함으로써 AI가 학습 효과가 높은 고유한 상황극 예문을 창조하도록 유도합니다.

### 3. ID-based Data Mapping (Implemented)

* **Data Integrity**: 동음이의어 및 중복 단어 식별 문제를 해결하기 위해 DB PK(Primary Key) 기반의 매핑 시스템을 구축하여 AI 응답과 엔티티 간의 정확한 연결을 보장합니다.

### 4. High Performance Pipeline (WIP 🚧)

* **Async Batching**: `asyncio`와 Semaphore를 활용하여 여러 단어를 효율적으로 처리하는 병렬 배치 파이프라인을 구축 중입니다. (API 비용 절감 및 속도 최적화 예정)

### 5. Robust Data Integrity (WIP 🚧)

* **Feedback Loop**: Pydantic 기반의 스키마 검증 및 데이터 누락 시 AI에게 피드백을 전달하여 스스로 수정하게 하는 재시도(Retry) 로직을 개발 중입니다.

---

## 🛠 Tech Stack

* **Core**: Python 3.11+, FastAPI
* **AI Ecosystem**: LangChain, OpenAI GPT-4o-mini, LangSmith (Tracing)
* **Data Layer**: SQLModel (SQLAlchemy-based ORM), SQLite/PostgreSQL
* **NLP**: Janome (Japanese Morphological Analysis), jaconv

---

## 🏗 System Architecture

1. **Parsing Phase**: 자막 파일에서 문장과 단어를 추출하고 정규화합니다.
2. **Interleaving Phase**: 추출된 단어들을 원본 문장별 '바구니'에 담은 뒤, 각 바구니에서 하나씩 꺼내어 새로운 학습 그룹(3개 단위)을 생성합니다.
3. **Agent Logic Phase**: LangChain을 통해 `contexts`(참조용 자막)와 `word_groups`(섞인 단어들)를 AI에게 전달합니다.
4. **Content Generation**: AI는 제공된 맥락을 이해하되, 원본과 다른 고유한 예문과 상세 메타데이터(읽기, 의미, JLPT 등급)를 생성합니다.

---

## 📝 Core Logic: Interleaving Strategy

본 프로젝트의 핵심 차별점은 AI에게 전달되는 데이터의 **엔트로피(Entropy)** 를 조절하는 것입니다.

```python
# 문장별 단어 바구니에서 라운드 로빈 방식으로 단어를 섞는 핵심 로직
buckets = list(sentence_buckets.values())
max_len = max(len(b) for b in buckets)
for i in range(max_len):
    for bucket in buckets:
        if i < len(bucket):
            interleaved_words.append(bucket[i])

```

이 과정을 통해 AI는 특정 자막 문장에 매몰되지 않고, 창의적인 언어 조합을 만들어내어 사용자에게 풍부한 학습 경험을 제공합니다.

---

## 📈 Roadmap & Upcoming Tasks

* [ ] **Multilingual Support**: 다국어(영어, 중국어 등) 학습 지원을 위한 `language_code` 기반 테이블 정규화.
* [ ] **Example Table Separation**: 데이터 중복 방지를 위한 예문 전용 테이블 분리 및 N:M 관계 고도화.
* [ ] **Self-Correction Logic**: AI 응답 품질을 스스로 검사하고 교정하는 Self-Correction 에이전트 도입.

---

## 🔗 Related Projects

* **[Subtitle Analyzer (Java/Spring Version)](https://github.com/wjdrkdudWkd/subtitleanalyzer)**: 본 프로젝트의 모태가 된 기존 백엔드 시스템입니다.

---
