# AI 텍스트 마이닝 스튜디오

영어 논술 텍스트 마이닝 실습 및 AI 글쓰기 분석 플랫폼

## 📋 프로젝트 개요

학생들의 영어 에세이를 AI 기반으로 분석하여 글쓰기 실력을 진단하고 개선 방향을 제시하는 교육용 텍스트 마이닝 플랫폼입니다.

## 🎯 주요 기능

### 1. 감성 분석 원리 체험 (4단계)
- **1단계**: 어휘 사전 기반 감성 분석
- **2단계**: TF-IDF + 머신러닝 기반 분석
- **3단계**: VADER 고급 규칙 기반 분석
- **4단계**: 8개 감정별 단어 분석 (기쁨, 분노, 슬픔, 두려움, 놀라움, 혐오, 신뢰, 기대)

### 2. 품사 분석 원리 체험 (3단계)
- **1단계**: 수동 규칙 기반 품사 태깅
- **2단계**: NLTK 라이브러리 기반 분석
- **3단계**: 패턴 발견 및 통계 분석

### 3. 글쓰기 수준 종합 진단
- **1단계**: 통계적 텍스트 분석 (문장 길이, 어휘 다양성, 품사 패턴)
- **2단계**: 어휘 수준 분석 (고급 어휘 비율, 학술 어휘 점수)
- **3단계**: 문장 유사도 분석 (논리적 연결성, 주제 일관성)
- **4단계**: 문법 오류 패턴 분석 (시제, 관사, 수일치, 전치사, 문장 구조)
- **종합 진단**: 전체 글쓰기 수준 평가 및 맞춤형 학습 계획 제시

## 🛠 기술 스택

- **Framework**: Streamlit
- **NLP Libraries**: NLTK, VADER Sentiment Analysis
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib
- **Database**: Redis (세션 관리)
- **Deployment**: Streamlit Cloud

## 📦 설치 및 실행

### 필수 요구사항
```bash
python >= 3.8
```

### 로컬 실행
```bash
# 저장소 클론
git clone https://github.com/dingsong2022/ai-text-mining-studio.git
cd ai-text-mining-studio

# 패키지 설치
pip install -r requirements.txt

# NLTK 데이터 다운로드
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('vader_lexicon')"

# 앱 실행
streamlit run main.py
```

## 🔧 최근 업데이트 (2025-10-14)

### ✨ 새로운 기능
- 8개 감정별 단어 분석 추가 (감성 분석 4단계)
- 문법 오류 패턴 분석 표시 추가 (글쓰기 진단 4단계)
- Redis 연결 상태 표시기 추가

### 🐛 버그 수정
- 문장 유사도 분석 빈 결과 표시 문제 수정
- 문법 분석 차트 데이터 구조 오류 수정
- 중복 import로 인한 UnboundLocalError 해결
- 차트에 [object Object] 표시되는 문제 해결

## 📊 데이터 구조

### 사용자 에세이 데이터
```python
{
    'topic_name': str,          # 에세이 주제
    'essay_text': str,          # 본문
    'created_at': datetime,     # 작성 시간
    'username': str             # 작성자
}
```

### 분석 결과 데이터
```python
{
    'step1_statistical': {...},    # 통계 분석
    'step1a_emotion': {...},       # 8개 감정 분석
    'step2_vocabulary': {...},     # 어휘 수준
    'step3_grammar': {...},        # 문법 분석
    'step3_similarity': {...},     # 문장 유사도
    'step5_comprehensive': {...}   # 종합 진단
}
```

## 🔒 환경 변수

Streamlit Cloud 배포시 필요한 secrets:
```toml
[redis]
host = "your-redis-host"
port = 6379
password = "your-redis-password"
```

## 📁 프로젝트 구조

```
ai-text-mining-studio/
├── main.py                      # 메인 애플리케이션
├── modules/
│   └── preprocessor.py          # 텍스트 분석 엔진
├── requirements.txt             # 패키지 의존성
├── .streamlit/
│   └── secrets.toml            # 환경 설정 (로컬)
└── README.md                    # 프로젝트 문서
```

## 🎓 교육적 활용

- 텍스트 마이닝 원리 학습
- NLP 기법 단계별 체험
- 영어 글쓰기 실력 자가 진단
- AI 기반 피드백 시스템 이해

## 📈 향후 계획

- [ ] 다국어 지원 (한국어 텍스트 분석)
- [ ] 에세이 비교 분석 기능
- [ ] 실시간 작문 피드백
- [ ] 학습 진도 추적 대시보드

## 👨‍💻 기여자

- Main Developer: dingsong2022

## 📄 라이선스

이 프로젝트는 교육용으로 제작되었습니다.

## 🔗 관련 링크

- [Streamlit 배포 URL](https://ai-text-mining-studio.streamlit.app)
- [GitHub Repository](https://github.com/dingsong2022/ai-text-mining-studio)

---

**마지막 업데이트**: 2025-10-14
**버전**: 1.2.0
