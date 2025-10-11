import streamlit as st
import pandas as pd
from modules.data_loader import DataLoader
from modules.preprocessor import TextPreprocessor
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(
    page_title="AI Text Mining Studio",
    page_icon="🔍",
    layout="wide"
)

# 로그인 함수
def check_login(username, password):
    """사용자 인증 (Redis 캐싱)"""
    try:
        data_loader = st.session_state.get('data_loader')
        if not data_loader:
            data_loader = DataLoader()
            st.session_state.data_loader = data_loader

        # 1. Redis 캐시에서 사용자 정보 확인
        cache_key = "users:credentials"
        credentials = data_loader.cache.get(cache_key)

        if credentials is None:
            print(f"Cache MISS: {cache_key} - Fetching from Google Sheets")

            # Google Sheets 연결 확인
            if not data_loader.sheet:
                st.error("Google Sheets 연결에 실패했습니다. Streamlit Secrets 설정을 확인해주세요.")
                return False

            # 2. Google Sheets에서 사용자 정보 가져오기
            users_sheet = data_loader.sheet.worksheet("사용자정보")
            all_values = users_sheet.get_all_values()

            # 사용자명:비밀번호 딕셔너리로 변환
            credentials = {}
            for row in all_values[1:]:  # 헤더 제외
                if len(row) >= 2:
                    stored_username = row[0].strip()
                    stored_password = row[1].strip()
                    if stored_username and stored_password:
                        credentials[stored_username] = stored_password

            # 3. Redis 캐시에 저장 (10분)
            data_loader.cache.set(cache_key, credentials, ttl=600)
        else:
            print(f"Cache HIT: {cache_key}")

        # 캐시된 인증 정보에서 확인
        if username in credentials and credentials[username] == password:
            return True

        return False

    except Exception as e:
        st.error(f"로그인 확인 오류: {e}")
        return False

def login_page():
    """로그인 페이지"""
    st.markdown('<h1 class="main-header">🔍 AI Text Mining Studio</h1>', unsafe_allow_html=True)
    st.markdown("**개인 맞춤형 영어 에세이 텍스트 마이닝 분석 플랫폼**")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.subheader("🔑 로그인")
        
        with st.form("login_form"):
            username = st.text_input("아이디")
            password = st.text_input("비밀번호", type="password")
            login_button = st.form_submit_button("로그인", use_container_width=True)
            
            if login_button:
                if username and password:
                    if check_login(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.success("로그인 성공!")
                        st.rerun()
                    else:
                        st.error("아이디 또는 비밀번호가 잘못되었습니다.")
                else:
                    st.warning("아이디와 비밀번호를 입력해주세요.")
        
        st.markdown("---")
        st.info("💡 **English Essay Writing Studio**에서 사용하던 계정으로 로그인하세요!")

def main_app():
    """메인 애플리케이션"""
    # 데이터 로더 및 전처리기 초기화
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader()
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = TextPreprocessor()
    
    data_loader = st.session_state.data_loader
    preprocessor = st.session_state.preprocessor
    username = st.session_state.username
    
    # 헤더
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown('<h1 class="main-header">🔍 AI Text Mining Studio</h1>', unsafe_allow_html=True)
        st.markdown(f"**{username}님의 영어 에세이 텍스트 마이닝 분석**")
    
    with col2:
        if st.button("로그아웃", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    st.markdown("---")
    
    # 사용자 데이터 로딩
    with st.spinner(f"{username}님의 에세이 데이터를 불러오는 중..."):
        essay_data = data_loader.get_student_essays(username)
    
    if essay_data.empty:
        st.warning(f"📝 {username}님의 에세이 데이터가 없습니다.")
        st.info("**English Essay Writing Studio**에서 먼저 에세이를 작성해주세요!")
        return
    
    # 기본 통계 표시
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("내 에세이 수", len(essay_data))
    with col2:
        if 'total_score' in essay_data.columns:
            avg_score = essay_data['total_score'].mean()
            st.metric("평균 점수", f"{avg_score:.1f}점")
        else:
            st.metric("평균 점수", "N/A")
    with col3:
        total_words = 0
        for text in essay_data['essay_text']:
            if text and not pd.isna(text):
                total_words += len(str(text).split())
        st.metric("총 작성 단어", f"{total_words:,}")
    with col4:
        unique_topics = essay_data['topic_name'].nunique() if 'topic_name' in essay_data.columns else 0
        st.metric("다룬 주제 수", unique_topics)
    
    # 탭으로 기능 구분
    tab1, tab2, tab3 = st.tabs([
        "📚 내 에세이 모음", 
        "🔬 텍스트 마이닝 실습", 
        "🎓 종합 분석"
    ])
    
    with tab1:
        show_essay_collection(essay_data, username, data_loader)
    
    with tab2:
        show_text_mining_practice(essay_data, preprocessor, username, data_loader)
    
    with tab3:
        show_comprehensive_analysis(essay_data, preprocessor, username)

def show_essay_collection(essay_data, username, data_loader):
    """에세이 모음 표시"""
    st.subheader(f"📝 {username}님이 작성한 모든 에세이")
    
    if essay_data.empty:
        st.warning("작성한 에세이가 없습니다.")
        return
    
    # 기본 통계
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("총 에세이 수", len(essay_data))
    with col2:
        total_words = 0
        for text in essay_data['essay_text']:
            if text and not pd.isna(text):
                total_words += len(str(text).split())
        st.metric("총 작성 단어", f"{total_words:,}")
    with col3:
        if 'total_score' in essay_data.columns:
            avg_score = essay_data['total_score'].mean()
            st.metric("평균 점수", f"{avg_score:.1f}점")
    
    # 합친 텍스트 보기
    st.subheader("📄 모든 에세이를 하나로 합친 텍스트")
    
    if st.button("내 모든 에세이 텍스트 보기"):
        with st.spinner("에세이 텍스트를 합치는 중..."):
            combined_text = data_loader.get_combined_essay_text(username)
            
            if combined_text:
                st.success(f"✅ 총 {len(combined_text.split())}개 단어로 구성된 텍스트입니다!")
                
                # 텍스트 미리보기
                preview_length = 500
                preview_text = combined_text[:preview_length]
                if len(combined_text) > preview_length:
                    preview_text += "..."
                
                st.text_area(
                    "합쳐진 텍스트 미리보기:", 
                    preview_text, 
                    height=200,
                    help=f"전체 텍스트는 {len(combined_text)}자입니다."
                )
                
                # 세션에 저장
                st.session_state.combined_text = combined_text
                st.info("✅ 텍스트가 준비되었습니다! '텍스트 마이닝 실습' 탭에서 분석을 시작하세요.")
            else:
                st.warning("에세이 텍스트를 찾을 수 없습니다.")
    
    # 개별 에세이 목록
    st.subheader("📋 개별 에세이 목록")
    
    for i, (_, row) in enumerate(essay_data.iterrows(), 1):
        with st.expander(f"에세이 {i}: {row.get('topic_name', 'Unknown')[:50]}..."):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**주제:**", row.get('topic_description', 'N/A'))
                st.write("**작성일:**", row.get('created_at', 'N/A'))
            
            with col2:
                if 'total_score' in row and not pd.isna(row['total_score']):
                    st.metric("점수", f"{row['total_score']:.0f}점")
            
            # 에세이 내용
            essay_text = row.get('essay_text', '')
            if essay_text and not pd.isna(essay_text):
                st.text_area(
                    "에세이 내용:", 
                    str(essay_text), 
                    height=150, 
                    disabled=True,
                    key=f"essay_{i}"
                )
            else:
                st.warning("에세이 내용이 없습니다.")

def show_text_mining_practice(essay_data, preprocessor, username, data_loader):
    """단계별 텍스트 마이닝 실습"""
    st.subheader(f"🔬 {username}님의 텍스트 마이닝 실습")
    
    # 합쳐진 텍스트 확인
    if 'combined_text' not in st.session_state:
        st.warning("📝 먼저 '내 에세이 모음' 탭에서 텍스트를 준비해주세요!")
        return
    
    combined_text = st.session_state.combined_text
    
    if not combined_text:
        st.warning("분석할 텍스트가 없습니다.")
        return
    
    st.success(f"✅ 분석 준비 완료! (총 {len(combined_text.split())}개 단어)")
    
    # 단계별 실습
    st.markdown("---")
    
    # 전처리 단계별 실습
    st.subheader("🔧 텍스트 전처리 단계별 실습")
    st.write("텍스트를 단계별로 정제해보며 각 단계의 효과를 확인해보겠습니다.")
    
    # 샘플 텍스트 선택
    sample_length = min(500, len(combined_text))
    sample_text = combined_text[:sample_length]
    
    st.write("**분석할 샘플 텍스트:**")
    st.text_area("", sample_text, height=100, disabled=True, key="sample_text")
    
    if st.button("🔍 전처리 단계별 분석 시작"):
        with st.spinner("단계별 전처리를 진행하는 중..."):
            # 직접 NLTK로 전처리 단계별 결과 생성
            from nltk.stem import PorterStemmer, WordNetLemmatizer
            
            # 각 단계별 처리
            steps = {}
            
            # 원본
            steps['원본'] = sample_text
            steps['원본_단어수'] = len(sample_text.split()) if sample_text else 0
            
            # 1단계: 기본 정제
            step1 = preprocessor.step1_basic_cleaning(sample_text)
            steps['1단계_기본정제'] = step1
            steps['1단계_단어수'] = len(step1.split()) if step1 else 0
            
            # 2단계: 불용어 제거
            step2 = preprocessor.step2_remove_stopwords(step1)
            steps['2단계_불용어제거'] = step2
            steps['2단계_단어수'] = len(step2.split()) if step2 else 0
            
            # 3단계: 어간 추출 (직접 구현)
            if step2:
                stemmer = PorterStemmer()
                words = step2.split()
                stemmed_words = [stemmer.stem(word) for word in words]
                step3 = ' '.join(stemmed_words)
            else:
                step3 = step2
            steps['3단계_어간추출'] = step3
            steps['3단계_단어수'] = len(step3.split()) if step3 else 0
            
            # 4단계: 표제어 추출 (직접 구현)
            if step2:
                lemmatizer = WordNetLemmatizer()
                words = step2.split()
                lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
                step4 = ' '.join(lemmatized_words)
            else:
                step4 = step2
            steps['4단계_표제어추출'] = step4
            steps['4단계_단어수'] = len(step4.split()) if step4 else 0
            
            # 5단계: 개체명 인식 (NER) - 원본 텍스트에서 수행
            import re
            try:
                import spacy
                # spacy 모델이 없을 경우를 대비한 간단한 패턴 기반 NER
                ner_results = []
                
                # 인명 패턴 (대문자로 시작하는 연속된 단어)
                person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
                persons = re.findall(person_pattern, sample_text)
                
                # 장소명 패턴 (특정 키워드와 함께 나오는 대문자 단어)
                place_keywords = r'\b(?:in|at|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
                places = re.findall(place_keywords, sample_text)
                
                # 기관명 패턴 (School, University, Company 등이 포함된 패턴)
                org_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:School|University|College|Company|Corporation|Inc|Ltd))\b'
                organizations = re.findall(org_pattern, sample_text)
                
                # 시간 패턴 (년도, 월, 요일 등)
                time_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|\d{4}|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|today|yesterday|tomorrow)\b'
                times = re.findall(time_pattern, sample_text, re.IGNORECASE)
                
                ner_results = {
                    'PERSON': list(set(persons)),
                    'PLACE': list(set(places)),
                    'ORG': list(set(organizations)),
                    'TIME': list(set(times))
                }
                
            except ImportError:
                # spacy가 없을 경우 간단한 패턴 기반 NER만 사용
                import re
                ner_results = {}
                
                # 인명 패턴 (대문자로 시작하는 연속된 단어)
                person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
                persons = re.findall(person_pattern, sample_text)
                
                # 장소명 패턴
                place_keywords = r'\b(?:in|at|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
                places = re.findall(place_keywords, sample_text)
                
                # 기관명 패턴
                org_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:School|University|College|Company|Corporation|Inc|Ltd))\b'
                organizations = re.findall(org_pattern, sample_text)
                
                # 시간 패턴
                time_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|\d{4}|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|today|yesterday|tomorrow)\b'
                times = re.findall(time_pattern, sample_text, re.IGNORECASE)
                
                ner_results = {
                    'PERSON': list(set(persons)),
                    'PLACE': list(set(places)), 
                    'ORG': list(set(organizations)),
                    'TIME': list(set(times))
                }
            
            steps['5단계_NER결과'] = ner_results
            steps['5단계_개체수'] = sum(len(entities) for entities in ner_results.values())
            
            preprocessing_steps = steps
            
            # 결과 표시
            st.subheader("📋 전처리 단계별 결과")
            
            # 1단계: 기본 정제
            with st.expander("🔧 1단계: 기본 정제 (HTML 태그, 특수문자, 공백 정리)", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Before (원본):**")
                    st.write(f"단어 수: **{preprocessing_steps['원본_단어수']}개**")
                    st.text_area("", preprocessing_steps['원본'][:200] + "...", height=120, disabled=True, key="before_1")
                
                with col2:
                    st.write("**After (기본 정제):**")
                    st.write(f"단어 수: **{preprocessing_steps['1단계_단어수']}개**")
                    st.text_area("", preprocessing_steps['1단계_기본정제'][:200] + "...", height=120, disabled=True, key="after_1")
                
                word_diff_1 = preprocessing_steps['원본_단어수'] - preprocessing_steps['1단계_단어수']
                if word_diff_1 > 0:
                    st.success(f"✅ {word_diff_1}개 단어가 정리되었습니다! (HTML 태그, 특수문자 제거)")
                else:
                    st.info("특별히 제거된 내용이 없습니다.")
            
            # 2단계: 불용어 제거
            with st.expander("🚫 2단계: 불용어 제거 (의미 없는 단어들 제거)", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Before (기본 정제):**")
                    st.write(f"단어 수: **{preprocessing_steps['1단계_단어수']}개**")
                    st.text_area("", preprocessing_steps['1단계_기본정제'][:200] + "...", height=120, disabled=True, key="before_2")
                
                with col2:
                    st.write("**After (불용어 제거):**")
                    st.write(f"단어 수: **{preprocessing_steps['2단계_단어수']}개**")
                    st.text_area("", preprocessing_steps['2단계_불용어제거'][:200] + "...", height=120, disabled=True, key="after_2")
                
                word_diff_2 = preprocessing_steps['1단계_단어수'] - preprocessing_steps['2단계_단어수']
                if word_diff_2 > 0:
                    st.success(f"✅ {word_diff_2}개 불용어가 제거되었습니다! (the, a, and, is 등)")
                else:
                    st.info("제거된 불용어가 없습니다.")
                
                st.info("💡 **불용어**: 'the', 'a', 'an', 'and', 'or', 'is', 'are' 등 분석에 도움이 되지 않는 단어들")
            
            # 3단계: 어간 추출
            with st.expander("✂️ 3단계: 어간 추출 (단어의 기본 형태로 변환)", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Before (불용어 제거):**")
                    st.write(f"단어 수: **{preprocessing_steps['2단계_단어수']}개**")
                    st.text_area("", preprocessing_steps['2단계_불용어제거'][:200] + "...", height=120, disabled=True, key="before_3")
                
                with col2:
                    st.write("**After (어간 추출):**")
                    st.write(f"단어 수: **{preprocessing_steps['3단계_단어수']}개**")
                    st.text_area("", preprocessing_steps['3단계_어간추출'][:200] + "...", height=120, disabled=True, key="after_3")
                
                # 어간 추출 변화 예시 보여주기
                if preprocessing_steps['2단계_불용어제거'] and preprocessing_steps['3단계_어간추출']:
                    from nltk.stem import PorterStemmer
                    stemmer = PorterStemmer()
                    
                    before_words = preprocessing_steps['2단계_불용어제거'].split()[:20]  # 처음 20개 단어만
                    stemmed_examples = []
                    
                    for word in before_words:
                        stemmed = stemmer.stem(word)
                        if word != stemmed:  # 변화된 단어만 표시
                            stemmed_examples.append(f"'{word}' → '{stemmed}'")
                    
                    if stemmed_examples:
                        st.success(f"✅ {len(stemmed_examples)}개 단어가 어간 추출되었습니다!")
                        st.write("**형태가 변화된 단어 예시:**")
                        # 최대 10개까지만 표시
                        for example in stemmed_examples[:10]:
                            st.write(f"• {example}")
                        if len(stemmed_examples) > 10:
                            st.write(f"... 외 {len(stemmed_examples) - 10}개 더")
                    else:
                        st.info("이 텍스트에서는 어간 추출로 변화된 단어가 없습니다.")
                
                st.info("💡 **어간 추출**: 'running' → 'run', 'studies' → 'studi', 'better' → 'better'")
                st.warning("⚠️ 어간 추출은 단순한 규칙을 사용해서 때로는 부정확할 수 있습니다.")
            
            # 4단계: 표제어 추출
            with st.expander("🎯 4단계: 표제어 추출 (사전 형태로 정확하게 변환)", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Before (불용어 제거):**")
                    st.write(f"단어 수: **{preprocessing_steps['2단계_단어수']}개**")
                    st.text_area("", preprocessing_steps['2단계_불용어제거'][:200] + "...", height=120, disabled=True, key="before_4")
                
                with col2:
                    st.write("**After (표제어 추출):**")
                    st.write(f"단어 수: **{preprocessing_steps['4단계_단어수']}개**")
                    st.text_area("", preprocessing_steps['4단계_표제어추출'][:200] + "...", height=120, disabled=True, key="after_4")
                
                # 표제어 추출 변화 예시 보여주기
                if preprocessing_steps['2단계_불용어제거'] and preprocessing_steps['4단계_표제어추출']:
                    from nltk.stem import WordNetLemmatizer
                    lemmatizer = WordNetLemmatizer()
                    
                    before_words = preprocessing_steps['2단계_불용어제거'].split()[:20]  # 처음 20개 단어만
                    lemmatized_examples = []
                    
                    for word in before_words:
                        lemmatized = lemmatizer.lemmatize(word)
                        if word != lemmatized:  # 변화된 단어만 표시
                            lemmatized_examples.append(f"'{word}' → '{lemmatized}'")
                    
                    if lemmatized_examples:
                        st.success(f"✅ {len(lemmatized_examples)}개 단어가 표제어 추출되었습니다!")
                        st.write("**형태가 변화된 단어 예시:**")
                        # 최대 10개까지만 표시
                        for example in lemmatized_examples[:10]:
                            st.write(f"• {example}")
                        if len(lemmatized_examples) > 10:
                            st.write(f"... 외 {len(lemmatized_examples) - 10}개 더")
                    else:
                        st.info("이 텍스트에서는 표제어 추출로 변화된 단어가 없습니다.")
                
                # 어간 추출 vs 표제어 추출 비교
                if preprocessing_steps['3단계_어간추출'] != preprocessing_steps['4단계_표제어추출']:
                    st.write("**🔍 어간 추출 vs 표제어 추출 비교:**")
                    from nltk.stem import PorterStemmer, WordNetLemmatizer
                    stemmer = PorterStemmer()
                    lemmatizer = WordNetLemmatizer()
                    
                    before_words = preprocessing_steps['2단계_불용어제거'].split()[:15]
                    comparison_examples = []
                    
                    for word in before_words:
                        stemmed = stemmer.stem(word)
                        lemmatized = lemmatizer.lemmatize(word)
                        if stemmed != lemmatized:  # 두 결과가 다른 경우만
                            comparison_examples.append(f"'{word}' → 어간: '{stemmed}' vs 표제어: '{lemmatized}'")
                    
                    if comparison_examples:
                        for example in comparison_examples[:5]:  # 최대 5개만
                            st.write(f"• {example}")
                
                st.info("💡 **표제어 추출**: 'running' → 'run', 'studies' → 'study', 'better' → 'better' (사전 기반으로 더 정확함)")
                st.success("✅ 표제어 추출이 어간 추출보다 더 정확한 결과를 제공합니다!")
            
            # 5단계: 개체명 인식 (NER)
            with st.expander("🏷️ 5단계: 개체명 인식 (NER) - 인물, 장소, 기관, 시간 추출", expanded=True):
                st.markdown("**🎯 개체명 인식이란?**")
                st.write("텍스트에서 특정 의미를 가진 고유명사들(인물명, 지명, 기관명, 시간 등)을 자동으로 찾아내는 기술입니다.")
                
                ner_results = preprocessing_steps['5단계_NER결과']
                total_entities = preprocessing_steps['5단계_개체수']
                
                if total_entities > 0:
                    st.success(f"✅ 총 **{total_entities}개**의 개체명이 발견되었습니다!")
                    
                    # 개체명 유형별 표시
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("👤 인물명", len(ner_results['PERSON']))
                        if ner_results['PERSON']:
                            for person in ner_results['PERSON'][:5]:  # 최대 5개만 표시
                                st.write(f"• {person}")
                            if len(ner_results['PERSON']) > 5:
                                st.write(f"... 외 {len(ner_results['PERSON']) - 5}개 더")
                    
                    with col2:
                        st.metric("🗺️ 장소명", len(ner_results['PLACE']))
                        if ner_results['PLACE']:
                            for place in ner_results['PLACE'][:5]:  # 최대 5개만 표시
                                st.write(f"• {place}")
                            if len(ner_results['PLACE']) > 5:
                                st.write(f"... 외 {len(ner_results['PLACE']) - 5}개 더")
                    
                    with col3:
                        st.metric("🏢 기관명", len(ner_results['ORG']))
                        if ner_results['ORG']:
                            for org in ner_results['ORG'][:5]:  # 최대 5개만 표시
                                st.write(f"• {org}")
                            if len(ner_results['ORG']) > 5:
                                st.write(f"... 외 {len(ner_results['ORG']) - 5}개 더")
                    
                    with col4:
                        st.metric("⏰ 시간 표현", len(ner_results['TIME']))
                        if ner_results['TIME']:
                            for time in ner_results['TIME'][:5]:  # 최대 5개만 표시
                                st.write(f"• {time}")
                            if len(ner_results['TIME']) > 5:
                                st.write(f"... 외 {len(ner_results['TIME']) - 5}개 더")
                    
                    # 분석 의미 해석
                    st.markdown("**📊 분석 해석:**")
                    interpretation = []
                    
                    if len(ner_results['PERSON']) > 0:
                        interpretation.append(f"• **인물 언급**: {len(ner_results['PERSON'])}명의 인물을 구체적으로 언급하여 내용의 신뢰성을 높였습니다.")
                    
                    if len(ner_results['PLACE']) > 0:
                        interpretation.append(f"• **장소 정보**: {len(ner_results['PLACE'])}곳의 구체적 장소를 제시하여 상황을 명확히 했습니다.")
                    
                    if len(ner_results['ORG']) > 0:
                        interpretation.append(f"• **기관 정보**: {len(ner_results['ORG'])}개 기관을 언급하여 전문성과 객관성을 나타냈습니다.")
                    
                    if len(ner_results['TIME']) > 0:
                        interpretation.append(f"• **시간 정보**: {len(ner_results['TIME'])}개의 시간 표현으로 시간적 맥락을 제공했습니다.")
                    
                    if interpretation:
                        for interp in interpretation:
                            st.write(interp)
                    
                    # 구체성 점수 계산
                    specificity_score = min(100, total_entities * 10)  # 개체명 개수 × 10점, 최대 100점
                    
                    if specificity_score >= 70:
                        st.success(f"🌟 **구체성 점수: {specificity_score}점** - 매우 구체적이고 풍부한 정보를 담고 있습니다!")
                    elif specificity_score >= 40:
                        st.info(f"📈 **구체성 점수: {specificity_score}점** - 적절한 수준의 구체적 정보가 있습니다.")
                    else:
                        st.warning(f"💡 **구체성 점수: {specificity_score}점** - 더 구체적인 인물, 장소, 기관명을 언급하면 글의 설득력이 높아집니다.")
                
                else:
                    st.info("🔍 이 텍스트에서는 명확한 개체명이 발견되지 않았습니다.")
                    st.write("💡 **개선 제안**: 구체적인 인물명, 장소명, 기관명을 언급하면 글의 신뢰성과 설득력이 높아집니다.")
                
                st.info("""
                💡 **개체명 인식 기술:**
                - **패턴 기반**: 정규표현식으로 특정 패턴(대문자 연속, 키워드 조합 등)을 탐지
                - **기계학습**: 대량의 데이터로 학습된 모델이 문맥을 고려해 개체명 분류
                - **실제 활용**: Google 검색, 뉴스 분류, 문서 요약, 정보 추출 등에서 핵심 기술
                """)
            
            # 요약 통계
            st.markdown("---")
            st.subheader("📊 전처리 요약 통계")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "원본", 
                    f"{preprocessing_steps['원본_단어수']}개",
                    help="원본 텍스트의 단어 수"
                )
            
            with col2:
                diff_1 = preprocessing_steps['1단계_단어수'] - preprocessing_steps['원본_단어수']
                st.metric(
                    "기본 정제", 
                    f"{preprocessing_steps['1단계_단어수']}개",
                    f"{diff_1:+d}개",
                    help="HTML 태그, 특수문자 제거 후"
                )
            
            with col3:
                diff_2 = preprocessing_steps['2단계_단어수'] - preprocessing_steps['1단계_단어수']
                st.metric(
                    "불용어 제거", 
                    f"{preprocessing_steps['2단계_단어수']}개",
                    f"{diff_2:+d}개",
                    help="불용어 제거 후"
                )
            
            with col4:
                diff_4 = preprocessing_steps['4단계_단어수'] - preprocessing_steps['2단계_단어수']
                st.metric(
                    "표제어 추출", 
                    f"{preprocessing_steps['4단계_단어수']}개",
                    f"{diff_4:+d}개" if diff_4 != 0 else "변화없음",
                    help="최종 전처리 완료 (불용어 제거 후 기준)"
                )
            
            with col5:
                st.metric(
                    "🏷️ 개체명", 
                    f"{preprocessing_steps['5단계_개체수']}개",
                    help="인물, 장소, 기관, 시간 등 고유명사"
                )
            
            # 세션에 최종 결과 저장
            st.session_state.final_preprocessed_text = preprocessing_steps['4단계_표제어추출']
            
            # 전체 요약
            total_reduction = preprocessing_steps['원본_단어수'] - preprocessing_steps['4단계_단어수']
            st.success(f"🎉 모든 전처리 단계가 완료되었습니다! 총 {total_reduction}개 단어가 정제/변환되었습니다.")
    
    # 다음 단계 안내
    if 'final_preprocessed_text' in st.session_state:
        st.markdown("---")
        st.info("✅ 전처리가 완료되었습니다! 이제 단어 빈도 분석을 진행해보세요.")
        
        # 빈도 분석
        if st.button("📊 단어 빈도 분석 시작"):
            final_text = st.session_state.final_preprocessed_text
            
            if final_text:
                from collections import Counter
                words = final_text.split()
                word_freq = Counter(words)
                top_words = word_freq.most_common(20)
                
                if top_words:
                    st.subheader("📈 내가 가장 많이 사용한 단어 TOP 20")
                    
                    words_list, freq_list = zip(*top_words)
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(freq_list), 
                            y=list(words_list), 
                            orientation='h',
                            marker_color='lightcoral',
                            text=[f'{freq}회' for freq in freq_list],
                            textposition='outside'
                        )
                    ])
                    
                    fig.update_layout(
                        title="전처리 후 단어 빈도 분석 결과",
                        xaxis_title="사용 빈도",
                        yaxis_title="단어",
                        height=500,
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 빈도 분석 인사이트
                    st.subheader("🔍 분석 인사이트")
                    most_used_word = top_words[0]
                    st.write(f"• 가장 많이 사용한 단어: **'{most_used_word[0]}'** ({most_used_word[1]}회)")
                    st.write(f"• 총 고유 단어 수: **{len(word_freq)}개**")
                    st.write(f"• 전처리 후 총 단어 수: **{len(words)}개**")
                    
                    st.session_state.word_freq = word_freq
                    st.success("✅ 빈도 분석이 완료되었습니다!")
    
    # 워드클라우드
    if 'word_freq' in st.session_state:
        st.markdown("---")
        st.subheader("☁️ 워드클라우드 생성")
        st.write("전처리된 단어들을 시각적으로 표현해보겠습니다.")
        
        if st.button("워드클라우드 생성"):
            try:
                from wordcloud import WordCloud
                import matplotlib.pyplot as plt
                
                word_freq = st.session_state.word_freq
                
                if word_freq:
                    # 영어 단어만 필터링 + 사용자명 제외
                    english_only_freq = {}
                    exclude_words = {username.lower(), 'y11111', 'test', 'user'}  # 사용자명 관련 단어들 제외
                    
                    for word, freq in word_freq.items():
                        if (word.isascii() and 
                            word.isalpha() and 
                            len(word) > 2 and 
                            word.lower() not in exclude_words):  # 사용자명 제외
                            english_only_freq[word] = freq
                    
                    if english_only_freq and len(english_only_freq) >= 5:
                        # 워드클라우드 설정 개선
                        wordcloud = WordCloud(
                            width=400,  # 더 작게
                            height=200,  # 더 작게
                            background_color='white',
                            max_words=20,
                            colormap='tab10',  # 더 선명한 색상
                            relative_scaling=1.0,  # 크기 차이 극대화
                            prefer_horizontal=0.7,
                            min_font_size=12,   # 최소 크기 증가
                            max_font_size=80,   # 최대 크기 증가
                            font_step=4,        # 크기 단계 증가
                            collocations=False, # 단어 조합 방지
                            margin=10
                        ).generate_from_frequencies(english_only_freq)
                        
                        # 컨테이너 크기 제한
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            fig, ax = plt.subplots(figsize=(6, 3))  # 더 작게
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            ax.set_title(f"{username}'s Most Used Words", 
                                       fontsize=12, fontweight='bold', pad=10)
                            
                            # 여백 제거
                            plt.tight_layout(pad=0.5)
                            st.pyplot(fig, use_container_width=False)
                            plt.close()
                        
                        st.success("🎉 워드클라우드가 생성되었습니다!")
                        st.info("💡 빈도가 높은 단어일수록 크게 표시됩니다.")
                    else:
                        st.warning("워드클라우드 생성에 충분한 영어 단어가 없습니다.")
                
                # 전체 실습 완료
                st.markdown("---")
                st.success("🎉 모든 텍스트 마이닝 실습이 완료되었습니다!")
                
            except ImportError:
                st.error("워드클라우드 라이브러리를 설치해주세요: pip install wordcloud")
            except Exception as e:
                st.error(f"워드클라우드 생성 오류: {e}")

def show_comprehensive_analysis(essay_data, preprocessor, username):
    """종합 분석"""
    st.header("🎓 종합 분석")
    st.markdown("""
    **텍스트 마이닝의 핵심 원리들을 단계별로 체험해보세요!**  
    각 분석은 독립적으로 실행할 수 있으며, 다양한 AI 기술의 작동 원리를 이해할 수 있습니다.
    """)
    
    if essay_data.empty:
        st.warning("분석할 에세이가 없습니다.")
        return
    
    # 전체 에세이 텍스트 합치기
    all_essays_text = ""
    for _, row in essay_data.iterrows():
        essay_text = row.get('essay_text', '')
        if essay_text and not pd.isna(essay_text):
            cleaned_text = preprocessor.extract_essay_content(essay_text)
            if cleaned_text:
                all_essays_text += cleaned_text + " "
    
    if not all_essays_text.strip():
        st.warning("분석할 텍스트가 없습니다.")
        return
    
    st.info(f"📚 **분석 대상**: {len(essay_data)}개 에세이의 통합 텍스트 (총 {len(all_essays_text.split())}개 단어)")
    
    # 체험 선택 탭
    analysis_tabs = st.tabs([
        "😊 감성 분석 원리 체험", 
        "📝 품사 분석 원리 체험", 
        "🏆 글쓰기 수준 종합 진단"
    ])
    
    # 1. 감성 분석 탭
    with analysis_tabs[0]:
        st.subheader("😊 감성 분석 4단계 원리 체험")
        st.markdown("감성 분석의 4가지 접근법을 직접 체험하며 텍스트 마이닝의 원리를 이해해보세요.")
        
        if st.button("😊 4단계 감성 분석 원리 체험 시작", key="educational_sentiment"):
            with st.spinner("감성 분석 원리를 단계별로 분석하는 중..."):
                # 통합 텍스트로 교육적 감성 분석 실행
                
                # 1단계: 어휘 사전 기반
                sentiment_method1 = preprocessor.educational_sentiment_analysis_step1_lexicon(all_essays_text)
                
                # 2단계: TF-IDF + 머신러닝
                sentiment_method2 = preprocessor.educational_sentiment_analysis_step2_tfidf(all_essays_text, [all_essays_text])
                
                # 3단계: VADER
                sentiment_method3 = preprocessor.educational_sentiment_analysis_step3_vader(all_essays_text)
                
                # 4단계: 다중 감성 분석
                sentiment_method4 = preprocessor.educational_sentiment_analysis_step4_emotions(all_essays_text)
                
                sentiment_comparison_result = {
                    'essay_info': {
                        'topic': f"{username}님의 모든 에세이 통합 텍스트",
                        'text_preview': all_essays_text[:200] + "...",
                        'total_words': len(all_essays_text.split()),
                        'total_essays': len(essay_data)
                    },
                    'method1_lexicon': sentiment_method1,
                    'method2_tfidf': sentiment_method2,
                    'method3_vader': sentiment_method3,
                    'method4_emotions': sentiment_method4
                }
                
                if sentiment_comparison_result:
                    # 에세이 정보
                    essay_info = sentiment_comparison_result['essay_info']
                    st.subheader(f"📝 감성 분석 대상: {essay_info['topic']}")
                    st.write(f"• 총 에세이 수: {essay_info['total_essays']}개")
                    st.write(f"• 총 단어 수: {essay_info['total_words']}개")
                    
                    st.markdown("**분석할 텍스트 미리보기:**")
                    st.text_area("", essay_info['text_preview'], height=100, disabled=True, key="sentiment_preview_text")
                    
                    st.markdown("---")
                    
                    # 1단계: 어휘 사전 기반 감성 분석
                    method1 = sentiment_comparison_result['method1_lexicon']
                    
                    st.markdown("## 📚 1단계: 어휘 사전 기반 감성 분석")
                    st.markdown("""
                    **🔍 원리 설명:**
                    - **감성 어휘 사전**에서 각 단어별 감성 점수를 조회
                    - **단순 합산 방식**으로 전체 텍스트의 감성 계산
                    - **빠르고 직관적**이지만 문맥을 고려하지 못하는 한계
                    """)
                    
                    if 'error' not in method1:
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            # 감성 점수와 결과 (수정된 키 사용)
                            sentiment_score = method1.get('total_score', 0)
                            sentiment_label = method1.get('sentiment', 'Unknown')
                            emoji = method1.get('emoji', '')
                            
                            if '긍정' in sentiment_label:
                                st.success(f"**감성 결과**: {sentiment_label} {emoji}")
                            elif '부정' in sentiment_label:
                                st.error(f"**감성 결과**: {sentiment_label} {emoji}")
                            else:
                                st.info(f"**감성 결과**: {sentiment_label} {emoji}")
                            
                            st.metric("감성 점수", f"{sentiment_score}")
                            
                            # 실제 찾은 단어들 표시
                            positive_words = method1.get('positive_words_found', [])
                            negative_words = method1.get('negative_words_found', [])
                            st.write(f"• 긍정 단어: {len(positive_words)}개")
                            st.write(f"• 부정 단어: {len(negative_words)}개")
                        
                        with col2:
                            # 감성 분포 시각화 (수정된 데이터 사용)
                            positive_words = method1.get('positive_words_found', [])
                            negative_words = method1.get('negative_words_found', [])
                            positive_count = len(positive_words)
                            negative_count = len(negative_words)
                            
                            if positive_count + negative_count > 0:
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=['긍정', '부정'],
                                        y=[positive_count, negative_count],
                                        marker_color=['#28a745', '#dc3545'],
                                        text=[positive_count, negative_count],
                                        textposition='auto'
                                    )
                                ])
                                
                                fig.update_layout(
                                    title="어휘 사전 기반 감성 분포",
                                    xaxis_title="감성 범주",
                                    yaxis_title="단어 개수",
                                    height=300
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("감성 단어가 발견되지 않았습니다.")
                        
                        # 감성 단어 예시 (수정된 데이터 사용)
                        positive_words = method1.get('positive_words_found', [])
                        negative_words = method1.get('negative_words_found', [])
                        
                        if positive_words or negative_words:
                            st.write("**발견된 감성 단어 예시:**")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if positive_words:
                                    st.success("**긍정 단어들**")
                                    for word, score in positive_words[:5]:
                                        st.write(f"• '{word}': +{score}점")
                                    if len(positive_words) > 5:
                                        st.write(f"... 외 {len(positive_words) - 5}개 더")
                            
                            with col2:
                                if negative_words:
                                    st.error("**부정 단어들**")
                                    for word, score in negative_words[:5]:
                                        st.write(f"• '{word}': {score}점")
                                    if len(negative_words) > 5:
                                        st.write(f"... 외 {len(negative_words) - 5}개 더")
                    else:
                        st.warning(method1.get('error', ''))
                    
                    st.markdown("""
                    **✅ 장점:** 빠른 처리, 이해하기 쉬움, 구현 간단  
                    **❌ 한계:** 문맥 무시, 복합 감정 처리 어려움, 사전 의존성
                    """)
                    
                    st.markdown("---")
                    
                    # 2단계: TF-IDF + 머신러닝 기반 감성 분석
                    method2 = sentiment_comparison_result['method2_tfidf']
                    
                    st.markdown("## 🤖 2단계: TF-IDF + 머신러닝 기반 감성 분석")
                    st.markdown("""
                    **🔍 원리 설명:**
                    - **TF-IDF**로 단어의 중요도를 계산하여 가중치 부여
                    - **머신러닝 모델**이 패턴을 학습하여 감성 분류
                    - **통계적 접근**으로 더 정확하고 세밀한 분석 가능
                    """)
                    
                    if 'error' not in method2:
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            # TF-IDF 기반 감성 결과 (수정된 키 사용)
                            tfidf_sentiment = method2.get('sentiment', 'Unknown')
                            final_score = method2.get('final_score', 0)
                            emoji = method2.get('emoji', '')
                            
                            if '긍정' in tfidf_sentiment:
                                st.success(f"**TF-IDF 감성 결과**: {tfidf_sentiment} {emoji}")
                            elif '부정' in tfidf_sentiment:
                                st.error(f"**TF-IDF 감성 결과**: {tfidf_sentiment} {emoji}")
                            else:
                                st.info(f"**TF-IDF 감성 결과**: {tfidf_sentiment} {emoji}")
                            
                            # 점수 기반으로 신뢰도 계산
                            confidence = min(abs(final_score) * 30, 100)  # 점수를 퍼센트로 변환
                            st.metric("신뢰도", f"{confidence:.1f}%")
                            st.metric("TF-IDF 점수", f"{final_score:.3f}")
                        
                        with col2:
                            # 주요 TF-IDF 단어들
                            top_words = method2.get('top_tfidf_words', [])
                            if top_words:
                                st.write("**주요 TF-IDF 단어들:**")
                                tfidf_data = []
                                for word, score in top_words[:8]:
                                    # numpy 타입 처리
                                    score_val = float(score) if hasattr(score, 'item') else score
                                    tfidf_data.append({'단어': word, 'TF-IDF 점수': f"{score_val:.4f}"})
                                
                                df_tfidf = pd.DataFrame(tfidf_data)
                                st.dataframe(df_tfidf, use_container_width=True)
                            
                            # 감성 점수 백분율
                            positive_score = method2.get('positive_score', 0)
                            negative_score = method2.get('negative_score', 0)
                            st.write(f"**감성 점수 분해:**")
                            st.write(f"• 긍정 가중치: {positive_score:.3f}")
                            st.write(f"• 부정 가중치: {negative_score:.3f}")
                        
                        # TF-IDF 점수 시각화
                        if top_words:
                            words = [word for word, score in top_words[:10]]
                            # numpy 타입 처리
                            scores = [float(score) if hasattr(score, 'item') else score for word, score in top_words[:10]]
                            
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=words,
                                    y=scores,
                                    marker_color='#17a2b8',
                                    text=[f"{score:.3f}" for score in scores],
                                    textposition='auto'
                                )
                            ])
                            
                            fig.update_layout(
                                title="주요 단어별 TF-IDF 점수",
                                xaxis_title="단어",
                                yaxis_title="TF-IDF 점수",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.warning(method2.get('error', ''))
                    
                    st.markdown("""
                    **✅ 장점:** 높은 정확도, 단어 중요도 고려, 통계적 신뢰성  
                    **❌ 한계:** 계산 복잡도, 데이터 의존성, 문맥 제한적 이해
                    """)
                    
                    st.markdown("---")
                    
                    # 3단계: VADER 감성 분석
                    method3 = sentiment_comparison_result['method3_vader']
                    
                    st.markdown("## 🎯 3단계: VADER 고급 규칙 기반 감성 분석")
                    st.markdown("""
                    **🔍 원리 설명:**
                    - **문맥과 강도**를 고려한 고급 규칙 기반 분석
                    - **감정 강화어, 부정어, 구두점** 등을 종합적으로 분석
                    - **실시간 소셜미디어** 텍스트에 최적화된 최신 기법
                    """)
                    
                    if 'error' not in method3:
                        # VADER 감성 점수 (수정된 키 사용)
                        compound_score = method3.get('compound', 0)
                        detailed_scores = method3.get('detailed_scores', {})
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.metric("VADER 복합 점수", f"{compound_score:.3f}")
                            
                            # 감성 결과
                            vader_sentiment = method3.get('sentiment', 'Unknown')
                            emoji = method3.get('emoji', '')
                            if '긍정' in vader_sentiment:
                                st.success(f"**VADER 감성 결과**: {vader_sentiment} {emoji}")
                            elif '부정' in vader_sentiment:
                                st.error(f"**VADER 감성 결과**: {vader_sentiment} {emoji}")
                            else:
                                st.info(f"**VADER 감성 결과**: {vader_sentiment} {emoji}")
                        
                        with col2:
                            st.write("**세부 감성 점수:**")
                            st.write(f"• 긍정 (Positive): {detailed_scores.get('positive', 0):.3f}")
                            st.write(f"• 중립 (Neutral): {detailed_scores.get('neutral', 0):.3f}")
                            st.write(f"• 부정 (Negative): {detailed_scores.get('negative', 0):.3f}")
                        
                        # VADER 세부 점수 시각화
                        categories = ['긍정', '중립', '부정']
                        values = [detailed_scores.get('positive', 0), detailed_scores.get('neutral', 0), detailed_scores.get('negative', 0)]
                        colors = ['#28a745', '#6c757d', '#dc3545']
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=categories,
                                y=values,
                                marker_color=colors,
                                text=[f"{val:.3f}" for val in values],
                                textposition='auto'
                            )
                        ])
                        
                        fig.update_layout(
                            title="VADER 세부 감성 점수 분포",
                            xaxis_title="감성 범주",
                            yaxis_title="점수 (0-1)",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.warning(method3.get('error', ''))
                    
                    st.markdown("""
                    **✅ 장점:** 문맥 고려, 감정 강도 측정, 실시간 처리 가능  
                    **❌ 한계:** 언어별 튜닝 필요, 도메인 특화 어려움
                    """)
                    
                    st.markdown("---")
                    
                    # 4단계: 다중 감성 분석 (8가지 기본 감정)
                    method4 = sentiment_comparison_result['method4_emotions']
                    
                    st.markdown("## 🌈 4단계: 다중 감성 분석 - 8가지 기본 감정 탐지")
                    st.markdown("""
                    **🔍 원리 설명:**
                    - **8가지 기본 감정**을 각각 독립적으로 분석 (기쁨, 슬픔, 분노, 두려움, 놀람, 혐오, 신뢰, 기대)
                    - **감정별 키워드 사전**을 활용한 다차원 감정 분석
                    - **감정의 강도와 다양성**까지 정량적으로 측정
                    """)
                    
                    if 'error' not in method4:
                        # 주도적 감정과 전체 통계
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            dominant_emotion = method4.get('dominant_emotion', '중립')
                            dominant_emoji = method4.get('dominant_emoji', '😐')
                            st.success(f"**주도적 감정**: {dominant_emotion} {dominant_emoji}")
                            
                        with col2:
                            emotion_intensity = method4.get('emotion_intensity', 0)
                            st.metric("감정 강도", f"{emotion_intensity}%")
                            
                        with col3:
                            emotion_variety = method4.get('emotion_variety', 0)
                            st.metric("감정 다양성", f"{emotion_variety}가지")
                        
                        # 8가지 감정별 상세 결과
                        st.markdown("### 🎭 감정별 상세 분석")
                        emotion_details = method4.get('emotion_details', {})
                        
                        if emotion_details:
                            # 2행 4열로 감정 표시
                            row1_cols = st.columns(4)
                            row2_cols = st.columns(4)
                            
                            emotions_list = list(emotion_details.items())
                            
                            # 첫 번째 행 (기쁨, 슬픔, 분노, 두려움)
                            for i, col in enumerate(row1_cols):
                                if i < len(emotions_list):
                                    emotion_name, details = emotions_list[i]
                                    with col:
                                        score = details.get('score', 0)
                                        emoji = details.get('emoji', '😐')
                                        found_words = details.get('found_words', [])
                                        
                                        st.metric(f"{emoji} {emotion_name.split('(')[0].strip()}", f"{score}개")
                                        if found_words:
                                            st.write("**발견된 단어:**")
                                            for word in found_words[:3]:  # 최대 3개만
                                                st.write(f"• {word}")
                                        else:
                                            st.write("발견된 단어 없음")
                            
                            # 두 번째 행 (놀람, 혐오, 신뢰, 기대)
                            for i, col in enumerate(row2_cols):
                                emotion_index = i + 4
                                if emotion_index < len(emotions_list):
                                    emotion_name, details = emotions_list[emotion_index]
                                    with col:
                                        score = details.get('score', 0)
                                        emoji = details.get('emoji', '😐')
                                        found_words = details.get('found_words', [])
                                        
                                        st.metric(f"{emoji} {emotion_name.split('(')[0].strip()}", f"{score}개")
                                        if found_words:
                                            st.write("**발견된 단어:**")
                                            for word in found_words[:3]:  # 최대 3개만
                                                st.write(f"• {word}")
                                        else:
                                            st.write("발견된 단어 없음")
                        
                        # 문장별 감정 분석 결과
                        sentence_emotions = method4.get('sentence_emotions', [])
                        if sentence_emotions:
                            st.markdown("### 📝 문장별 감정 분석")
                            for i, sent_emotion in enumerate(sentence_emotions, 1):
                                sentence = sent_emotion.get('sentence', '')
                                emotion = sent_emotion.get('emotion', '중립')
                                emoji = sent_emotion.get('emoji', '😐')
                                score = sent_emotion.get('score', 0)
                                
                                st.write(f"**문장 {i}:** {sentence}")
                                st.write(f"→ {emoji} **{emotion}** (감정 단어 {score}개)")
                                st.write("")
                        
                        # 해석 및 조언
                        interpretations = method4.get('interpretation', [])
                        if interpretations:
                            st.markdown("### 💡 감정 분석 해석")
                            for interp in interpretations:
                                st.write(f"• {interp}")
                    
                    else:
                        st.error(f"다중 감성 분석 오류: {method4.get('error', '알 수 없는 오류')}")
                    
                    st.info("""
                    **✅ 장점:** 세밀한 감정 분류, 감정 강도 측정, 다양성 분석 가능  
                    **❌ 한계:** 키워드 기반 한계, 복합 감정 처리 어려움, 문맥 의존성
                    """)
                    
                    # 결과 비교 및 학습 정리
                    st.markdown("---")
                    st.subheader("📊 4가지 감성 분석 방법 결과 비교")
                    
                    # 비교 테이블
                    comparison_data = []
                    
                    if method1 and 'error' not in method1:
                        comparison_data.append({
                            '방법': '1. 어휘 사전 기반',
                            '감성 결과': method1.get('sentiment', 'Unknown'),
                            '점수/신뢰도': f"{method1.get('total_score', 0)}",
                            '처리 속도': '빠름',
                            '정확도': '보통'
                        })
                    
                    if method2 and 'error' not in method2:
                        final_score = method2.get('final_score', 0)
                        confidence = min(abs(final_score) * 30, 100)
                        comparison_data.append({
                            '방법': '2. TF-IDF + ML',
                            '감성 결과': method2.get('sentiment', 'Unknown'),
                            '점수/신뢰도': f"{confidence:.1f}%",
                            '처리 속도': '보통',
                            '정확도': '높음'
                        })
                    
                    if method3 and 'error' not in method3:
                        comparison_data.append({
                            '방법': '3. VADER 고급',
                            '감성 결과': method3.get('sentiment', 'Unknown'),
                            '점수/신뢰도': f"{method3.get('compound', 0):.3f}",
                            '처리 속도': '빠름',
                            '정확도': '매우 높음'
                        })
                    
                    if method4 and 'error' not in method4:
                        comparison_data.append({
                            '방법': '4. 다중 감성',
                            '감성 결과': method4.get('dominant_emotion', 'Unknown'),
                            '점수/신뢰도': f"{method4.get('emotion_intensity', 0)}%",
                            '처리 속도': '보통',
                            '정확도': '세밀함'
                        })
                    
                    if comparison_data:
                        df_sentiment_comparison = pd.DataFrame(comparison_data)
                        st.table(df_sentiment_comparison)
                    
                    # 학습 정리
                    st.subheader("🎓 감성 분석 학습 정리")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info("""
                        **🔍 배운 내용:**
                        - 어휘 사전 기반의 단순 합산 방식
                        - TF-IDF를 활용한 가중치 기반 분석
                        - VADER의 문맥 고려 고급 규칙
                        - 다중 감성으로 세밀한 감정 분류 체험
                        - 각 방법론의 장단점과 적용 분야
                        """)
                    
                    with col2:
                        st.success("""
                        **💡 실무 적용:**
                        - 빠른 분석: 어휘 사전 기반
                        - 정확한 분류: TF-IDF + 머신러닝
                        - 소셜미디어: VADER
                        - 세밀한 분석: 다중 감성
                        - 종합 분석: 여러 방법 조합 활용
                        """)
                    
                    # 세션에 결과 저장
                    st.session_state.educational_sentiment_results = sentiment_comparison_result
                    
                    st.success("✅ 텍스트 마이닝 감성 분석 원리 체험이 완료되었습니다!")
                
                else:
                    st.error("감성 분석 결과를 생성할 수 없습니다.")
    
    # 2. 품사 분석 탭
    with analysis_tabs[1]:
        st.subheader("📝 품사 분석 3단계 원리 체험")
        st.markdown("형태소 분석의 3가지 접근법을 단계별로 체험하며 언어처리 원리를 이해해보세요.")
        
        # 항상 표시되도록 변경
        with st.spinner("품사 분석 원리를 단계별로 분석하는 중..."):
            try:
                # 통합 텍스트로 교육적 품사 분석 실행
                
                # 1단계: 수동 규칙 기반
                pos_method1 = preprocessor.educational_pos_analysis_step1_manual_rules(all_essays_text)
                st.success("✅ 1단계 성공!")
            except Exception as e:
                st.error(f"❌ 1단계 에러: {type(e).__name__}: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                pos_method1 = {'error': str(e)}
                
        # 2단계: NLTK 기본 품사 태깅
        try:
            pos_method2 = preprocessor.educational_pos_analysis_step2_nltk_basic(all_essays_text)
            st.success("✅ 2단계 성공!")
        except Exception as e:
            st.error(f"❌ 2단계 에러: {type(e).__name__}: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            pos_method2 = {'error': str(e)}
        
        # 3단계: 패턴 발견
        try:
            pos_method3 = preprocessor.educational_pos_analysis_step3_pattern_discovery(all_essays_text)
            st.success("✅ 3단계 성공!")
        except Exception as e:
            st.error(f"❌ 3단계 에러: {type(e).__name__}: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            pos_method3 = {'error': str(e)}
                
        pos_comparison_result = {
            'essay_info': {
                'topic': f"{username}님의 모든 에세이 통합 텍스트",
                'text_preview': all_essays_text[:200] + "...",
                'total_words': len(all_essays_text.split()),
                'total_essays': len(essay_data)
            },
            'method1_manual': pos_method1,
            'method2_nltk': pos_method2,
            'method3_patterns': pos_method3
        }
                
        if pos_comparison_result:
            # 에세이 정보
            essay_info = pos_comparison_result['essay_info']
            st.subheader(f"📝 품사 분석 대상: {essay_info['topic']}")
            st.write(f"• 총 에세이 수: {essay_info['total_essays']}개")
            st.write(f"• 총 단어 수: {essay_info['total_words']}개")
            
            st.markdown("**분석할 텍스트 미리보기:**")
            st.text_area("", essay_info['text_preview'], height=100, disabled=True, key="pos_preview_text")
            
            st.markdown("---")
            
            # 상세 UI 표시
            essay_info = {
                'topic': f"{username}님의 모든 에세이 통합 텍스트",
                'text_preview': all_essays_text[:200] + "...",
                'total_words': len(all_essays_text.split()),
                'total_essays': len(essay_data)
            }
            
            # 1단계: 수동 규칙 기반
            st.markdown("## 📖 1단계: 수동 규칙 기반 품사 분석")
            st.markdown("""
            **🔍 원리 설명:**
            - **단어 패턴**과 **어미**를 기반으로 품사 분류
            - **규칙 기반** 접근법으로 명확하고 예측 가능
            - **빠른 처리**가 가능하지만 예외 상황 처리 어려움
            """)
            
            
            if pos_method1 and 'error' not in pos_method1:
                pos_counts = pos_method1.get('pos_counts', {})
                if pos_counts:
                    # 품사별 차트
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(pos_counts.keys()),
                            y=list(pos_counts.values()),
                            marker_color='lightblue',
                            text=list(pos_counts.values()),
                            textposition='auto'
                        )
                    ])
                    fig.update_layout(
                        title="1단계: 수동 규칙 기반 품사 분포",
                        xaxis_title="품사",
                        yaxis_title="단어 개수",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 품사별 비율 파이 차트 추가
                    pos_ratios = pos_method1.get('pos_ratios', {})
                    if pos_ratios:
                        st.markdown("**📊 품사별 비율 분포:**")
                        
                        # 0이 아닌 값만 파이 차트에 표시
                        filtered_ratios = {k: v for k, v in pos_ratios.items() if v > 0}
                        
                        if filtered_ratios:
                            pie_fig = go.Figure(data=[go.Pie(
                                labels=list(filtered_ratios.keys()),
                                values=list(filtered_ratios.values()),
                                hole=0.3,  # 도넛 차트로 만들기
                                textinfo='label+percent',
                                textfont_size=12,
                                marker=dict(
                                    colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
                                    line=dict(color='#FFFFFF', width=2)
                                )
                            )])
                            
                            pie_fig.update_layout(
                                title="1단계: 품사별 비율 분포",
                                height=400,
                                showlegend=True,
                                legend=dict(
                                    orientation="v",
                                    yanchor="middle",
                                    y=0.5,
                                    xanchor="left",
                                    x=1.01
                                )
                            )
                            st.plotly_chart(pie_fig, use_container_width=True)
                            
                            # 상세 통계 표시
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("총 분석 단어", f"{pos_method1.get('total_words', 0)}개")
                            with col2:
                                most_common = max(filtered_ratios, key=filtered_ratios.get)
                                st.metric("가장 많은 품사", f"{most_common} ({filtered_ratios[most_common]:.1f}%)")
                            with col3:
                                coverage = 100 - filtered_ratios.get('기타', 0)
                                st.metric("품사 인식률", f"{coverage:.1f}%")
                
                st.success("✅ 1단계 수동 규칙 기반 품사 분석 완료!")
            else:
                st.warning("1단계 분석에 문제가 발생했습니다.")
            
            st.markdown("---")
            
            # 2단계: NLTK 기반
            st.markdown("## 🤖 2단계: NLTK 머신러닝 기반 품사 분석")
            st.markdown("""
            **🔍 원리 설명:**
            - **통계 모델**과 **기계학습**을 활용한 품사 태깅
            - **문맥 정보**를 고려하여 더 정확한 분류
            - **대용량 데이터**로 훈련된 모델 사용
            """)
            
            
            if pos_method2 and 'error' not in pos_method2:
                st.success("✅ 2단계 NLTK 기계학습 기반 품사 분석 완료!")
                pos_tagged = pos_method2.get('pos_tagged_words', [])
                if pos_tagged:
                    # Before/After 비교 UI 추가
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🔴 변환 전 (원본 텍스트):**")
                        sample_words = [word for word, pos in pos_tagged[:10]]
                        for i, word in enumerate(sample_words):
                            st.write(f"{i+1}. {word}")
                    
                    with col2:
                        st.markdown("**🔵 변환 후 (품사 태깅):**")
                        for i, (word, pos) in enumerate(pos_tagged[:10]):
                            # 품사별 색상 코딩
                            if pos.startswith('NN'):  # 명사
                                color = "🟦"
                            elif pos.startswith('VB'):  # 동사
                                color = "🟩"
                            elif pos.startswith('JJ'):  # 형용사
                                color = "🟨"
                            elif pos.startswith('RB'):  # 부사
                                color = "🟪"
                            elif pos in ['DT', 'IN', 'TO']:  # 관사, 전치사
                                color = "🟫"
                            else:
                                color = "⚪"
                            st.write(f"{i+1}. {color} {word} → **{pos}**")
                    
                    # 품사 변화 통계
                    pos_counts = {}
                    for word, pos in pos_tagged:
                        pos_counts[pos] = pos_counts.get(pos, 0) + 1
                    
                    st.markdown("**📊 품사별 분포 (상위 8개):**")
                    top_pos = sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)[:8]
                    
                    # 품사 분포 차트
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[pos for pos, count in top_pos],
                            y=[count for pos, count in top_pos],
                            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
                                        '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'][:len(top_pos)],
                            text=[f"{count}개" for pos, count in top_pos],
                            textposition='auto'
                        )
                    ])
                    fig.update_layout(
                        title="2단계: NLTK 기반 품사 분포 (상위 8개)",
                        xaxis_title="품사 태그",
                        yaxis_title="개수",
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 체험형 인터랙티브 요소
                    st.markdown("**🎯 체험해보기: 직접 문장 입력하여 품사 분석**")
                    user_sentence = st.text_input("분석할 문장을 영어로 입력하세요:", 
                                                 value="I love studying English because it helps me grow.",
                                                 placeholder="예: I enjoy reading books in the library.",
                                                 key="user_pos_input")
                    
                    # 버튼과 결과를 함께 처리
                    if st.button("🔍 내 문장 품사 분석하기", key="pos_demo") and user_sentence.strip():
                        with st.spinner("문장을 분석하는 중..."):
                            try:
                                import nltk
                                tokens = nltk.word_tokenize(user_sentence)
                                pos_demo = nltk.pos_tag(tokens)
                                
                                st.write(f"**입력한 문장:** {user_sentence}")
                                st.write("**품사 분석 결과:**")
                                
                                # 품사별 색상 표시 (간단한 방식)
                                st.write("**품사별 분석 결과:**")
                                
                                # 컬럼으로 나누어 표시
                                cols = st.columns(4)
                                col_idx = 0
                                
                                for word, pos in pos_demo:
                                    # 품사별 이모지와 색상
                                    if pos.startswith('NN'):  # 명사
                                        emoji = "🟦"
                                        category = "명사"
                                    elif pos.startswith('VB') or pos == 'MD':  # 동사
                                        emoji = "🟩"
                                        category = "동사"
                                    elif pos.startswith('JJ'):  # 형용사
                                        emoji = "🟨"
                                        category = "형용사"
                                    elif pos.startswith('RB'):  # 부사
                                        emoji = "🟪"
                                        category = "부사"
                                    elif pos in ['PRP', 'PRP$']:  # 대명사
                                        emoji = "🟡"
                                        category = "대명사"
                                    elif pos in ['IN', 'TO']:  # 전치사
                                        emoji = "🟣"
                                        category = "전치사"
                                    elif pos == 'DT':  # 관사
                                        emoji = "🟢"
                                        category = "관사"
                                    elif pos == 'CC':  # 접속사
                                        emoji = "🟠"
                                        category = "접속사"
                                    else:
                                        emoji = "⚪"
                                        category = "기타"
                                    
                                    with cols[col_idx % 4]:
                                        st.write(f"{emoji} **{word}**")
                                        st.caption(f"{pos} ({category})")
                                    
                                    col_idx += 1
                                
                                # 품사 통계 추가
                                pos_stats = {}
                                for word, pos in pos_demo:
                                    category = '기타'
                                    if pos.startswith('NN'):
                                        category = '명사'
                                    elif pos.startswith('VB') or pos == 'MD':
                                        category = '동사/조동사'
                                    elif pos.startswith('JJ'):
                                        category = '형용사'
                                    elif pos.startswith('RB'):
                                        category = '부사'
                                    elif pos in ['PRP', 'PRP$']:
                                        category = '대명사'
                                    elif pos in ['DT', 'IN', 'CC']:
                                        category = '기능어'
                                    
                                    pos_stats[category] = pos_stats.get(category, 0) + 1
                                
                                if pos_stats:
                                    st.markdown("**📊 품사 분포:**")
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        for category, count in pos_stats.items():
                                            st.write(f"• {category}: {count}개")
                                    
                                    with col2:
                                        total_words = sum(pos_stats.values())
                                        most_common = max(pos_stats.items(), key=lambda x: x[1])
                                        st.metric("총 단어 수", f"{total_words}개")
                                        st.metric("가장 많은 품사", f"{most_common[0]} ({most_common[1]}개)")
                                
                                # 품사 설명 추가
                                st.markdown("---")
                                st.info("""
                                **품사 분류 기준:**
                                🟦 명사(NN) | 🟩 동사(VB) | 🟨 형용사(JJ) | 🟪 부사(RB) | 🟡 대명사(PRP) | 🟣 전치사(IN) | 🟢 관사(DT) | 🟠 접속사(CC)
                                """)
                                
                            except Exception as e:
                                st.error(f"문장 분석 중 오류: {e}")
                                st.warning("영어 문장을 정확히 입력했는지 확인해주세요.")
            else:
                st.warning("2단계 분석에 문제가 발생했습니다.")
            
            st.markdown("---")
            
            # 3단계: 패턴 발견
            st.markdown("## 🎯 3단계: 패턴 발견 및 언어적 특성 분석")
            st.markdown("""
            **🔍 원리 설명:**
            - **품사 패턴**과 **언어적 특성** 분석
            - **문장 구조**와 **어휘 다양성** 측정
            - **글쓰기 스타일** 특성 발견
            """)
            
            if pos_method3 and 'error' not in pos_method3:
                st.success("✅ 3단계 패턴 발견 및 언어적 특성 분석 완료!")
                
                # 패턴 분석 결과
                patterns = pos_method3.get('common_patterns', [])
                if patterns:
                    st.write("**🔍 발견된 품사 패턴 (상위 5개):**")
                    for i, pattern in enumerate(patterns[:5], 1):
                        st.write(f"{i}. `{pattern}`")
                
                # 언어적 특성 분석
                linguistic_features = pos_method3.get('linguistic_features', {})
                if linguistic_features:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("문장 복잡도", f"{linguistic_features.get('complexity_score', 0):.2f}")
                    with col2:
                        st.metric("어휘 다양성", f"{linguistic_features.get('lexical_diversity', 0):.2f}")
                    with col3:
                        st.metric("평균 문장 길이", f"{linguistic_features.get('avg_sentence_length', 0):.1f}")
                
                # 체험형 패턴 분석 요소 추가
                st.markdown("**🎯 체험해보기: 직접 입력한 문장의 언어적 특성 분석**")
                
                user_pattern_text = st.text_area("분석할 문장들을 영어로 입력하세요 (여러 문장 가능):", 
                                               value="I love reading books. They help me learn new things. Sometimes I write stories about my dreams.",
                                               height=100,
                                               placeholder="여러 문장을 입력하면 각 문장의 특성을 개별적으로 분석합니다.",
                                               key="user_pattern_input")
                
                if st.button("🔍 내 문장의 언어적 패턴 분석하기", key="pattern_analysis_user") and user_pattern_text.strip():
                    with st.spinner("언어적 패턴을 분석하는 중..."):
                        try:
                            import nltk
                            
                            # 사용자 입력 텍스트를 문장별로 분리
                            sentences = nltk.sent_tokenize(user_pattern_text)
                            
                            st.markdown(f"**📝 총 {len(sentences)}개 문장의 상세 분석:**")
                            
                            sentence_analyses = []
                            for i, sentence in enumerate(sentences[:5], 1):  # 최대 5개 문장까지
                                words = nltk.word_tokenize(sentence)
                                pos_tags = nltk.pos_tag(words)
                                
                                # 품사별 카운트
                                nouns = sum(1 for _, pos in pos_tags if pos.startswith('NN'))
                                verbs = sum(1 for _, pos in pos_tags if pos.startswith('VB'))
                                adjectives = sum(1 for _, pos in pos_tags if pos.startswith('JJ'))
                                total_content_words = nouns + verbs + adjectives
                                
                                # 비율 계산
                                if total_content_words > 0:
                                    noun_ratio = (nouns / total_content_words) * 100
                                    verb_ratio = (verbs / total_content_words) * 100
                                    adj_ratio = (adjectives / total_content_words) * 100
                                else:
                                    noun_ratio = verb_ratio = adj_ratio = 0
                                
                                # 문장 복잡도 판단 (15단어 이상 또는 복잡한 접속사 포함)
                                complex_words = ['because', 'although', 'since', 'while', 'if', 'when', 'where', 'which', 'that']
                                is_complex = len(words) > 15 or any(word.lower() in complex_words for word in words)
                                complexity = "Complex" if is_complex else "Simple"
                                
                                analysis = {
                                    'sentence': sentence,
                                    'word_count': len(words),
                                    'nouns': nouns,
                                    'verbs': verbs,
                                    'adjectives': adjectives,
                                    'content_words': total_content_words,
                                    'noun_ratio': noun_ratio,
                                    'verb_ratio': verb_ratio,
                                    'adj_ratio': adj_ratio,
                                    'complexity': complexity,
                                    'pos_tags': pos_tags
                                }
                                
                                sentence_analyses.append(analysis)
                                
                                with st.expander(f"📄 {i}번째 문장 상세 분석"):
                                    st.write(f"**문장:** {sentence}")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("단어 개수", f"{len(words)}개")
                                        st.metric("명사 개수", f"{nouns}개")
                                    with col2:
                                        st.metric("동사 개수", f"{verbs}개")
                                        st.metric("형용사 개수", f"{adjectives}개")
                                    with col3:
                                        st.metric("문장 복잡도", complexity)
                                        st.metric("내용어 총계", f"{total_content_words}개")
                                    
                                    # 품사 비율 시각화
                                    if total_content_words > 0:
                                        fig = go.Figure(data=[go.Bar(
                                            x=['명사', '동사', '형용사'],
                                            y=[noun_ratio, verb_ratio, adj_ratio],
                                            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                                            text=[f"{noun_ratio:.1f}%", f"{verb_ratio:.1f}%", f"{adj_ratio:.1f}%"],
                                            textposition='auto'
                                        )])
                                        fig.update_layout(
                                            title=f"{i}번째 문장의 품사 비율",
                                            yaxis_title="비율 (%)",
                                            height=300
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    # 품사 태그 상세 보기
                                    if st.checkbox(f"품사 태그 상세 보기", key=f"pos_detail_{i}"):
                                        st.write("**품사별 단어:**")
                                        pos_groups = {}
                                        for word, pos in pos_tags:
                                            if pos not in pos_groups:
                                                pos_groups[pos] = []
                                            pos_groups[pos].append(word)
                                        
                                        for pos, words_list in pos_groups.items():
                                            st.write(f"• **{pos}**: {', '.join(words_list)}")
                            
                            # 전체 텍스트 종합 분석
                            if len(sentence_analyses) >= 1:
                                st.markdown("---")
                                st.markdown("### 📊 전체 텍스트 종합 분석")
                                
                                # 평균 통계 계산
                                avg_noun_ratio = sum(s['noun_ratio'] for s in sentence_analyses) / len(sentence_analyses)
                                avg_verb_ratio = sum(s['verb_ratio'] for s in sentence_analyses) / len(sentence_analyses)
                                avg_adj_ratio = sum(s['adj_ratio'] for s in sentence_analyses) / len(sentence_analyses)
                                avg_sentence_length = sum(s['word_count'] for s in sentence_analyses) / len(sentence_analyses)
                                complex_count = sum(1 for s in sentence_analyses if s['complexity'] == 'Complex')
                                complexity_ratio = (complex_count / len(sentence_analyses)) * 100
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("평균 명사 비율", f"{avg_noun_ratio:.1f}%")
                                    st.metric("평균 동사 비율", f"{avg_verb_ratio:.1f}%")
                                with col2:
                                    st.metric("평균 형용사 비율", f"{avg_adj_ratio:.1f}%")
                                    st.metric("평균 문장 길이", f"{avg_sentence_length:.1f}단어")
                                with col3:
                                    st.metric("복잡한 문장 비율", f"{complexity_ratio:.1f}%")
                                    st.metric("총 문장 수", f"{len(sentence_analyses)}개")
                                
                                # 글쓰기 스타일 판정
                                st.markdown("**✨ 글쓰기 스타일 분석:**")
                                if avg_noun_ratio > 50:
                                    st.info("📊 **정보 전달형**: 명사가 많아 객관적이고 정보 중심적인 글쓰기입니다.")
                                elif avg_verb_ratio > 30:
                                    st.success("🏃 **동적 표현형**: 동사가 많아 활동적이고 생동감 있는 글쓰기입니다.")
                                elif avg_adj_ratio > 20:
                                    st.warning("🎨 **묘사적**: 형용사가 풍부해 감정적이고 묘사적인 글쓰기입니다.")
                                else:
                                    st.info("⚖️ **균형잡힌**: 각 품사가 고르게 사용된 균형잡힌 글쓰기입니다.")
                                
                                # 복잡도 평가
                                if complexity_ratio > 60:
                                    st.success("🎓 **고급 수준**: 복잡하고 정교한 문장 구조를 사용합니다!")
                                elif complexity_ratio > 30:
                                    st.info("📝 **중급 수준**: 적절한 복잡성을 가진 글쓰기입니다.")
                                else:
                                    st.warning("📚 **기초 수준**: 더 다양한 문장 구조를 시도해보세요.")
                            
                        except Exception as e:
                            st.error(f"패턴 분석 중 오류: {e}")
                            st.warning("영어 문장을 정확히 입력했는지 확인해주세요.")
                
                # 글쓰기 스타일 체험 요소
                writing_style = pos_method3.get('writing_style', {})
                if writing_style:
                    st.markdown("**✨ 발견된 글쓰기 스타일:**")
                    for style_key, style_desc in writing_style.items():
                        if style_key == 'noun_heavy':
                            st.info(f"📊 **정보 전달형**: {style_desc}")
                        elif style_key == 'verb_heavy':
                            st.success(f"🏃 **동적 표현형**: {style_desc}")
                        elif style_key == 'descriptive':
                            st.warning(f"🎨 **묘사적**: {style_desc}")
                        else:
                            st.info(f"⚖️ **균형잡힌 스타일**: {style_desc}")
                
                # 인터랙티브 패턴 체험
                st.markdown("**🔍 패턴 발견 체험:**")
                if st.button("🎯 내 글쓰기 패턴 분석해보기", key="pattern_analysis_demo"):
                    overall_patterns = pos_method3.get('overall_patterns', {})
                    if overall_patterns:
                        st.markdown("### 📊 종합 패턴 분석 결과")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**📈 품사 밀도 분석:**")
                            densities = {
                                '평균 명사 밀도': overall_patterns.get('avg_noun_density', 0),
                                '평균 동사 밀도': overall_patterns.get('avg_verb_density', 0),
                                '평균 형용사 밀도': overall_patterns.get('avg_adj_density', 0)
                            }
                            
                            for name, value in densities.items():
                                st.write(f"• {name}: {value:.1f}%")
                        
                        with col2:
                            st.markdown("**🏗️ 문장 구조 분석:**")
                            complexity_ratio = overall_patterns.get('complexity_ratio', 0)
                            total_sentences = overall_patterns.get('total_sentences', 0)
                            
                            st.write(f"• 복잡한 문장 비율: {complexity_ratio:.1f}%")
                            st.write(f"• 분석된 문장 수: {total_sentences}개")
                        
                        # 종합 평가
                        if complexity_ratio > 60:
                            st.success("🎓 **고급 수준**: 복잡하고 정교한 문장 구조를 사용합니다!")
                        elif complexity_ratio > 30:
                            st.info("📝 **중급 수준**: 적절한 복잡성을 가진 글쓰기입니다.")
                        else:
                            st.warning("📚 **기초 수준**: 더 다양한 문장 구조를 시도해보세요.")
            else:
                st.warning("3단계 분석에 문제가 발생했습니다.")
            
            st.markdown("---")
            
            # 3가지 방법 비교 분석 섹션 추가
            st.markdown("## 📊 3가지 방법 비교 분석")
            st.markdown("동일한 텍스트를 3가지 방법으로 분석한 결과를 비교해보세요!")
            
            if (pos_method1 and 'error' not in pos_method1 and 
                pos_method2 and 'error' not in pos_method2 and 
                pos_method3 and 'error' not in pos_method3):
                
                # 비교 테이블
                comparison_data = {
                    "분석 방법": ["1단계: 수동 규칙", "2단계: NLTK 머신러닝", "3단계: 패턴 발견"],
                    "장점": [
                        "빠른 처리, 예측 가능",
                        "높은 정확도, 문맥 고려", 
                        "언어적 특성 발견, 스타일 분석"
                    ],
                    "단점": [
                        "예외 처리 어려움",
                        "계산 비용 높음",
                        "복잡한 해석 필요"
                    ],
                    "적용 분야": [
                        "실시간 처리, 규칙적 텍스트",
                        "정확한 품사 태깅 필요한 분야",
                        "문체 분석, 글쓰기 평가"
                    ]
                }
                
                st.dataframe(comparison_data, use_container_width=True)
                
                # 진짜 3가지 방법 종합 비교
                st.markdown("**🔍 3가지 접근법의 특성 비교:**")
                
                # 3단계 모두 포함한 비교 차트 - 각자의 강점 표시
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**🔴 1단계: 규칙 기반**")
                    method1_pos = pos_method1.get('pos_counts', {})
                    if method1_pos:
                        total_words = sum(method1_pos.values())
                        st.metric("분석 단어 수", f"{total_words}개")
                        st.metric("처리 속도", "⚡ 빠름")
                        pos_categories = len(method1_pos.keys())
                        st.metric("품사 분류 종류", f"{pos_categories}개")
                    
                with col2:
                    st.markdown("**🔵 2단계: NLTK 기반**")
                    pos_tagged = pos_method2.get('pos_tagged_words', [])
                    if pos_tagged:
                        st.metric("분석 단어 수", f"{len(pos_tagged)}개")
                        st.metric("정확도", "🎯 높음")
                        pos_types = len(set(pos for word, pos in pos_tagged))
                        st.metric("품사 태그 종류", f"{pos_types}개")
                
                with col3:
                    st.markdown("**🟢 3단계: 패턴 발견**")
                    linguistic_features = pos_method3.get('linguistic_features', {})
                    overall_patterns = pos_method3.get('overall_patterns', {})
                    if linguistic_features and overall_patterns:
                        st.metric("복잡도 점수", f"{linguistic_features.get('complexity_score', 0):.2f}")
                        st.metric("분석 깊이", "🔬 심화")
                        st.metric("패턴 개수", f"{len(pos_method3.get('common_patterns', []))}개")
                
                # 3가지 방법의 상호 보완적 특성 보여주기
                st.markdown("**📊 종합 비교 분석:**")
                
                # 각 방법이 발견한 것들을 비교
                method1_total = sum(pos_method1.get('pos_counts', {}).values()) if pos_method1.get('pos_counts') else 0
                method2_total = len(pos_method2.get('pos_tagged_words', [])) if pos_method2.get('pos_tagged_words') else 0
                method3_patterns = len(pos_method3.get('common_patterns', [])) if pos_method3.get('common_patterns') else 0
                
                comparison_metrics = {
                    "분석 측면": ["기본 품사 인식", "정밀 품사 태깅", "언어 패턴 발견"],
                    "1단계 결과": [f"{method1_total}개 단어 분류", "기본 수준", "없음"],
                    "2단계 결과": [f"{method2_total}개 단어 분류", "상세 태그", "없음"], 
                    "3단계 결과": ["품사 밀도 분석", "문장별 특성", f"{method3_patterns}개 패턴"]
                }
                
                st.dataframe(comparison_metrics, use_container_width=True)
                
                # 상호 보완적 관계 설명
                st.info("""
                **🔄 3가지 방법의 상호 보완 관계:**
                - **1단계 → 2단계**: 규칙 기반의 빠른 분류 → 기계학습의 정확한 태깅  
                - **2단계 → 3단계**: 개별 단어 태깅 → 전체 텍스트의 언어적 패턴 발견
                - **종합 활용**: 빠른 1차 분류 + 정확한 태깅 + 고차원 패턴 분석
                """)
                
                # 체험형 비교 요소
                st.markdown("**🎯 직접 비교해보기:**")
                sample_text = st.text_input("비교할 문장을 입력하세요:", 
                                          value="I am writing a beautiful essay about my dreams.",
                                          key="pos_comparison_input")
                
                # 버튼과 결과를 함께 처리
                if st.button("🔍 3가지 방법으로 동시 분석하기", key="pos_compare_demo") and sample_text:
                        with st.spinner("3가지 방법으로 동시 분석하는 중..."):
                            try:
                                # 간단한 규칙 기반 분석
                                words = sample_text.lower().split()
                                rule_based = []
                                for word in words:
                                    if word.endswith('ing'):
                                        rule_based.append((word, 'VBG'))
                                    elif word in ['i', 'you', 'he', 'she', 'it', 'we', 'they']:
                                        rule_based.append((word, 'PRP'))
                                    elif word in ['a', 'an', 'the']:
                                        rule_based.append((word, 'DT'))
                                    elif word in ['am', 'is', 'are', 'was', 'were', 'be', 'been', 'being']:
                                        rule_based.append((word, 'VB'))
                                    elif word.endswith('ly'):
                                        rule_based.append((word, 'RB'))
                                    elif word in ['beautiful', 'good', 'bad', 'great', 'amazing', 'wonderful']:
                                        rule_based.append((word, 'JJ'))
                                    else:
                                        rule_based.append((word, 'NN'))
                                
                                # NLTK 기반 분석
                                import nltk
                                tokens = nltk.word_tokenize(sample_text)
                                nltk_based = nltk.pos_tag(tokens)
                                
                                # 결과 비교 표시
                                st.markdown("### 🔍 3가지 방법 비교 결과")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.markdown("**🔴 1단계: 규칙 기반**")
                                    st.markdown("---")
                                    for word, pos in rule_based:
                                        if pos == 'NN':
                                            st.markdown(f"🟦 {word} → **{pos}** (명사)")
                                        elif pos.startswith('VB'):
                                            st.markdown(f"🟩 {word} → **{pos}** (동사)")
                                        elif pos == 'JJ':
                                            st.markdown(f"🟨 {word} → **{pos}** (형용사)")
                                        elif pos == 'RB':
                                            st.markdown(f"🟪 {word} → **{pos}** (부사)")
                                        else:
                                            st.markdown(f"⚪ {word} → **{pos}** (기타)")
                                
                                with col2:
                                    st.markdown("**🔵 2단계: NLTK 기반**")
                                    st.markdown("---")
                                    for word, pos in nltk_based:
                                        if pos.startswith('NN'):
                                            st.markdown(f"🟦 {word} → **{pos}** (명사)")
                                        elif pos.startswith('VB'):
                                            st.markdown(f"🟩 {word} → **{pos}** (동사)")
                                        elif pos.startswith('JJ'):
                                            st.markdown(f"🟨 {word} → **{pos}** (형용사)")
                                        elif pos.startswith('RB'):
                                            st.markdown(f"🟪 {word} → **{pos}** (부사)")
                                        elif pos in ['DT', 'IN', 'TO', 'PRP']:
                                            st.markdown(f"🟫 {word} → **{pos}** (기능어)")
                                        else:
                                            st.markdown(f"⚪ {word} → **{pos}** (기타)")
                                
                                with col3:
                                    st.markdown("**🟢 차이점 분석**")
                                    st.markdown("---")
                                    differences = []
                                    same_count = 0
                                    
                                    # 단어별 비교 (길이가 다를 수 있으므로 조심스럽게 처리)
                                    rule_dict = {w.lower(): p for w, p in rule_based}
                                    nltk_dict = {w.lower(): p for w, p in nltk_based}
                                    
                                    all_words = set(rule_dict.keys()) | set(nltk_dict.keys())
                                    
                                    for word in all_words:
                                        rule_pos = rule_dict.get(word, 'N/A')
                                        nltk_pos = nltk_dict.get(word, 'N/A')
                                        
                                        if rule_pos != nltk_pos:
                                            differences.append(f"• **{word}**: {rule_pos} → {nltk_pos}")
                                        else:
                                            same_count += 1
                                    
                                    if differences:
                                        st.markdown("**주요 차이점:**")
                                        for diff in differences[:8]:  # 최대 8개만 표시
                                            st.markdown(diff)
                                        if len(differences) > 8:
                                            st.markdown(f"... 외 {len(differences)-8}개 더")
                                    
                                    st.markdown(f"**📊 비교 통계:**")
                                    st.markdown(f"• 동일: {same_count}개")
                                    st.markdown(f"• 다름: {len(differences)}개")
                                    
                                    accuracy = same_count / len(all_words) * 100 if all_words else 0
                                    st.markdown(f"• 일치율: {accuracy:.1f}%")
                                
                                st.success("✅ 3가지 방법 비교 분석이 완료되었습니다!")
                                
                                # 학습 포인트
                                st.info("""
                                **💡 학습 포인트:**
                                - **규칙 기반**: 빠르지만 단순한 분류
                                - **NLTK**: 더 정확하고 세밀한 품사 태깅
                                - **차이점**: 문맥을 고려한 정확도 차이 확인 가능
                                """)
                            
                            except Exception as e:
                                st.error(f"비교 분석 중 오류: {e}")
                                import traceback
                                st.code(traceback.format_exc())
            
            else:
                st.warning("3가지 방법 비교를 위해서는 모든 단계가 성공적으로 완료되어야 합니다.")
            
            # 종합 정리
            st.markdown("---")
            st.subheader("🎓 품사 분석 학습 정리")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("""
                **🔍 배운 내용:**
                - 수동 규칙 기반의 패턴 매칭
                - NLTK 기계학습 모델의 활용
                - 품사 패턴을 통한 언어적 특성 발견
                - 각 방법의 장단점과 적용 분야
                """)
            
            with col2:
                st.success("""
                **💡 실무 적용:**
                - 빠른 분석: 규칙 기반
                - 정확한 태깅: NLTK 기계학습
                - 문체 분석: 패턴 발견
                - 자연어처리: 종합 접근법 활용
                """)
            
            st.success("✅ 텍스트 마이닝 품사 분석 원리 체험이 완료되었습니다!")
                
        else:
            st.error("품사 분석 결과를 생성할 수 없습니다.")
    
    # 3. 글쓰기 수준 종합 진단 탭
    with analysis_tabs[2]:
        st.subheader("🏆 글쓰기 수준 종합 진단 원리 체험")
        st.markdown("""
        **🎯 학습 목표:**
        AI가 어떻게 글쓰기를 평가하는지 4단계로 체험해보세요!
        - **1단계**: 통계적 텍스트 분석 - 문장 구조와 어휘 다양성 측정
        - **2단계**: 어휘 수준 분석 - 고급 어휘 사용 평가
        - **3단계**: 문법 오류 패턴 분석 - 정확성 및 문법 규칙 검사
        - **4단계**: 문장 유사도 분석 - 논리적 연결성과 주제 일관성 평가
        """)
        
        st.info("💡 **체험 포인트**: 실제 AI 글쓰기 평가 시스템의 작동 원리를 단계별로 이해해보세요!")
        
        # 자동으로 종합 분석 실행
        if all_essays_text.strip():
            try:
                # 종합 분석 실행
                with st.spinner("📊 통합 에세이 텍스트 분석 중..."):
                    result = preprocessor.comprehensive_writing_analysis(all_essays_text)
                
                if result and 'error' not in result:
                    st.success("✅ 통합 에세이 데이터 분석 완료!")
                    
                    # 1단계: 통계적 텍스트 분석
                    st.markdown("## 📊 1단계: 통계적 텍스트 분석")
                    st.markdown("""
                    **🔍 원리 설명:**
                    - **문장 길이 분포**: 평균 문장 길이로 글쓰기 복잡도 측정
                    - **어휘 다양성**: TTR(Type-Token Ratio) 계산으로 어휘 풍부함 평가
                    - **품사 패턴**: 명사, 동사, 형용사 비율로 글쓰기 스타일 분석
                    """)
                    
                    # 학생 통합 에세이 통계 분석 결과 표시
                    st.markdown("### 📊 학생 통합 에세이 통계 분석 결과")
                    statistical_analysis = result.get('step1_stats', {})
                    if statistical_analysis:
                        user_stats = statistical_analysis.get('user_statistics', {})
                        if user_stats:
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                avg_sentence_length = user_stats.get('avg_sentence_length', 0)
                                st.metric("평균 문장 길이", f"{avg_sentence_length:.1f} 단어")
                                if avg_sentence_length > 15:
                                    st.caption("🟢 복합문 활용")
                                elif avg_sentence_length > 10:
                                    st.caption("🟡 적정 수준")
                                else:
                                    st.caption("🟠 단순문 위주")
                            
                            with col2:
                                lexical_diversity = user_stats.get('vocabulary_diversity', 0)
                                st.metric("어휘 다양성", f"{lexical_diversity:.3f}")
                                if lexical_diversity > 0.7:
                                    st.caption("🟢 매우 다양함")
                                elif lexical_diversity > 0.5:
                                    st.caption("🟡 적정함")
                                else:
                                    st.caption("🟠 단조로움")
                            
                            with col3:
                                total_sentences = user_stats.get('total_sentences', 0)
                                st.metric("총 문장 수", f"{total_sentences}개")
                            
                            with col4:
                                total_words = user_stats.get('total_words', 0)
                                st.metric("총 단어 수", f"{total_words}개")
                            
                            # 벤치마크 비교 결과 표시
                            st.markdown("### 🏆 전문가 기준과의 비교")
                            best_match = statistical_analysis.get('best_match', None)
                            if best_match:
                                match_name = best_match[1]['benchmark_name']
                                similarity_score = best_match[1]['similarity_score']
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("가장 유사한 글 유형", match_name)
                                with col2:
                                    st.metric("유사도 점수", f"{similarity_score:.1f}%")
                                    if similarity_score > 80:
                                        st.caption("🟢 매우 우수")
                                    elif similarity_score > 60:
                                        st.caption("🟡 양호")
                                    else:
                                        st.caption("🟠 개선 필요")
                                
                                # 세부 비교 점수
                                individual_scores = best_match[1]['individual_scores']
                                st.markdown("**📋 세부 항목별 점수:**")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.write(f"• 명사 비율: {individual_scores.get('noun_score', 0):.1f}%")
                                    st.write(f"• 동사 비율: {individual_scores.get('verb_score', 0):.1f}%")
                                
                                with col2:
                                    st.write(f"• 형용사 비율: {individual_scores.get('adj_score', 0):.1f}%")
                                    st.write(f"• 복잡도: {individual_scores.get('complexity_score', 0):.1f}%")
                                
                                with col3:
                                    st.write(f"• 어휘 다양성: {individual_scores.get('diversity_score', 0):.1f}%")
                                    st.write(f"• 문장 길이: {individual_scores.get('length_score', 0):.1f}%")
                            
                            # 통찰 및 개선점
                            insights = statistical_analysis.get('insights', [])
                            if insights:
                                st.markdown("### 💡 분석 통찰")
                                for insight in insights[:3]:
                                    st.info(f"• {insight}")
                    else:
                        st.warning("통합 에세이 통계 분석 결과가 없습니다.")
                    
                    # 체험형 인터랙티브 요소 - 1단계
                    st.markdown("**🎯 체험해보기: 내 글의 통계적 특성 분석**")
                    user_text_stats = st.text_area("분석할 영어 에세이를 입력하세요:", 
                                                   value="Education is very important for everyone. Students should study hard to achieve their goals. Learning helps us grow and become better people. We need to practice reading and writing every day.",
                                                   height=100,
                                                   key="user_stats_input")
                    
                    if st.button("🔍 내 글 통계 분석하기", key="stats_analysis") and user_text_stats.strip():
                        with st.spinner("통계적 특성 분석 중..."):
                            import nltk
                            sentences = nltk.sent_tokenize(user_text_stats)
                            words = nltk.word_tokenize(user_text_stats)
                            word_count = len([w for w in words if w.isalnum()])
                            unique_words = len(set(w.lower() for w in words if w.isalnum()))
                            
                            user_avg_length = word_count / len(sentences) if sentences else 0
                            user_diversity = unique_words / word_count if word_count > 0 else 0
                            
                            st.write(f"**입력한 글:** {user_text_stats}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("평균 문장 길이", f"{user_avg_length:.1f} 단어")
                            with col2:
                                st.metric("어휘 다양성", f"{user_diversity:.3f}")
                            with col3:
                                st.metric("문장 수", f"{len(sentences)}개")
                            
                            # 평가 코멘트
                            if user_avg_length > 12 and user_diversity > 0.6:
                                st.success("🎉 통계적으로 우수한 글쓰기 특성을 보입니다!")
                            elif user_avg_length > 8 and user_diversity > 0.4:
                                st.info("📝 적절한 수준의 글쓰기입니다. 어휘를 더 다양하게 사용해보세요.")
                            else:
                                st.warning("💡 문장을 좀 더 길고 복합적으로 작성하고, 다양한 어휘를 사용해보세요.")
                    
                    st.markdown("---")
                    
                    # 2단계: 어휘 수준 분석
                    st.markdown("## 📚 2단계: 어휘 수준 분석 - 고급 어휘 사용 평가")
                    st.markdown("""
                    **🔍 원리 설명:**
                    - **고급 어휘 사전**: 학술적, 고급 어휘의 사용 빈도 측정
                    - **수준별 분류**: 기초/중급/고급/학술 어휘로 분류하여 글쓰기 수준 평가
                    - **통찰 발견**: 어휘 다양성과 정교성을 통한 글쓰기 성숙도 측정
                    """)
                    
                    # 학생 통합 에세이 워드 임베딩 분석 결과
                    st.markdown("### 🧠 학생 통합 에세이 어휘 수준 분석 결과")
                    vocabulary_analysis = result.get('step2_vocabulary', {})
                    if vocabulary_analysis:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            vocab_level = vocabulary_analysis.get('overall_level', 'Unknown')
                            st.metric("전체 어휘 수준", vocab_level)
                        
                        with col2:
                            advanced_ratio = vocabulary_analysis.get('advanced_vocabulary_ratio', 0)
                            st.metric("고급 어휘 비율", f"{advanced_ratio:.1f}%")
                            if advanced_ratio > 20:
                                st.caption("🟢 높은 수준")
                            elif advanced_ratio > 10:
                                st.caption("🟡 보통 수준")
                            else:
                                st.caption("🟠 기초 수준")
                        
                        with col3:
                            academic_score = vocabulary_analysis.get('academic_vocabulary_score', 0)
                            st.metric("학술 어휘 점수", f"{academic_score:.1f}점")
                        
                        # 어휘 수준별 분석
                        level_analysis = vocabulary_analysis.get('level_analysis', {})
                        if level_analysis:
                            st.markdown("### 📚 수준별 어휘 분포")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                basic_count = level_analysis.get('basic_words', 0)
                                st.metric("기초 어휘", f"{basic_count}개")
                            
                            with col2:
                                intermediate_count = level_analysis.get('intermediate_words', 0) 
                                st.metric("중급 어휘", f"{intermediate_count}개")
                            
                            with col3:
                                advanced_count = level_analysis.get('advanced_words', 0)
                                st.metric("고급 어휘", f"{advanced_count}개")
                            
                            with col4:
                                academic_count = level_analysis.get('academic_words', 0)
                                st.metric("학술 어휘", f"{academic_count}개")
                        
                        # 어휘 다양성 및 복잡도
                        complexity_analysis = vocabulary_analysis.get('complexity_analysis', {})
                        if complexity_analysis:
                            st.markdown("### 🔬 어휘 복잡도 분석")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                avg_word_length = complexity_analysis.get('avg_word_length', 0)
                                st.metric("평균 단어 길이", f"{avg_word_length:.1f} 글자")
                                
                                unique_ratio = complexity_analysis.get('unique_word_ratio', 0)
                                st.metric("고유 단어 비율", f"{unique_ratio:.1f}%")
                            
                            with col2:
                                sophistication_score = complexity_analysis.get('sophistication_score', 0)
                                st.metric("어휘 정교성", f"{sophistication_score:.1f}점")
                                
                                st.write(f"**평가:** {complexity_analysis.get('level_description', '보통 수준')}")
                        
                        # 개선 추천 사항
                        recommendations = vocabulary_analysis.get('vocabulary_recommendations', [])
                        if recommendations:
                            st.markdown("### 💡 어휘 개선 방향")
                            for rec in recommendations[:3]:
                                st.info(f"• {rec}")
                    else:
                        st.warning("통합 에세이 어휘 분석 결과가 없습니다.")
                    
                    # 체험형 인터랙티브 요소 - 2단계
                    st.markdown("**🎯 체험해보기: 내 글의 어휘 수준 분석**")
                    
                    user_vocab_text = st.text_area("어휘 수준을 분석할 영어 텍스트를 입력하세요:", 
                                                   value="I think education is important because it helps students develop critical thinking skills and prepare for future challenges in their professional careers.",
                                                   height=100,
                                                   key="user_vocab_input")
                    
                    if st.button("🔍 내 글 어휘 수준 분석하기", key="vocab_analysis") and user_vocab_text.strip():
                        with st.spinner("어휘 수준 분석 중..."):
                            # 단어 전처리 (구두점 제거)
                            import re
                            words = re.findall(r'\b[a-zA-Z]+\b', user_vocab_text.lower())
                            total_words = len(words)
                            
                            # 자동 어휘 수준 분류
                            basic_words = []
                            intermediate_words = []
                            advanced_words = []
                            
                            # 고급 어미 패턴들
                            advanced_suffixes = ['tion', 'sion', 'ment', 'ness', 'ity', 'ous', 'ive', 'able', 'ible', 'ful', 'less', 'ence', 'ance', 'ize', 'ise', 'ate']
                            
                            for word in words:
                                word_len = len(word)
                                has_advanced_suffix = any(word.endswith(suffix) for suffix in advanced_suffixes)
                                
                                # 분류 로직
                                if word_len <= 4 and not has_advanced_suffix:
                                    basic_words.append(word)
                                elif word_len >= 8 or has_advanced_suffix:
                                    advanced_words.append(word)
                                else:
                                    intermediate_words.append(word)
                            
                            # 고급 어휘 비율 계산
                            advanced_count = len(advanced_words)
                            advanced_ratio = (advanced_count / total_words * 100) if total_words > 0 else 0
                            
                            # 카테고리별 결과 (체험용 간략화)
                            category_results = {
                                'length_based': [w for w in advanced_words if len(w) >= 8],
                                'suffix_based': [w for w in advanced_words if any(w.endswith(s) for s in advanced_suffixes)]
                            }
                            
                            st.write(f"**분석한 텍스트:** {user_vocab_text}")
                            
                            # 분류 결과 표시
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("총 단어 수", f"{total_words}개")
                            with col2:
                                st.metric("기초 어휘", f"{len(basic_words)}개")
                            with col3:
                                st.metric("중급 어휘", f"{len(intermediate_words)}개")
                            with col4:
                                st.metric("고급 어휘", f"{len(advanced_words)}개")
                            
                            # 고급 어휘 비율 및 평가
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("고급 어휘 비율", f"{advanced_ratio:.1f}%")
                            with col2:
                                if advanced_ratio > 20:
                                    st.metric("어휘 수준", "🟢 고급")
                                elif advanced_ratio > 10:
                                    st.metric("어휘 수준", "🟡 중급")
                                else:
                                    st.metric("어휘 수준", "🟠 기초")
                            
                            # 발견된 고급 어휘 예시 (최대 10개)
                            if advanced_words:
                                st.markdown("**📚 발견된 고급 어휘 (예시):**")
                                st.write("• " + ", ".join(advanced_words[:10]))
                                if len(advanced_words) > 10:
                                    st.caption(f"외 {len(advanced_words)-10}개 더...")
                            else:
                                st.info("💡 더 고급스러운 어휘를 사용해보세요: development, significant, challenging, professional 등")
                            
                            # 분류 기준 설명
                            st.markdown("**🔍 이 체험에서 사용한 자동 분류 기준:**")
                            st.write("• 기초: 4글자 이하 단어 (예: good, use, make)")
                            st.write("• 고급: 8글자 이상 또는 고급 어미(-tion, -ment, -ness, -ous, -ive 등)")
                            st.write("• 중급: 나머지 단어들")
                            
                            st.info("""
                            💡 **실제 AI 시스템에서는 더 정교한 방법 사용:**
                            - **CEFR 레벨 데이터** (A1~C2 수준별 어휘 분류)
                            - **Academic Word List** (학술 논문 필수 어휘)
                            - **General Service List** (일상 생활 기초 어휘)
                            - **Word2Vec 임베딩** (의미적 유사성 기반 분류)
                            """)
                    
                    st.markdown("---")
                    
                    # 3단계: 문법 오류 패턴 분석
                    st.markdown("## 📝 3단계: 문법 오류 패턴 분석 - 정확성 평가")
                    st.markdown("""
                    **🔍 원리 설명:**
                    - **주어-동사 수일치**: I am, He is, They are 등 기본 문법 규칙 검사
                    - **시제 일관성**: 한 문장 내에서 과거/현재 시제가 섞이지 않는지 확인
                    - **관사 사용**: a, an, the 등 관사 사용 패턴 분석
                    - **전치사 활용**: in, on, at 등 전치사 사용의 적절성 검토
                    """)
                    
                    # 학생 통합 에세이 문법 분석 결과
                    st.markdown("### 📝 학생 통합 에세이 문법 분석 결과")
                    
                    if 'step3_grammar' not in result:
                        with st.spinner("문법 오류 패턴 분석 중..."):
                            result['step3_grammar'] = preprocessor.analyze_grammar_patterns(essay_data['combined_text'])
                    
                    grammar_analysis = result.get('step3_grammar', {})
                    if grammar_analysis and 'error' not in grammar_analysis:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            grammar_score = grammar_analysis.get('grammar_score', 100)
                            st.metric("문법 정확도", f"{grammar_score:.1f}점")
                        
                        with col2:
                            total_sentences = grammar_analysis.get('total_sentences', 0)
                            sentences_with_issues = len(grammar_analysis.get('sentences_with_issues', []))
                            issue_rate = (sentences_with_issues / total_sentences * 100) if total_sentences > 0 else 0
                            st.metric("문제 문장 비율", f"{issue_rate:.1f}%")
                        
                        with col3:
                            error_types = len(grammar_analysis.get('error_count_by_type', {}))
                            st.metric("발견된 오류 유형", f"{error_types}개")
                        
                        # 주요 개선 영역
                        improvement_areas = grammar_analysis.get('improvement_areas', [])
                        if improvement_areas:
                            st.markdown("### 🎯 주요 개선 영역")
                            for i, area in enumerate(improvement_areas[:3], 1):
                                st.write(f"{i}. {area}")
                        
                        # 오류 패턴별 상세 분석
                        error_patterns = grammar_analysis.get('error_patterns', {})
                        if error_patterns:
                            st.markdown("### 📊 발견된 문법 패턴")
                            
                            for error_type, examples in error_patterns.items():
                                error_type_korean = {
                                    'subject_verb_agreement': '주어-동사 수일치',
                                    'tense_consistency': '시제 일관성',
                                    'article_usage': '관사 사용',
                                    'preposition_usage': '전치사 사용',
                                    'sentence_structure': '문장 구조',
                                    'sentence_length': '문장 길이'
                                }.get(error_type, error_type)
                                
                                with st.expander(f"📌 {error_type_korean} ({len(examples)}개 발견)", expanded=False):
                                    for i, example in enumerate(examples[:3], 1):  # 최대 3개 예시만
                                        st.write(f"**예시 {i}:**")
                                        st.write(f"• 문장: \"{example['sentence']}\"")
                                        st.write(f"• 문제점: {example['description']}")
                                        if example['suggestion']:
                                            st.write(f"• 개선 제안: {example['suggestion']}")
                                        st.write("")
                    else:
                        error_msg = grammar_analysis.get('error', '문법 분석 결과가 없습니다.')
                        st.warning(f"문법 분석 중 문제가 발생했습니다: {error_msg}")
                    
                    # 체험형 인터랙티브 요소 - 문법 검사
                    st.markdown("**🎯 체험해보기: 텍스트 마이닝으로 문법 패턴 탐지하기**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**📝 분석할 텍스트를 선택하세요:**")
                        sample_texts = [
                            "I love reading books. Books are amazing. I read books every day.",
                            "She go to school. He are happy. They was playing games.",
                            "The students study hard. A teacher explains well. An apple is red.",
                            "Yesterday I walk to school. Today I will walked home."
                        ]
                        selected_text = st.selectbox("텍스트를 선택하세요:", sample_texts, key="pattern_text")
                    
                    with col2:
                        st.markdown("**🔍 찾을 패턴을 선택하세요:**")
                        pattern_options = [
                            "주어-동사 수일치 오류",
                            "시제 일관성 문제", 
                            "관사 사용 패턴",
                            "반복 단어 빈도"
                        ]
                        selected_pattern = st.selectbox("패턴을 선택하세요:", pattern_options, key="pattern_type")
                    
                    if st.button("🔬 패턴 분석 실행", key="analyze_pattern"):
                        st.markdown("### 📊 텍스트 마이닝 분석 결과")
                        
                        import re
                        from collections import Counter
                        
                        if selected_pattern == "주어-동사 수일치 오류":
                            # 정규표현식으로 오류 패턴 탐지
                            error_patterns = [
                                (r'\bI are\b', 'I are → I am'),
                                (r'\bHe are\b|\bShe are\b', 'He/She are → He/She is'),
                                (r'\bThey is\b', 'They is → They are'),
                                (r'\bWe was\b', 'We was → We were')
                            ]
                            
                            found_errors = []
                            for pattern, correction in error_patterns:
                                matches = re.findall(pattern, selected_text, re.IGNORECASE)
                                if matches:
                                    found_errors.extend([(match, correction) for match in matches])
                            
                            if found_errors:
                                st.error("❌ 발견된 수일치 오류:")
                                for i, (error, correction) in enumerate(found_errors, 1):
                                    st.write(f"{i}. **{error}** → {correction}")
                            else:
                                st.success("✅ 주어-동사 수일치 오류가 발견되지 않았습니다!")
                                
                        elif selected_pattern == "시제 일관성 문제":
                            # 시제 관련 단어 패턴 분석
                            past_indicators = re.findall(r'\b(yesterday|ago|last|was|were|went|did)\b', selected_text.lower())
                            present_indicators = re.findall(r'\b(today|now|is|are|do|does|go|walk)\b', selected_text.lower())
                            future_indicators = re.findall(r'\b(tomorrow|will|going to)\b', selected_text.lower())
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("과거형 표현", len(past_indicators))
                                if past_indicators:
                                    st.write("발견된 단어:", ', '.join(past_indicators[:3]))
                            with col2:
                                st.metric("현재형 표현", len(present_indicators))
                                if present_indicators:
                                    st.write("발견된 단어:", ', '.join(present_indicators[:3]))
                            with col3:
                                st.metric("미래형 표현", len(future_indicators))
                                if future_indicators:
                                    st.write("발견된 단어:", ', '.join(future_indicators[:3]))
                                    
                            # 시제 일관성 분석
                            if len([x for x in [past_indicators, present_indicators, future_indicators] if x]) > 1:
                                st.warning("⚠️ 여러 시제가 혼재되어 있습니다. 문맥상 일관성을 확인해보세요.")
                            else:
                                st.success("✅ 시제 사용이 일관적입니다!")
                                
                        elif selected_pattern == "관사 사용 패턴":
                            # 관사 패턴 분석
                            articles = re.findall(r'\b(a|an|the)\b', selected_text.lower())
                            article_count = Counter(articles)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("'a' 사용", article_count.get('a', 0))
                            with col2:
                                st.metric("'an' 사용", article_count.get('an', 0))
                            with col3:
                                st.metric("'the' 사용", article_count.get('the', 0))
                                
                            # 관사 다음 단어 분석
                            article_patterns = re.findall(r'\b(a|an|the)\s+(\w+)', selected_text.lower())
                            if article_patterns:
                                st.write("**관사 + 명사 패턴:**")
                                for article, noun in article_patterns[:5]:
                                    st.write(f"• {article} {noun}")
                                    
                        elif selected_pattern == "반복 단어 빈도":
                            # 단어 빈도 분석
                            words = re.findall(r'\b[a-zA-Z]+\b', selected_text.lower())
                            word_count = Counter(words)
                            most_common = word_count.most_common(5)
                            
                            st.write("**상위 5개 빈출 단어:**")
                            for i, (word, count) in enumerate(most_common, 1):
                                st.write(f"{i}. **{word}** ({count}회)")
                                
                            # 반복도 분석
                            total_words = len(words)
                            unique_words = len(set(words))
                            repetition_rate = (total_words - unique_words) / total_words * 100 if total_words > 0 else 0
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("총 단어 수", total_words)
                            with col2:
                                st.metric("고유 단어 수", unique_words)
                            with col3:
                                st.metric("반복률", f"{repetition_rate:.1f}%")
                    
                    st.info("""
                    💡 **이 체험의 텍스트 마이닝 방법:**
                    - **정규표현식 패턴 매칭**: 특정 문법 오류 패턴을 자동으로 탐지
                    - **단어 빈도 분석**: Counter를 사용한 단어 출현 빈도 계산
                    - **패턴 분류**: 시제별, 관사별 표현을 체계적으로 분류 및 집계
                    - **통계적 분석**: 어휘 다양성, 반복률 등 수치적 지표 산출
                    
                    **실제 텍스트 마이닝**: 대규모 코퍼스에서 언어 패턴을 발견하고 분류하는 자동화 기술
                    """)
                    
                    st.markdown("---")
                    
                    # 4단계: 문장 유사도 분석 (기존 3단계에서 변경)
                    st.markdown("## 🔗 4단계: 문장 유사도 분석 - 논리적 연결성 평가")
                    st.markdown("""
                    **🔍 원리 설명:**
                    - **문장 간 연결성**: 인접 문장들 간의 의미적 유사도 측정
                    - **주제 일관성**: 글 전체의 주제 집중도와 이탈 정도 분석
                    - **논리적 흐름**: 연결어와 전환 표현을 통한 논리적 구조 평가
                    """)
                    
                    # 학생 통합 에세이 문장 유사도 분석 결과
                    st.markdown("### 🔗 학생 통합 에세이 문장 유사도 분석 결과")
                    
                    if 'step4_similarity' not in result:
                        with st.spinner("문장 유사도 분석 중..."):
                            result['step4_similarity'] = preprocessor.analyze_sentence_similarity(essay_data['combined_text'])
                    
                    similarity_analysis = result.get('step4_similarity', {})
                    if similarity_analysis:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            avg_similarity = similarity_analysis.get('average_similarity', 0)
                            st.metric("평균 문장 유사도", f"{avg_similarity:.3f}")
                            if avg_similarity > 0.6:
                                st.caption("🟢 높은 연결성")
                            elif avg_similarity > 0.3:
                                st.caption("🟡 보통 연결성")
                            else:
                                st.caption("🟠 낮은 연결성")
                        
                        with col2:
                            coherence_score = similarity_analysis.get('coherence_score', 0)
                            st.metric("글 일관성 점수", f"{coherence_score:.1f}/100")
                        
                        with col3:
                            logical_flow = similarity_analysis.get('logical_flow_level', 'Unknown')
                            st.metric("논리적 흐름", logical_flow)
                        
                        # 문장별 연결성 분석
                        sentence_pairs = similarity_analysis.get('sentence_pair_analysis', [])
                        if sentence_pairs:
                            st.markdown("### 📝 문장 간 연결성 분석 (상위 5개 쌍)")
                            for i, pair in enumerate(sentence_pairs[:5], 1):
                                similarity = pair.get('similarity', 0)
                                sent1_preview = pair.get('sentence1_preview', '문장 1')[:50] + "..."
                                sent2_preview = pair.get('sentence2_preview', '문장 2')[:50] + "..."
                                
                                if similarity > 0.5:
                                    icon = "🟢"
                                elif similarity > 0.3:
                                    icon = "🟡"
                                else:
                                    icon = "🟠"
                                
                                st.write(f"{i}. {icon} **유사도 {similarity:.3f}**")
                                st.write(f"   • {sent1_preview}")
                                st.write(f"   • {sent2_preview}")
                        
                        # 주제 일관성 분석
                        topic_consistency = similarity_analysis.get('topic_consistency', {})
                        if topic_consistency:
                            st.markdown("### 📋 주제 일관성 분석")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                topic_drift_score = topic_consistency.get('topic_drift_score', 0)
                                st.metric("주제 이탈도", f"{topic_drift_score:.1f}%")
                                
                            with col2:
                                main_theme_strength = topic_consistency.get('main_theme_strength', 0)
                                st.metric("주제 집중도", f"{main_theme_strength:.1f}%")
                    else:
                        st.warning("통합 에세이 문장 유사도 분석 결과가 없습니다.")
                    
                    # 체험형 인터랙티브 요소 - 3단계
                    st.markdown("**🎯 체험해보기: 문장 간 의미적 유사도 측정**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        sentence1 = st.text_input("첫 번째 문장:", 
                                                 value="Education plays a crucial role in personal development.",
                                                 key="similarity_sentence1")
                    with col2:
                        sentence2 = st.text_input("두 번째 문장:", 
                                                 value="Learning is essential for individual growth and success.",
                                                 key="similarity_sentence2")
                    
                    if st.button("🔍 문장 유사도 측정하기", key="similarity_check") and sentence1.strip() and sentence2.strip():
                        with st.spinner("의미적 유사도 계산 중..."):
                            # 더 정교한 유사도 계산 (다층적 접근)
                            import re
                            
                            # 1. 전처리: 단어 추출 및 정규화
                            words1 = re.findall(r'\b[a-zA-Z]+\b', sentence1.lower())
                            words2 = re.findall(r'\b[a-zA-Z]+\b', sentence2.lower())
                            
                            # 2. 의미적 클러스터 분석
                            semantic_clusters = {
                                'education': ['education', 'learning', 'study', 'school', 'knowledge', 'teach', 'learn', 'academic', 'student'],
                                'development': ['development', 'growth', 'progress', 'improve', 'advance', 'evolve', 'enhance', 'success'],
                                'importance': ['crucial', 'important', 'essential', 'vital', 'significant', 'necessary', 'key', 'critical'],
                                'personal': ['personal', 'individual', 'self', 'own', 'private', 'person', 'people', 'human'],
                                'process': ['process', 'method', 'way', 'approach', 'system', 'procedure', 'technique', 'strategy']
                            }
                            
                            # 3. 의미적 유사도 계산
                            semantic_matches = {}
                            for cluster_name, cluster_words in semantic_clusters.items():
                                words1_in_cluster = [w for w in words1 if w in cluster_words]
                                words2_in_cluster = [w for w in words2 if w in cluster_words]
                                if words1_in_cluster or words2_in_cluster:
                                    semantic_matches[cluster_name] = {
                                        'sentence1': words1_in_cluster,
                                        'sentence2': words2_in_cluster,
                                        'match': bool(words1_in_cluster and words2_in_cluster)
                                    }
                            
                            # 4. 구조적 유사성 분석
                            structure_score = 0
                            # 문장 길이 유사성
                            len_diff = abs(len(words1) - len(words2))
                            length_similarity = max(0, 1 - len_diff / max(len(words1), len(words2), 1))
                            structure_score += length_similarity * 30
                            
                            # 품사 패턴 유사성 (간단 추정)
                            def estimate_pos_pattern(words):
                                pattern = []
                                for word in words:
                                    if word.endswith('ing') or word.endswith('ed'):
                                        pattern.append('V')  # 동사
                                    elif word.endswith('tion') or word.endswith('ment'):
                                        pattern.append('N')  # 명사
                                    elif word.endswith('ly'):
                                        pattern.append('R')  # 부사
                                    elif word in ['a', 'an', 'the']:
                                        pattern.append('D')  # 관사
                                    else:
                                        pattern.append('N')  # 기본 명사
                                return pattern[:5]  # 첫 5개 패턴만
                            
                            pattern1 = estimate_pos_pattern(words1)
                            pattern2 = estimate_pos_pattern(words2)
                            pattern_similarity = sum(1 for a, b in zip(pattern1, pattern2) if a == b) / max(len(pattern1), len(pattern2), 1)
                            structure_score += pattern_similarity * 20
                            
                            # 5. 종합 유사도 점수
                            semantic_score = len([m for m in semantic_matches.values() if m['match']]) * 20
                            word_overlap_score = len(set(words1).intersection(set(words2))) / len(set(words1).union(set(words2))) * 30 if words1 or words2 else 0
                            
                            total_similarity = (semantic_score + structure_score + word_overlap_score) / 100
                            total_similarity = min(1.0, total_similarity)  # 최대 1.0으로 제한
                            
                            # 6. 결과 표시
                            st.write(f"**문장 1:** {sentence1}")
                            st.write(f"**문장 2:** {sentence2}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("의미적 유사도", f"{total_similarity:.3f}")
                            with col2:
                                if total_similarity > 0.6:
                                    st.metric("연결성 판정", "🟢 강함")
                                elif total_similarity > 0.3:
                                    st.metric("연결성 판정", "🟡 보통")
                                else:
                                    st.metric("연결성 판정", "🟠 약함")
                            with col3:
                                common_themes = len([m for m in semantic_matches.values() if m['match']])
                                st.metric("공통 주제", f"{common_themes}개")
                            
                            # 7. 상세 분석 결과
                            if semantic_matches:
                                st.markdown("**🎯 의미적 연결 분석:**")
                                for theme, data in semantic_matches.items():
                                    if data['match']:
                                        st.write(f"✅ **{theme.capitalize()}** 주제 연결: {', '.join(data['sentence1'])} ↔ {', '.join(data['sentence2'])}")
                                    elif data['sentence1'] or data['sentence2']:
                                        words = data['sentence1'] or data['sentence2']
                                        st.write(f"⚠️ **{theme.capitalize()}** 주제 (한쪽만): {', '.join(words)}")
                            
                            # 8. 구조적 분석
                            st.markdown("**📋 구조적 분석:**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"• 길이 유사성: {length_similarity:.2f}")
                                st.write(f"• 패턴 유사성: {pattern_similarity:.2f}")
                            with col2:
                                st.write(f"• 어휘 중복도: {word_overlap_score/30:.2f}")
                                st.write(f"• 의미 클러스터: {semantic_score/20:.0f}개 일치")
                            
                            st.info("""
                            💡 **이 체험의 분석 방법:**
                            - **의미적 클러스터**: 관련 단어들을 주제별로 그룹화하여 의미 연결성 측정
                            - **구조적 유사성**: 문장 길이, 품사 패턴 등 구조적 특징 비교
                            - **종합 점수**: 여러 차원의 유사성을 종합하여 최종 점수 산출
                            
                            **실제 AI 시스템**: BERT, Sentence-BERT 등을 사용하여 768차원 벡터 공간에서 코사인 유사도 계산
                            """)
                    
                    st.markdown("---")
                    
                    # 종합 진단 결과
                    comprehensive_results = result.get('step5_comprehensive', {})
                    if comprehensive_results:
                        st.success("🎊 **종합 진단 완료!**")
                        
                        # 종합 점수 표시
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            overall_score = comprehensive_results.get('overall_score', 0)
                            st.metric("종합 점수", f"{overall_score}/100")
                        
                        with col2:
                            final_level = comprehensive_results.get('final_level', 'Unknown')
                            st.metric("등급", final_level)
                        
                        with col3:
                            level_description = comprehensive_results.get('level_description', '')
                            st.write(f"**평가:** {level_description}")
                        
                        # 상세 분석 결과
                        st.markdown("### 📋 상세 분석 결과")
                        
                        # 강점과 약점 분석
                        strengths = comprehensive_results.get('strengths', [])
                        weaknesses = comprehensive_results.get('weaknesses', [])
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**🟢 강점 분석:**")
                            if strengths:
                                for strength in strengths[:3]:
                                    st.write(f"• {strength}")
                            else:
                                st.write("• 문장 구성 능력")
                                st.write("• 기본적인 어휘 사용")
                        
                        with col2:
                            st.markdown("**🟠 개선점:**")
                            if weaknesses:
                                for weakness in weaknesses[:3]:
                                    st.write(f"• {weakness}")
                            else:
                                st.write("• 어휘 다양성 향상 필요")
                                st.write("• 복합문 구조 연습")
                                st.write("• 논리적 연결성 강화")
                        
                        # 맞춤형 학습 계획
                        st.markdown("### 🎯 맞춤형 학습 계획")
                        recommendations = comprehensive_results.get('recommendations', [])
                        
                        if recommendations:
                            for i, rec in enumerate(recommendations[:4], 1):
                                st.write(f"{i}. **{rec}**")
                        else:
                            st.write("1. **어휘력 향상**: 다양한 동의어와 전문 용어 학습")
                            st.write("2. **문장 구조 다양화**: 복합문과 복문 사용 연습") 
                            st.write("3. **논리적 연결**: 접속사를 활용한 문장 간 연결성 강화")
                            st.write("4. **독서 확대**: 다양한 장르의 영문 텍스트 읽기")
                    
                    # 학생 주도형 학습 계획 생성 활동
                    st.markdown("### 🎯 나만의 맞춤형 학습 계획 만들기")
                    st.info("💡 분석 결과를 바탕으로 스스로 학습 계획을 세워보세요. 단계적 안내를 통해 나만의 목표를 찾아갈 수 있습니다.")
                    
                    # 1단계: 자기 반성 및 현재 상태 파악
                    st.markdown("#### 📋 1단계: 내 글쓰기 현재 상태 분석하기")
                    
                    # 자기 반성 질문
                    st.markdown("**🤔 스스로에게 물어보기:**")
                    reflection_questions = [
                        "내 글쓰기에서 가장 만족스러운 부분은 무엇인가요?",
                        "어떤 부분에서 어려움을 느끼나요?",
                        "글쓰기 실력을 향상시키고 싶은 이유는 무엇인가요?"
                    ]
                    
                    reflection_answers = {}
                    for i, question in enumerate(reflection_questions):
                        reflection_answers[f"q{i+1}"] = st.text_area(
                            question, 
                            placeholder="솔직하게 생각해보고 적어주세요...",
                            key=f"reflection_{i+1}",
                            height=80
                        )
                    
                    # 2단계: 목표 설정
                    st.markdown("#### 🎯 2단계: 구체적인 목표 정하기")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**우선순위 정하기:**")
                        priority_areas = st.multiselect(
                            "가장 집중하고 싶은 영역을 1-2개 선택하세요:",
                            ["어휘 다양성 늘리기", "고급 어휘 사용하기", "문장 연결성 강화하기", 
                             "문법 정확성 높이기", "창의적 표현 늘리기", "논리적 구조 만들기"],
                            key="priority_areas"
                        )
                    
                    with col2:
                        st.markdown("**학습 스타일:**")
                        learning_style = st.radio(
                            "나에게 맞는 학습 방식은:",
                            ["매일 조금씩 꾸준히", "주말에 집중적으로", "친구들과 함께", "혼자 차근차근"],
                            key="learning_style"
                        )
                    
                    # 목표 구체화
                    target_period = st.selectbox(
                        "언제까지 목표를 달성하고 싶나요?",
                        ["2주 후", "4주 후", "8주 후", "학기말까지"],
                        key="target_period"
                    )
                    
                    success_measure = st.text_input(
                        "목표 달성을 어떻게 확인할 건가요? (예: 에세이 점수 10점 향상, 새로운 단어 100개 사용)",
                        key="success_measure",
                        placeholder="구체적인 성공 기준을 적어보세요..."
                    )
                    
                    # 3단계: 실행 계획 만들기
                    if priority_areas and st.button("📝 나만의 학습 계획 완성하기", key="generate_personal_plan"):
                        st.markdown("#### ✨ 3단계: 완성된 나만의 학습 계획")
                        
                        # 버튼 클릭 시에만 풍선 표시 (세션별 고유 키 사용)
                        balloon_key = f"learning_plan_balloons_{hash(str(priority_areas) + target_period + learning_style)}"
                        if balloon_key not in st.session_state:
                            st.balloons()
                            st.session_state[balloon_key] = True
                        
                        # 학생이 선택한 내용을 바탕으로 개인화된 계획 제시
                        st.success("🎊 축하합니다! 나만의 맞춤형 학습 계획이 완성되었습니다.")
                        
                        plan_container = st.container()
                        with plan_container:
                            st.markdown("---")
                            st.markdown(f"### 📋 {st.session_state.get('user_name', '나')}의 글쓰기 향상 계획")
                            
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.write(f"**🎯 주요 목표:** {', '.join(priority_areas)}")
                                st.write(f"**⏰ 목표 달성 기한:** {target_period}")
                                st.write(f"**📏 성공 기준:** {success_measure if success_measure else '꾸준한 연습과 자기 평가'}")
                            with col2:
                                st.write(f"**📚 학습 방식:** {learning_style}")
                                st.write(f"**💪 집중 영역:** {len(priority_areas)}개 분야")
                            
                            # 구체적 실행 방법 제안 (선택한 영역별로)
                            st.markdown("### 📝 주간별 실행 계획")
                            
                            # 중학교 1학년 수준 맞춤 학습 계획
                            action_plans = {
                                "어휘 다양성 늘리기": {
                                    "주간 목표": "새로운 영어 단어 15개 확실히 외우고 사용해보기",
                                    "일일 활동": "📚 **매일 10분 어휘 학습**\n• EBS 중학영어나 능률교육 사이트에서 중1 수준 영어동화 1개 읽기\n• 모르는 단어 3개를 공책에 크게 써서 뜻과 발음기호 적기\n• 새로 배운 단어로 '나는 ___이다' 형식의 쉬운 문장 만들기\n• 스마트폰 단어장 앱(예: 네이버사전)에 새 단어 저장하고 소리내어 읽기",
                                    "체크포인트": "🎯 **주간 확인 활동**\n• 수요일: 이번 주 배운 단어 10개로 내 일상 소개하는 5문장 만들기\n• 금요일: 부모님/형제에게 새로 배운 단어 3개 영어로 설명해드리기\n• 일요일: 지난 주 단어 5개를 친구에게 퀴즈로 내고 맞추기 게임"
                                },
                                "고급 어휘 사용하기": {
                                    "주간 목표": "simple한 단어를 좀 더 멋진 단어로 바꿔 쓰기",
                                    "일일 활동": "✨ **매일 10분 단어 업그레이드**\n• 내가 쓴 영어 일기나 숙제에서 easy, good, bad 같은 쉬운 단어 찾기\n• 중학영어 교과서 뒷편 단어장에서 비슷한 뜻의 어려운 단어 찾기\n• 네이버 영어사전에서 예문 2개 읽고 따라 써보기\n• 바뀐 문장을 큰 소리로 읽으면서 어색한 곳 없는지 확인하기",
                                    "체크포인트": "🎯 **주간 확인 활동**\n• 화요일: 'My hobby is...' 문장을 basic 단어와 advanced 단어로 각각 써보기\n• 금요일: 영어선생님께 내가 바꾼 문장이 자연스러운지 물어보기\n• 일요일: 이번 주 배운 멋진 단어 10개로 단어카드 만들어 벽에 붙이기"
                                },
                                "문장 연결성 강화하기": {
                                    "주간 목표": "짧은 문장들을 자연스럽게 이어서 긴 문장 만들기",
                                    "일일 활동": "🔗 **매일 10분 문장 연결 연습**\n• and, but, so, because 같은 연결어 하나씩 집중 학습\n• 중학교 영어교과서에서 연결어가 들어간 문장 5개 찾아 공책에 쓰기\n• 'I like pizza.' + 'Pizza is delicious.' → 'I like pizza because it is delicious.' 연습\n• 만든 문장을 녹음기로 녹음해서 들어보고 자연스러운지 확인",
                                    "체크포인트": "🎯 **주간 확인 활동**\n• 목요일: 내 하루 일과를 5개 짧은 문장으로 쓴 다음 연결어로 3문장으로 줄이기\n• 토요일: 친구와 서로 쓴 문장 읽어주고 이상한 곳 찾아주기\n• 일요일: 좋아하는 연예인/운동선수 소개를 연결어 사용해서 100자로 쓰기"
                                },
                                "문법 정확성 높이기": {
                                    "주간 목표": "be동사, 일반동사, 복수형 실수 없이 쓰기",
                                    "일일 활동": "📝 **매일 10분 문법 점검**\n• EBS 중학영어 또는 '문법이 쉬워지는 영어' 앱에서 오늘의 문법 1개 학습\n• 학습한 문법으로 내 경험 문장 3개 만들기 (예: am/is/are 구분해서)\n• 어제 쓴 영어 문장에서 틀린 부분 빨간펜으로 고치기\n• 고친 부분을 노트에 '자주 틀리는 실수' 목록으로 만들기",
                                    "체크포인트": "🎯 **주간 확인 활동**\n• 수요일: 내 소개 5문장 쓰고 be동사/일반동사 색깔펜으로 구분하기\n• 금요일: 짝꿍과 서로 쓴 문장 문법 검사해주고 틀린 곳 설명해주기\n• 일요일: 이번 주 가장 많이 틀린 문법 1개 골라서 예문 10개 만들기"
                                },
                                "창의적 표현 늘리기": {
                                    "주간 목표": "딱딱한 문장 대신 재미있고 생동감 있는 표현 쓰기",
                                    "일일 활동": "🎨 **매일 10분 표현 연습**\n• 디즈니 영화나 스튜디오 지브리 영화 명대사 1개 찾아 따라 쓰기\n• '비가 온다' 대신 'Rain is dancing'처럼 의인법으로 표현해보기\n• 감정을 색깔로 표현하기 (I'm feeling blue = 슬프다)\n• 내가 만든 창의적 문장을 가족에게 읽어주고 반응 보기",
                                    "체크포인트": "🎯 **주간 확인 활동**\n• 화요일: '학교에 간다'를 3가지 다른 재미있는 방식으로 표현하기\n• 금요일: 좋아하는 음식을 의인법이나 비유법으로 소개하는 문장 쓰기\n• 일요일: 이번 주 날씨를 창의적 표현으로 일기 3줄 쓰기"
                                },
                                "논리적 구조 만들기": {
                                    "주간 목표": "First, Next, Finally 순서로 차근차근 설명하는 글쓰기",
                                    "일일 활동": "📊 **매일 10분 순서 정리 연습**\n• 라면 끓이는 방법, 학교 가는 길 등 일상적인 순서를 First, Next, Finally로 쓰기\n• 중학교 영어교과서 reading 부분에서 순서를 나타내는 표현 찾기\n• 내 의견을 I think → because → For example 순서로 설명하는 연습\n• 쓴 글을 소리내어 읽으며 순서가 논리적인지 확인하기",
                                    "체크포인트": "🎯 **주간 확인 활동**\n• 수요일: 좋아하는 게임/취미 방법을 순서대로 영어 5문장으로 설명하기\n• 토요일: 친구에게 내 설명을 읽어주고 이해하기 쉬운지 물어보기\n• 일요일: 우리 학교 자랑거리를 First, Second, Third로 나눠서 소개하기"
                                }
                            }
                            
                            weeks_mapping = {"2주 후": 2, "4주 후": 4, "8주 후": 8, "학기말까지": 12}
                            weeks = weeks_mapping.get(target_period, 4)
                            
                            for i, area in enumerate(priority_areas, 1):
                                if area in action_plans:
                                    plan = action_plans[area]
                                    with st.expander(f"📌 영역 {i}: {area}", expanded=True):
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.write(f"**주간 목표:**")
                                            st.write(plan["주간 목표"])
                                        with col2:
                                            st.write(f"**일일 활동:**")
                                            st.write(plan["일일 활동"])
                                        with col3:
                                            st.write(f"**체크포인트:**")
                                            st.write(plan["체크포인트"])
                            
                            # 학습 스타일에 따른 맞춤 조언
                            st.markdown("### 💡 나만의 학습 전략")
                            style_advice = {
                                "매일 조금씩 꾸준히": "⏰ 매일 15분씩 정해진 시간에 학습하고, 스마트폰 알림 설정하기",
                                "주말에 집중적으로": "📅 토요일/일요일 2시간씩 집중 학습 시간 확보하고, 주중엔 복습만",
                                "친구들과 함께": "👥 스터디 그룹 만들어 서로 글 검토해주고, 영어 대화 연습하기",
                                "혼자 차근차근": "📚 조용한 환경에서 집중하고, 학습 일지 작성해 스스로 점검하기"
                            }
                            
                            st.info(f"**{learning_style} 스타일 맞춤 조언:** {style_advice.get(learning_style, '자신만의 페이스로 꾸준히 학습하세요!')}")
                            
                            # 동기부여 및 점검 시스템
                            st.markdown("### 🏆 성공을 위한 체크리스트")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**📈 주간 점검 질문:**")
                                st.write("□ 이번 주 목표를 달성했나요?")
                                st.write("□ 어떤 부분이 가장 어려웠나요?")
                                st.write("□ 실제로 실력이 늘었다고 느끼나요?")
                                st.write("□ 다음 주 계획을 조정할 부분이 있나요?")
                            
                            with col2:
                                st.markdown("**🎯 동기부여 방법:**")
                                st.write("• 작은 성취도 기록하고 축하하기")
                                st.write("• 학습 전후 글쓰기 비교해보기")
                                st.write("• 힘들 때 학습 목적 다시 떠올리기")
                                st.write(f"• {target_period} 후 성취감 미리 상상하기")
                            
                            # 응급처치 계획
                            st.markdown("### 🆘 학습이 어려울 때 대처법")
                            emergency_tips = [
                                "**동기가 떨어질 때**: 처음에 세운 목표와 이유를 다시 읽어보기",
                                "**시간이 부족할 때**: 하루 5분이라도 영어 문장 하나 읽고 분석하기",
                                "**너무 어려울 때**: 목표를 작은 단위로 나누어 달성 가능한 수준으로 조정하기",
                                "**혼자 하기 힘들 때**: 선생님께 조언 구하거나 온라인 학습 커뮤니티 활용하기"
                            ]
                            
                            for tip in emergency_tips:
                                st.write(f"• {tip}")
                            
                            st.markdown("---")
                            st.success("🌟 **기억하세요**: 완벽한 계획보다 꾸준한 실행이 더 중요합니다. 나만의 속도로 차근차근 성장해나가세요!")
                    
                else:
                    st.error("종합 글쓰기 진단에 필요한 데이터가 부족합니다.")
                    
            except Exception as e:
                st.error(f"❌ 종합 글쓰기 진단 에러: {type(e).__name__}: {str(e)}")
        else:
            st.warning("📚 먼저 '에세이 수집' 탭에서 에세이 데이터를 불러와주세요.")
    
    # 완료 안내
    if 'educational_sentiment_results' in st.session_state or 'educational_pos_results' in st.session_state:
        st.markdown("---")
        st.info("🎊 텍스트 마이닝 원리 체험을 완료하셨습니다! 각 분석은 독립적으로 실행할 수 있습니다.")

# CSS 스타일
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #2E86AB;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# 메인 실행
def main():
    # 세션 상태 초기화
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    # 로그인 상태에 따라 페이지 표시
    if st.session_state.logged_in:
        main_app()
    else:
        login_page()

if __name__ == "__main__":
    main()