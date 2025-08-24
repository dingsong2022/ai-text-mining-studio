import streamlit as st
import nltk
import re
from collections import Counter
import pandas as pd

# NLTK 데이터 다운로드 (안정화 버전)
@st.cache_resource
def download_nltk_data():
    try:
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # 필수 NLTK 데이터 다운로드
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)
        
        # 📍 새로 추가! 📍
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        except LookupError:
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
            
        return True
    except Exception as e:
        st.error(f"NLTK 데이터 다운로드 오류: {e}")
        return False

class TextPreprocessor:
    def __init__(self):
        # NLTK 데이터 다운로드
        download_nltk_data()
        
        try:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))
        except:
            # 기본 불용어 리스트
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
    
    def extract_essay_content(self, text):
        """에세이 내용만 추출 (평가 결과 부분 제외)"""
        if not text or pd.isna(text):
            return ""
        
        text = str(text)
        
        # **EVALUATION RESULTS** 부분 제거
        text = re.sub(r'\*\*EVALUATION RESULTS\*\*.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'\*\*.*?\*\*', '', text)
        
        # 점수 패턴 제거 (예: "점수: 85점")
        text = re.sub(r'점수\s*:\s*\d+점?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'score\s*:\s*\d+', '', text, flags=re.IGNORECASE)
        
        # 기본 정제
        text = re.sub(r'<[^>]+>', '', text)  # HTML 태그
        text = re.sub(r'\s+', ' ', text)     # 연속 공백
        
        return text.strip()
    
    def basic_cleaning(self, text):
        """기본 텍스트 정제"""
        if not text or pd.isna(text):
            return ""
        
        text = str(text)
        
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # **EVALUATION RESULTS** 같은 평가 결과 부분 제거
        text = re.sub(r'\*\*.*?\*\*', '', text)
        
        # 특수문자 정리 (문장부호는 유지)
        text = re.sub(r'[^\w\s.,!?;:\'-]', ' ', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def step1_basic_cleaning(self, text):
        """1단계: 기본 정제 (HTML, 특수문자, 공백 정리)"""
        if not text or pd.isna(text):
            return ""
        
        text = str(text)
        
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # **EVALUATION RESULTS** 같은 평가 결과 부분 제거
        text = re.sub(r'\*\*.*?\*\*', '', text)
        
        # 특수문자 정리 (문장부호는 유지)
        text = re.sub(r'[^\w\s.,!?;:\'-]', ' ', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 양옆 공백 제거
        text = text.strip()
        
        return text
    
    def step2_remove_stopwords(self, text):
        """2단계: 불용어 제거"""
        if not text:
            return ""
        
        try:
            words = nltk.word_tokenize(text.lower())
        except:
            words = text.lower().split()
        
        # 불용어 제거 + 짧은 단어 제거
        filtered_words = []
        for word in words:
            if (word.isalpha() and 
                len(word) > 2 and 
                word not in self.stop_words):
                filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def step3_stemming(self, text):
        """3단계: 어간 추출 (Stemming)"""
        if not text:
            return ""
        
        try:
            from nltk.stem import PorterStemmer
            stemmer = PorterStemmer()
            
            words = text.split()
            stemmed_words = [stemmer.stem(word) for word in words]
            return ' '.join(stemmed_words)
        except:
            # NLTK가 없으면 그대로 반환
            return text
    
    def step4_lemmatization(self, text):
        """4단계: 표제어 추출 (Lemmatization) - 더 정확함"""
        if not text:
            return ""
        
        try:
            from nltk.stem import WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()
            
            words = text.split()
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
            return ' '.join(lemmatized_words)
        except:
            # NLTK가 없으면 어간 추출로 대체
            return self.step3_stemming(text)
    
    def get_preprocessing_steps(self, text):
        """모든 전처리 단계별 결과 반환"""
        steps = {}
        
        # 원본
        steps['원본'] = text
        steps['원본_단어수'] = len(text.split()) if text else 0
        
        # 1단계: 기본 정제
        step1 = self.step1_basic_cleaning(text)
        steps['1단계_기본정제'] = step1
        steps['1단계_단어수'] = len(step1.split()) if step1 else 0
        
        # 2단계: 불용어 제거
        step2 = self.step2_remove_stopwords(step1)
        steps['2단계_불용어제거'] = step2
        steps['2단계_단어수'] = len(step2.split()) if step2 else 0
        
        # 3단계: 어간 추출
        step3 = self.step3_stemming(step2)
        steps['3단계_어간추출'] = step3
        steps['3단계_단어수'] = len(step3.split()) if step3 else 0
        
        # 4단계: 표제어 추출
        step4 = self.step4_lemmatization(step2)  # step2에서 바로 진행
        steps['4단계_표제어추출'] = step4
        steps['4단계_단어수'] = len(step4.split()) if step4 else 0
        
        return steps
    
    # ===========================================
    # 기본 감성 분석 (3가지: 긍정/부정/중립)
    # ===========================================
    
    def sentiment_analysis(self, text):
        """개별 텍스트의 기본 감성 분석"""
        if not text or pd.isna(text):
            return {'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 0}
        
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            
            # 에세이 내용만 추출
            cleaned_text = self.extract_essay_content(text)
            if not cleaned_text:
                return {'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 0}
            
            scores = analyzer.polarity_scores(cleaned_text)
            return {
                'compound': scores['compound'],  # 전체 감성 점수 (-1~1)
                'positive': scores['pos'],       # 긍정 비율
                'negative': scores['neg'],       # 부정 비율  
                'neutral': scores['neu']         # 중립 비율
            }
        except Exception as e:
            st.warning(f"감성 분석 중 오류: {e}")
            return {'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 0}

    def analyze_all_essays_sentiment(self, essay_data):
        """모든 에세이의 기본 감성 분석"""
        results = []
        
        for idx, row in essay_data.iterrows():
            essay_text = row.get('essay_text', '')
            topic_name = row.get('topic_name', f'Essay {idx+1}')
            created_at = row.get('created_at', '')
            
            sentiment = self.sentiment_analysis(essay_text)
            
            # 감성 레이블 결정
            compound = sentiment['compound']
            if compound >= 0.05:
                sentiment_label = "긍정적"
                emoji = "😊"
            elif compound <= -0.05:
                sentiment_label = "부정적" 
                emoji = "😟"
            else:
                sentiment_label = "중립적"
                emoji = "😐"
            
            results.append({
                'essay_num': idx + 1,
                'topic_name': topic_name,
                'created_at': created_at,
                'sentiment_label': sentiment_label,
                'emotion_label': sentiment_label,  # 호환성을 위해 추가
                'emoji': emoji,
                'compound': compound,
                'positive': sentiment['positive'],
                'negative': sentiment['negative'],
                'neutral': sentiment['neutral']
            })
        
        return results

    def get_sentiment_statistics(self, sentiment_results):
        """기본 감성 분석 통계"""
        if not sentiment_results:
            return {}
        
        total_essays = len(sentiment_results)
        positive_count = sum(1 for r in sentiment_results if r['sentiment_label'] == "긍정적")
        negative_count = sum(1 for r in sentiment_results if r['sentiment_label'] == "부정적")
        neutral_count = sum(1 for r in sentiment_results if r['sentiment_label'] == "중립적")
        
        avg_compound = sum(r['compound'] for r in sentiment_results) / total_essays
        
        return {
            'total_essays': total_essays,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'positive_ratio': positive_count / total_essays * 100,
            'negative_ratio': negative_count / total_essays * 100,
            'neutral_ratio': neutral_count / total_essays * 100,
            'avg_compound': avg_compound
        }
    
    # ===========================================
    # 향상된 감성 분석 (8가지 감정)
    # ===========================================
    
    def enhanced_sentiment_analysis(self, text):
        """향상된 감성 분석 - 8가지 감정 분류"""
        if not text or pd.isna(text):
            return {}
        
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            
            # 에세이 내용만 추출
            cleaned_text = self.extract_essay_content(text)
            if not cleaned_text:
                return {}
            
            # VADER 감성 점수
            vader_scores = analyzer.polarity_scores(cleaned_text)
            
            # 키워드 기반 감정 분석
            emotion_keywords = {
                'joy': ['happy', 'joy', 'excited', 'wonderful', 'amazing', 'great', 'excellent', 'fantastic', 'love', 'enjoy'],
                'anger': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'hate', 'terrible', 'awful', 'worst', 'disgusting'],
                'sadness': ['sad', 'depressed', 'lonely', 'crying', 'disappointed', 'hurt', 'pain', 'sorrow', 'grief', 'unhappy'],
                'fear': ['afraid', 'scared', 'worried', 'anxious', 'nervous', 'terrified', 'panic', 'frightened', 'concerned', 'stress'],
                'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected', 'sudden', 'wow', 'incredible', 'unbelievable'],
                'disgust': ['disgusted', 'sick', 'revolting', 'gross', 'horrible', 'disgusting', 'repulsive', 'nasty'],
                'trust': ['trust', 'believe', 'reliable', 'confident', 'faith', 'honest', 'sincere', 'loyal', 'dependable'],
                'anticipation': ['hope', 'expect', 'anticipate', 'looking forward', 'eager', 'excited', 'optimistic', 'future']
            }
            
            # 텍스트를 소문자로 변환
            lower_text = cleaned_text.lower()
            
            # 각 감정별 키워드 개수 계산
            emotion_scores = {}
            total_emotion_words = 0
            
            for emotion, keywords in emotion_keywords.items():
                count = sum(lower_text.count(keyword) for keyword in keywords)
                emotion_scores[emotion] = count
                total_emotion_words += count
            
            # 비율 계산
            emotion_ratios = {}
            for emotion, count in emotion_scores.items():
                emotion_ratios[emotion] = (count / total_emotion_words * 100) if total_emotion_words > 0 else 0
            
            # 주요 감정 결정
            if total_emotion_words > 0:
                primary_emotion = max(emotion_scores, key=emotion_scores.get)
                primary_emotion_score = emotion_scores[primary_emotion]
            else:
                primary_emotion = "neutral"
                primary_emotion_score = 0
            
            # VADER와 키워드 분석 결합
            compound = vader_scores['compound']
            
            # 최종 감정 분류
            if primary_emotion_score > 2:  # 강한 감정 키워드가 많으면
                final_emotion = primary_emotion
                confidence = "높음"
            elif compound >= 0.1:
                final_emotion = "positive"
                confidence = "중간"
            elif compound <= -0.1:
                final_emotion = "negative"
                confidence = "중간"
            else:
                final_emotion = "neutral"
                confidence = "낮음"
            
            return {
                'vader_scores': vader_scores,
                'emotion_scores': emotion_scores,
                'emotion_ratios': emotion_ratios,
                'primary_emotion': primary_emotion,
                'final_emotion': final_emotion,
                'confidence': confidence,
                'total_emotion_words': total_emotion_words
            }
            
        except Exception as e:
            st.warning(f"향상된 감성 분석 중 오류: {e}")
            return {}

    def analyze_all_essays_enhanced_sentiment(self, essay_data):
        """모든 에세이의 향상된 감성 분석"""
        results = []
        
        for idx, row in essay_data.iterrows():
            essay_text = row.get('essay_text', '')
            topic_name = row.get('topic_name', f'Essay {idx+1}')
            created_at = row.get('created_at', '')
            
            sentiment = self.enhanced_sentiment_analysis(essay_text)
            
            if sentiment:
                # 감성 라벨과 이모지 결정
                emotion_emojis = {
                    'joy': '😊', 'anger': '😠', 'sadness': '😢', 'fear': '😰',
                    'surprise': '😲', 'disgust': '🤢', 'trust': '🤝', 'anticipation': '🤗',
                    'positive': '😊', 'negative': '😟', 'neutral': '😐'
                }
                
                final_emotion = sentiment.get('final_emotion', 'neutral')
                emoji = emotion_emojis.get(final_emotion, '😐')
                
                # 한국어 감정명
                emotion_korean = {
                    'joy': '기쁨', 'anger': '분노', 'sadness': '슬픔', 'fear': '두려움',
                    'surprise': '놀라움', 'disgust': '혐오', 'trust': '신뢰', 'anticipation': '기대',
                    'positive': '긍정적', 'negative': '부정적', 'neutral': '중립적'
                }
                
                emotion_label = emotion_korean.get(final_emotion, '중립적')
                
                results.append({
                    'essay_num': idx + 1,
                    'topic_name': topic_name,
                    'created_at': created_at,
                    'emotion_label': emotion_label,
                    'emoji': emoji,
                    'final_emotion': final_emotion,
                    'confidence': sentiment.get('confidence', '낮음'),
                    'compound': sentiment['vader_scores']['compound'],
                    'emotion_scores': sentiment.get('emotion_scores', {}),
                    'primary_emotion': sentiment.get('primary_emotion', 'neutral')
                })
        
        return results

    def get_enhanced_sentiment_statistics(self, sentiment_results):
        """향상된 감성 분석 통계"""
        if not sentiment_results:
            return {}
        
        total_essays = len(sentiment_results)
        
        # 감정별 개수 계산
        emotion_counts = {}
        for result in sentiment_results:
            emotion = result['final_emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # 평균 compound 점수
        avg_compound = sum(r['compound'] for r in sentiment_results) / total_essays
        
        # 가장 많은 감정
        most_common_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'neutral'
        
        return {
            'total_essays': total_essays,
            'emotion_counts': emotion_counts,
            'avg_compound': avg_compound,
            'most_common_emotion': most_common_emotion,
            'emotion_distribution': {k: v/total_essays*100 for k, v in emotion_counts.items()}
        }
    
    # ===========================================
    # 품사 분석 (안정화 버전)
    # ===========================================
    
    def advanced_pos_analysis(self, text):
        """고급 품사 분석 (안전한 버전)"""
        if not text or pd.isna(text):
            return {}
        
        try:
            # 에세이 내용만 추출
            cleaned_text = self.extract_essay_content(text)
            if not cleaned_text:
                return {}
            
            # 기본 토큰화 시도
            try:
                sentences = nltk.sent_tokenize(cleaned_text)
            except:
                # NLTK가 실패하면 간단한 문장 분리
                sentences = [s.strip() + '.' for s in cleaned_text.split('.') if s.strip()]
            
            all_pos_tags = []
            pos_counts = {}
            
            for sentence in sentences:
                try:
                    # 단어 토큰화 및 품사 태깅
                    words = nltk.word_tokenize(sentence)
                    pos_tags = nltk.pos_tag(words)
                except:
                    # NLTK가 실패하면 간단한 단어 분리 및 기본 태깅
                    words = sentence.split()
                    pos_tags = [(word, 'NN' if word[0].isupper() else 'VB' if word.endswith('ed') else 'JJ' if word.endswith('ly') else 'NN') for word in words if word.isalpha()]
                
                all_pos_tags.extend(pos_tags)
                
                # 품사별 카운트
                for word, pos in pos_tags:
                    if pos not in pos_counts:
                        pos_counts[pos] = 0
                    pos_counts[pos] += 1
            
            # 주요 품사별 분류
            noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']  # 명사
            verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']  # 동사
            adjective_tags = ['JJ', 'JJR', 'JJS']  # 형용사
            adverb_tags = ['RB', 'RBR', 'RBS']  # 부사
            
            categorized = {
                'nouns': sum(pos_counts.get(tag, 0) for tag in noun_tags),
                'verbs': sum(pos_counts.get(tag, 0) for tag in verb_tags),
                'adjectives': sum(pos_counts.get(tag, 0) for tag in adjective_tags),
                'adverbs': sum(pos_counts.get(tag, 0) for tag in adverb_tags),
                'total_words': len(all_pos_tags)
            }
            
            # 비율 계산
            total = categorized['total_words']
            if total > 0:
                categorized['noun_ratio'] = categorized['nouns'] / total * 100
                categorized['verb_ratio'] = categorized['verbs'] / total * 100
                categorized['adjective_ratio'] = categorized['adjectives'] / total * 100
                categorized['adverb_ratio'] = categorized['adverbs'] / total * 100
            else:
                categorized['noun_ratio'] = categorized['verb_ratio'] = categorized['adjective_ratio'] = categorized['adverb_ratio'] = 0
            
            return {
                'detailed_pos': pos_counts,
                'categorized': categorized,
                'pos_tags': all_pos_tags
            }
            
        except Exception as e:
            st.warning(f"품사 분석 중 오류: {e}")
            return {}

    def analyze_all_essays_pos(self, essay_data):
        """모든 에세이의 품사 분석"""
        all_results = []
        
        for idx, row in essay_data.iterrows():
            essay_text = row.get('essay_text', '')
            topic_name = row.get('topic_name', f'Essay {idx+1}')
            
            pos_result = self.advanced_pos_analysis(essay_text)
            
            if pos_result and 'categorized' in pos_result:
                result = {
                    'essay_num': idx + 1,
                    'topic_name': topic_name,
                    **pos_result['categorized']
                }
                all_results.append(result)
        
        return all_results

    def get_pos_statistics(self, pos_results):
        """품사 분석 종합 통계"""
        if not pos_results:
            return {}
        
        total_essays = len(pos_results)
        
        # 평균 계산
        avg_noun_ratio = sum(r['noun_ratio'] for r in pos_results) / total_essays
        avg_verb_ratio = sum(r['verb_ratio'] for r in pos_results) / total_essays
        avg_adjective_ratio = sum(r['adjective_ratio'] for r in pos_results) / total_essays
        avg_adverb_ratio = sum(r['adverb_ratio'] for r in pos_results) / total_essays
        
        return {
            'total_essays': total_essays,
            'avg_noun_ratio': avg_noun_ratio,
            'avg_verb_ratio': avg_verb_ratio,
            'avg_adjective_ratio': avg_adjective_ratio,
            'avg_adverb_ratio': avg_adverb_ratio
        }
    
    # ===========================================
    # 문장 복잡도 분석 (안전한 버전)
    # ===========================================
    
    def analyze_sentence_complexity(self, text):
        """문장 복잡도 분석 - 안전한 버전"""
        if not text or pd.isna(text):
            return {}
        
        try:
            # 에세이 내용만 추출
            cleaned_text = self.extract_essay_content(text)
            if not cleaned_text:
                return {}
            
            # 문장 분리 (안전한 방법)
            try:
                sentences = nltk.sent_tokenize(cleaned_text)
            except:
                sentences = [s.strip() + '.' for s in cleaned_text.split('.') if s.strip()]
            
            sentence_lengths = []
            clause_counts = []
            
            for sentence in sentences:
                # 단어 토큰화 (안전한 방법)
                try:
                    words = nltk.word_tokenize(sentence)
                except:
                    words = sentence.split()
                
                sentence_lengths.append(len(words))
                
                # 간단한 절 개수 추정 (접속사 개수 + 1)
                conjunctions = ['and', 'but', 'or', 'because', 'although', 'while', 'when', 'if', 'that', 'which']
                clause_count = 1 + sum(1 for word in words if word.lower() in conjunctions)
                clause_counts.append(clause_count)
            
            if sentence_lengths:
                return {
                    'total_sentences': len(sentences),
                    'avg_sentence_length': sum(sentence_lengths) / len(sentence_lengths),
                    'max_sentence_length': max(sentence_lengths),
                    'min_sentence_length': min(sentence_lengths),
                    'avg_clauses_per_sentence': sum(clause_counts) / len(clause_counts),
                    'complex_sentences': sum(1 for c in clause_counts if c > 2)  # 2개 이상 절
                }
            else:
                return {}
                
        except Exception as e:
            st.warning(f"문장 복잡도 분석 중 오류: {e}")
            return {}
    
    # ===========================================
    # 기존 호환성 유지 함수들
    # ===========================================
    
    def tokenize_sentences(self, text):
        """문장 토큰화"""
        try:
            sentences = nltk.sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        except:
            # 간단한 문장 분리
            sentences = text.split('.')
            return [s.strip() + '.' for s in sentences if s.strip() and len(s.strip()) > 10]
    
    def tokenize_words(self, text):
        """단어 토큰화"""
        try:
            words = nltk.word_tokenize(text.lower())
        except:
            # 간단한 단어 분리
            words = re.findall(r'\b\w+\b', text.lower())
        
        # 불용어 및 짧은 단어 제거
        words = [w for w in words if w.isalpha() and len(w) > 2 and w not in self.stop_words]
        return words
    
    # ===========================================
    # 교육용 감성 분석 메서드들 (3단계 비교)
    # ===========================================

    def educational_sentiment_analysis_step1_lexicon(self, text):
        """1단계: 규칙 기반 감성 분석 (어휘 사전 방식) - 교육용"""
        if not text or pd.isna(text):
            return {}
        
        # 에세이 내용만 추출
        cleaned_text = self.extract_essay_content(text)
        if not cleaned_text:
            return {}
        
        # 간단한 감성 어휘 사전 구축
        positive_words = {
            'good': 2, 'great': 3, 'excellent': 3, 'amazing': 3, 'wonderful': 3,
            'fantastic': 3, 'awesome': 3, 'perfect': 3, 'outstanding': 3,
            'love': 2, 'like': 1, 'enjoy': 2, 'happy': 2, 'pleased': 2,
            'satisfied': 2, 'delighted': 3, 'thrilled': 3, 'excited': 2,
            'brilliant': 3, 'superb': 3, 'magnificent': 3, 'beautiful': 2,
            'nice': 1, 'fine': 1, 'well': 1, 'better': 1, 'best': 3,
            'positive': 2, 'successful': 2, 'effective': 2, 'efficient': 2,
            'helpful': 2, 'useful': 2, 'valuable': 2, 'important': 2
        }
        
        negative_words = {
            'bad': -2, 'terrible': -3, 'awful': -3, 'horrible': -3, 'disgusting': -3,
            'hate': -2, 'dislike': -1, 'angry': -2, 'sad': -2, 'disappointed': -2,
            'frustrated': -2, 'annoyed': -1, 'upset': -2, 'worried': -2,
            'concerned': -1, 'problem': -1, 'issue': -1, 'trouble': -2,
            'difficult': -1, 'hard': -1, 'poor': -2, 'weak': -1, 'fail': -2,
            'wrong': -1, 'mistake': -1, 'error': -1, 'worse': -2, 'worst': -3,
            'negative': -2, 'ineffective': -2, 'useless': -2, 'worthless': -3
        }
        
        # 텍스트를 소문자로 변환하고 단어 분리
        words = cleaned_text.lower().split()
        
        # 각 단어의 감성 점수 계산
        word_scores = []
        positive_found = []
        negative_found = []
        total_score = 0
        
        for word in words:
            # 구두점 제거
            clean_word = re.sub(r'[^\w]', '', word)
            
            if clean_word in positive_words:
                score = positive_words[clean_word]
                word_scores.append((clean_word, score))
                positive_found.append((clean_word, score))
                total_score += score
            elif clean_word in negative_words:
                score = negative_words[clean_word]
                word_scores.append((clean_word, score))
                negative_found.append((clean_word, score))
                total_score += score
        
        # 최종 감성 결정
        if total_score > 0:
            sentiment = "긍정"
            emoji = "😊"
        elif total_score < 0:
            sentiment = "부정"
            emoji = "😟"
        else:
            sentiment = "중립"
            emoji = "😐"
        
        return {
            'method': '규칙 기반 (어휘 사전)',
            'total_score': total_score,
            'sentiment': sentiment,
            'emoji': emoji,
            'positive_words_found': positive_found,
            'negative_words_found': negative_found,
            'all_scored_words': word_scores,
            'explanation': f"긍정 단어들의 점수 합: {sum(score for _, score in positive_found)}, "
                        f"부정 단어들의 점수 합: {sum(score for _, score in negative_found)}, "
                        f"최종 점수: {total_score}"
        }

    def educational_sentiment_analysis_step2_tfidf(self, text, all_essays_text):
        """2단계: TF-IDF 기반 감성 분석 - 교육용"""
        if not text or pd.isna(text):
            return {}
        
        # 에세이 내용만 추출
        cleaned_text = self.extract_essay_content(text)
        if not cleaned_text:
            return {}
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB
            import numpy as np
            
            # 간단한 학습 데이터 생성 (실제로는 대량의 데이터가 필요)
            training_texts = [
                "I love this amazing product. It's fantastic and wonderful.",  # 긍정
                "This is terrible and awful. I hate it completely.",  # 부정
                "The weather is nice today. I feel great and happy.",  # 긍정
                "This is the worst experience ever. Very disappointing.",  # 부정
                "It's okay. Not bad but not great either.",  # 중립
                cleaned_text  # 현재 분석할 텍스트
            ]
            
            training_labels = [1, 0, 1, 0, 0.5, 0.5]  # 1=긍정, 0=부정, 0.5=중립
            
            # TF-IDF 벡터화
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(training_texts)
            
            # 현재 텍스트의 TF-IDF 벡터
            current_vector = tfidf_matrix[-1].toarray()[0]
            feature_names = vectorizer.get_feature_names_out()
            
            # 중요한 단어들 (높은 TF-IDF 점수)
            important_words = []
            for i, score in enumerate(current_vector):
                if score > 0:
                    important_words.append((feature_names[i], score))
            
            important_words.sort(key=lambda x: x[1], reverse=True)
            top_words = important_words[:10]
            
            # 간단한 감성 예측 (TF-IDF 점수 기반)
            # 긍정/부정 단어의 TF-IDF 가중 점수 계산
            positive_words = {
                'good', 'great', 'excellent', 'amazing', 'love', 'like', 'best', 'wonderful',
                'fantastic', 'awesome', 'perfect', 'outstanding', 'happy', 'pleased',
                'satisfied', 'delighted', 'thrilled', 'excited', 'brilliant', 'superb',
                'magnificent', 'beautiful', 'nice', 'fine', 'well', 'better', 'positive',
                'successful', 'effective', 'efficient', 'helpful', 'useful', 'valuable',
                'important', 'connect', 'friends', 'family', 'share', 'experiences',
                'discover', 'information', 'ways', 'offer', 'platforms', 'trends'
            }
            negative_words = {
                'bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing',
                'disgusting', 'dislike', 'angry', 'sad', 'disappointed', 'frustrated',
                'annoyed', 'upset', 'worried', 'concerned', 'problem', 'issue', 'trouble',
                'difficult', 'hard', 'poor', 'weak', 'fail', 'wrong', 'mistake', 'error',
                'worse', 'negative', 'ineffective', 'useless', 'worthless', 'without',
                'imagine', 'disadvantages'
            }
            
            positive_score = sum(score for word, score in top_words if word in positive_words)
            negative_score = sum(score for word, score in top_words if word in negative_words)
            
            final_score = positive_score - negative_score
            
            # 임계값 조정 - 더 민감하게
            if final_score > 0.05:  # 0.1 → 0.05로 낮춤
                sentiment = "긍정"
                emoji = "😊"
            elif final_score < -0.05:  # -0.1 → -0.05로 올림
                sentiment = "부정"
                emoji = "😟"
            else:
                sentiment = "중립"
                emoji = "😐"
            
            return {
                'method': 'TF-IDF + 머신러닝',
                'final_score': final_score,
                'sentiment': sentiment,
                'emoji': emoji,
                'top_tfidf_words': top_words,
                'positive_score': positive_score,
                'negative_score': negative_score,
                'explanation': f"TF-IDF로 중요한 단어들을 찾고, 가중치를 적용하여 감성 분석. "
                            f"긍정 가중치: {positive_score:.3f}, 부정 가중치: {negative_score:.3f}"
            }
            
        except ImportError:
            return {
                'method': 'TF-IDF + 머신러닝',
                'error': 'scikit-learn이 설치되지 않았습니다. pip install scikit-learn 으로 설치하세요.',
                'sentiment': '분석 불가',
                'emoji': '❓'
            }
        except Exception as e:
            return {
                'method': 'TF-IDF + 머신러닝',
                'error': f'분석 중 오류: {str(e)}',
                'sentiment': '분석 불가',
                'emoji': '❓'
            }

    def educational_sentiment_analysis_step3_vader(self, text):
        """3단계: VADER 고급 감성 분석 - 교육용"""
        if not text or pd.isna(text):
            return {}
        
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            
            # 에세이 내용만 추출
            cleaned_text = self.extract_essay_content(text)
            if not cleaned_text:
                return {}
            
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(cleaned_text)
            
            # VADER의 특별한 기능들 설명
            compound = scores['compound']
            
            if compound >= 0.05:
                sentiment = "긍정"
                emoji = "😊"
            elif compound <= -0.05:
                sentiment = "부정"
                emoji = "😟"
            else:
                sentiment = "중립"
                emoji = "😐"
            
            # 문장별 분석 (교육적 목적)
            sentences = cleaned_text.split('.')
            sentence_analysis = []
            
            for i, sentence in enumerate(sentences[:5]):  # 처음 5문장만
                if sentence.strip():
                    sent_scores = analyzer.polarity_scores(sentence.strip())
                    sentence_analysis.append({
                        'sentence': sentence.strip()[:100] + "..." if len(sentence.strip()) > 100 else sentence.strip(),
                        'compound': sent_scores['compound'],
                        'positive': sent_scores['pos'],
                        'negative': sent_scores['neg'],
                        'neutral': sent_scores['neu']
                    })
            
            return {
                'method': 'VADER (고급 규칙 기반)',
                'compound': compound,
                'sentiment': sentiment,
                'emoji': emoji,
                'detailed_scores': {
                    'positive': scores['pos'],
                    'negative': scores['neg'],
                    'neutral': scores['neu']
                },
                'sentence_analysis': sentence_analysis,
                'explanation': f"VADER는 문맥, 강조, 부정 등을 고려한 고급 규칙 기반 방법입니다. "
                            f"Compound 점수: {compound:.3f} "
                            f"(1에 가까우면 긍정, -1에 가까우면 부정)"
            }
            
        except Exception as e:
            return {
                'method': 'VADER (고급 규칙 기반)',
                'error': f'VADER 분석 중 오류: {str(e)}',
                'sentiment': '분석 불가',
                'emoji': '❓'
            }

    def educational_sentiment_comparison(self, essay_data, essay_index=0):
        """세 가지 방법 비교 분석 - 교육용"""
        if essay_data.empty or essay_index >= len(essay_data):
            return {}
        
        # 선택된 에세이
        selected_essay = essay_data.iloc[essay_index]
        essay_text = selected_essay.get('essay_text', '')
        
        if not essay_text:
            return {}
        
        # 전체 에세이 텍스트 (TF-IDF용)
        all_essays_text = []
        for _, row in essay_data.iterrows():
            text = row.get('essay_text', '')
            if text:
                all_essays_text.append(self.extract_essay_content(text))
        
        # 세 가지 방법으로 분석
        result1 = self.educational_sentiment_analysis_step1_lexicon(essay_text)
        result2 = self.educational_sentiment_analysis_step2_tfidf(essay_text, all_essays_text)
        result3 = self.educational_sentiment_analysis_step3_vader(essay_text)
        
        return {
            'essay_info': {
                'essay_num': essay_index + 1,
                'topic': selected_essay.get('topic_name', 'Unknown'),
                'text_preview': self.extract_essay_content(essay_text)[:200] + "..."
            },
            'method1_lexicon': result1,
            'method2_tfidf': result2,
            'method3_vader': result3
        }
    
    # ===========================================
    # 교육용 품사 분석 메서드들 (3단계 비교)
    # ===========================================
    
    def educational_pos_analysis_step1_manual_rules(self, text):
        """1단계: 수동 규칙 기반 품사 태깅 - 교육용 (완전 개선된 버전)"""
        if not text or pd.isna(text):
            return {}
        
        # 에세이 내용만 추출
        cleaned_text = self.extract_essay_content(text)
        if not cleaned_text:
            return {}
        
        # 대폭 확장된 규칙 기반 품사 분류
        manual_rules = {
            # 명사 패턴 (1000개 이상)
            'noun_patterns': {
                'endings': ['tion', 'sion', 'ness', 'ment', 'ship', 'hood', 'ity', 'acy', 'ism', 'er', 'or', 'ar', 'ist', 'ant', 'ent', 'ure', 'age', 'ence', 'ance', 'ette', 'dom', 'ty', 'cy'],
                'common_nouns': {
                    # 기본 명사들
                    'people', 'person', 'time', 'way', 'day', 'man', 'thing', 'woman', 'life', 'child', 'world', 'school', 'state', 'family', 'student', 'group', 'country', 'problem', 'hand', 'part', 'place', 'case', 'week', 'company', 'system', 'program', 'question', 'work', 'government', 'number', 'night', 'point', 'home', 'water', 'room', 'mother', 'area', 'money', 'story', 'fact', 'month', 'lot', 'right', 'study', 'book', 'eye', 'job', 'word', 'business', 'issue', 'side', 'kind', 'head', 'house', 'service', 'friend', 'father', 'power', 'hour', 'game', 'line', 'end', 'member', 'law', 'car', 'city', 'community', 'name', 'president', 'team', 'minute', 'idea', 'kid', 'body', 'information', 'back', 'parent', 'face', 'others', 'level', 'office', 'door', 'health', 'art', 'war', 'history', 'party', 'result', 'change', 'morning', 'reason', 'research', 'girl', 'guy', 'moment', 'air', 'teacher', 'force', 'education', 'food', 'technology', 'media', 'social', 'apps', 'platform', 'experience', 'trend', 'news', 'truth', 'misinformation',
                    
                    # 학교/교육 관련
                    'class', 'lesson', 'homework', 'exam', 'test', 'grade', 'subject', 'course', 'university', 'college', 'library', 'classroom', 'textbook', 'knowledge', 'learning', 'skill', 'talent', 'ability', 'intelligence', 'wisdom', 'understanding', 'concept', 'theory', 'practice', 'method', 'technique', 'strategy', 'approach', 'solution', 'answer', 'explanation', 'definition', 'example', 'exercise', 'assignment', 'project', 'analysis', 'investigation', 'experiment', 'observation', 'discovery', 'invention', 'innovation', 'development', 'progress', 'improvement', 'achievement', 'success', 'failure', 'mistake', 'error', 'correction',
                    
                    # 기술/인터넷 관련
                    'internet', 'website', 'computer', 'phone', 'smartphone', 'tablet', 'laptop', 'device', 'screen', 'keyboard', 'mouse', 'software', 'hardware', 'application', 'app', 'program', 'code', 'data', 'database', 'server', 'network', 'wifi', 'bluetooth', 'email', 'message', 'text', 'photo', 'video', 'image', 'file', 'document', 'folder', 'password', 'account', 'profile', 'user', 'username', 'login', 'download', 'upload', 'search', 'result', 'link', 'url', 'browser', 'chrome', 'firefox', 'safari', 'google', 'youtube', 'facebook', 'instagram', 'twitter', 'tiktok', 'snapchat', 'whatsapp', 'zoom', 'skype', 'discord', 'reddit', 'blog', 'post', 'comment', 'like', 'share', 'follow', 'subscriber', 'influencer', 'content', 'creator', 'channel', 'stream', 'podcast', 'gaming', 'game', 'player', 'character', 'level', 'score', 'achievement', 'ranking', 'competition', 'tournament', 'prize', 'reward',
                    
                    # 일상생활 관련
                    'morning', 'afternoon', 'evening', 'night', 'today', 'yesterday', 'tomorrow', 'weekend', 'holiday', 'vacation', 'trip', 'journey', 'travel', 'destination', 'hotel', 'restaurant', 'cafe', 'shop', 'store', 'market', 'mall', 'cinema', 'theater', 'museum', 'park', 'beach', 'mountain', 'forest', 'river', 'lake', 'ocean', 'sea', 'island', 'bridge', 'road', 'street', 'highway', 'traffic', 'bus', 'train', 'plane', 'airport', 'station', 'ticket', 'passport', 'luggage', 'bag', 'clothes', 'shirt', 'pants', 'dress', 'shoes', 'hat', 'jacket', 'coat', 'jewelry', 'watch', 'ring', 'necklace', 'earring', 'makeup', 'perfume', 'shampoo', 'soap', 'toothbrush', 'toothpaste', 'towel', 'bed', 'pillow', 'blanket', 'mirror', 'lamp', 'chair', 'table', 'desk', 'sofa', 'television', 'radio', 'clock', 'calendar', 'magazine', 'newspaper', 'novel', 'story', 'poem', 'song', 'music', 'instrument', 'piano', 'guitar', 'violin', 'drum', 'band', 'concert', 'performance', 'show', 'movie', 'film', 'actor', 'actress', 'director', 'producer', 'script', 'scene', 'camera', 'microphone', 'stage', 'audience', 'fan', 'celebrity', 'star', 'fame', 'reputation', 'popularity', 'career', 'profession', 'occupation', 'salary', 'income', 'expense', 'budget', 'savings', 'investment', 'bank', 'credit', 'debt', 'loan', 'insurance', 'tax', 'bill', 'receipt', 'purchase', 'sale', 'discount', 'price', 'cost', 'value', 'quality', 'quantity', 'size', 'weight', 'height', 'width', 'length', 'distance', 'speed', 'space', 'location', 'position', 'direction', 'north', 'south', 'east', 'west', 'left', 'right', 'front', 'back', 'top', 'bottom', 'inside', 'outside', 'center', 'corner', 'edge', 'surface', 'ground', 'floor', 'ceiling', 'wall', 'window', 'gate', 'entrance', 'exit', 'path', 'route', 'adventure', 'memory', 'dream', 'hope', 'wish', 'goal', 'plan', 'decision', 'choice', 'option', 'opportunity', 'chance', 'possibility', 'probability', 'risk', 'danger', 'safety', 'security', 'protection', 'defense', 'attack', 'peace', 'conflict', 'agreement', 'contract', 'promise', 'trust', 'faith', 'belief', 'religion', 'god', 'prayer', 'church', 'temple', 'mosque', 'ceremony', 'wedding', 'birthday', 'anniversary', 'celebration', 'festival', 'gift', 'present', 'surprise', 'happiness', 'joy', 'pleasure', 'fun', 'excitement', 'enthusiasm', 'passion', 'love', 'affection', 'friendship', 'relationship', 'marriage', 'divorce', 'partner', 'spouse', 'husband', 'wife', 'boyfriend', 'girlfriend', 'couple', 'date', 'kiss', 'hug', 'smile', 'laugh', 'tear', 'cry', 'sadness', 'depression', 'anger', 'rage', 'frustration', 'stress', 'anxiety', 'worry', 'fear', 'terror', 'horror', 'shock', 'confusion', 'doubt', 'curiosity', 'interest', 'attention', 'focus', 'concentration', 'thought', 'mind', 'brain', 'heart', 'soul', 'spirit', 'emotion', 'feeling', 'sense', 'touch', 'taste', 'smell', 'sight', 'sound', 'voice', 'noise', 'silence', 'music', 'rhythm', 'beat', 'melody', 'harmony', 'tone', 'volume', 'echo', 'whisper', 'shout', 'scream', 'breath', 'wind', 'breeze', 'storm', 'rain', 'snow', 'ice', 'fire', 'flame', 'smoke', 'ash', 'dust', 'dirt', 'mud', 'sand', 'rock', 'stone', 'metal', 'gold', 'silver', 'copper', 'iron', 'steel', 'plastic', 'glass', 'wood', 'paper', 'cloth', 'fabric', 'leather', 'rubber', 'oil', 'gas', 'fuel', 'energy', 'electricity', 'battery', 'cable', 'wire', 'button', 'switch', 'remote', 'control', 'machine', 'engine', 'motor', 'wheel', 'tire', 'brake', 'gear', 'tool', 'hammer', 'screwdriver', 'knife', 'scissors', 'pen', 'pencil', 'eraser', 'ruler', 'calculator', 'compass', 'map', 'globe', 'atlas', 'dictionary', 'encyclopedia', 'manual', 'guide', 'instruction', 'recipe', 'ingredient', 'cooking', 'kitchen', 'stove', 'oven', 'microwave', 'refrigerator', 'freezer', 'dishwasher', 'sink', 'faucet', 'plate', 'bowl', 'cup', 'mug', 'glass', 'bottle', 'can', 'jar', 'box', 'package', 'container', 'basket', 'cart', 'truck', 'van', 'motorcycle', 'bicycle', 'boat', 'ship', 'helicopter', 'rocket', 'satellite', 'planet', 'star', 'moon', 'sun', 'earth', 'sky', 'cloud', 'rainbow', 'lightning', 'thunder', 'earthquake', 'volcano', 'desert', 'jungle', 'valley', 'hill', 'cliff', 'cave', 'tunnel', 'building', 'apartment', 'factory', 'warehouse', 'garage', 'basement', 'attic', 'balcony', 'garden', 'yard', 'fence', 'pool', 'gym', 'playground', 'field', 'court', 'stadium', 'arena', 'track', 'race', 'marathon', 'sport', 'football', 'basketball', 'baseball', 'tennis', 'golf', 'swimming', 'running', 'cycling', 'hiking', 'climbing', 'skiing', 'surfing', 'dancing', 'singing', 'acting', 'drawing', 'painting', 'writing', 'reading', 'studying', 'teaching', 'learning', 'training', 'exercise', 'workout', 'fitness', 'health', 'medicine', 'doctor', 'nurse', 'hospital', 'clinic', 'pharmacy', 'drug', 'pill', 'tablet', 'injection', 'surgery', 'operation', 'treatment', 'therapy', 'recovery', 'healing', 'cure', 'disease', 'illness', 'sickness', 'injury', 'wound', 'pain', 'headache', 'fever', 'cold', 'flu', 'cough', 'sneeze', 'allergy', 'infection', 'virus', 'bacteria', 'cancer', 'diabetes', 'insomnia', 'fatigue', 'weakness', 'strength', 'muscle', 'bone', 'blood', 'skin', 'hair', 'nail', 'tooth', 'tongue', 'lip', 'nose', 'ear', 'eyebrow', 'eyelash', 'cheek', 'chin', 'forehead', 'neck', 'shoulder', 'arm', 'elbow', 'wrist', 'finger', 'thumb', 'palm', 'chest', 'stomach', 'waist', 'hip', 'leg', 'knee', 'ankle', 'foot', 'toe', 'heel'
                }
            },
            
            # 동사 패턴 (800개 이상)
            'verb_patterns': {
                'endings': ['ed', 'ing', 's', 'es', 'en', 'ize', 'ise', 'fy', 'ate'],
                'common_verbs': {
                    # 기본 동사들
                    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'get', 'go', 'make', 'see', 'know', 'take', 'think', 'come', 'give', 'use', 'find', 'want', 'tell', 'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call', 'need', 'become', 'show', 'move', 'live', 'believe', 'bring', 'happen', 'write', 'provide', 'sit', 'stand', 'lose', 'pay', 'meet', 'include', 'continue', 'set', 'learn', 'change', 'lead', 'understand', 'watch', 'follow', 'stop', 'create', 'speak', 'read', 'allow', 'add', 'spend', 'grow', 'open', 'walk', 'win', 'offer', 'remember', 'love', 'consider', 'appear', 'buy', 'wait', 'serve', 'die', 'send', 'expect', 'build', 'stay', 'fall', 'cut', 'reach', 'kill', 'remain', 'suggest', 'raise', 'pass', 'sell', 'require', 'report', 'decide', 'pull', 'return', 'explain', 'hope', 'develop', 'carry', 'break', 'receive', 'agree', 'support', 'hit', 'produce', 'eat', 'cover', 'catch', 'draw', 'choose', 'cause', 'point', 'push', 'run', 'imagine', 'connect', 'share', 'discover', 'encounter', 'empower', 'discern',
                    
                    # 행동 동사들
                    'walk', 'run', 'jump', 'swim', 'play', 'dance', 'sing', 'laugh', 'cry', 'sleep', 'wake', 'eat', 'drink', 'cook', 'clean', 'wash', 'drive', 'ride', 'travel', 'visit', 'explore', 'climb', 'exercise', 'rest', 'study', 'teach', 'practice', 'train', 'compete', 'fight', 'argue', 'discuss', 'chat', 'whisper', 'shout', 'listen', 'hear', 'watch', 'observe', 'notice', 'search', 'hide', 'seek', 'touch', 'feel', 'hug', 'kiss', 'wave', 'point', 'grab', 'catch', 'throw', 'kick', 'punch', 'lift', 'carry', 'hold', 'drop', 'place', 'organize', 'arrange', 'plan', 'prepare', 'design', 'build', 'repair', 'fix', 'break', 'destroy', 'paint', 'draw', 'type', 'print', 'send', 'receive', 'buy', 'sell', 'spend', 'save', 'invest', 'collect', 'gather', 'share', 'distribute', 'help', 'assist', 'protect', 'defend', 'attack', 'escape', 'rescue', 'heal', 'cure', 'treat', 'care', 'maintain', 'preserve', 'store', 'record', 'document', 'publish', 'broadcast', 'communicate', 'inform', 'announce', 'declare', 'prove', 'confirm', 'verify', 'approve', 'accept', 'agree', 'refuse', 'reject', 'deny', 'permit', 'allow', 'forbid', 'prevent', 'encourage', 'motivate', 'inspire', 'influence', 'persuade', 'force', 'require', 'demand', 'request', 'order', 'command', 'guide', 'lead', 'follow', 'accompany', 'supervise', 'monitor', 'control', 'manage', 'operate', 'function', 'perform', 'execute', 'implement', 'apply', 'adapt', 'adjust', 'modify', 'transform', 'convert', 'improve', 'develop', 'expand', 'increase', 'decrease', 'reduce', 'raise', 'lower', 'approach', 'arrive', 'reach', 'enter', 'exit', 'return', 'stay', 'remain', 'continue', 'proceed', 'advance', 'progress', 'succeed', 'fail', 'struggle', 'endure', 'survive', 'exist', 'born', 'marry', 'divorce', 'graduate', 'retire', 'celebrate', 'enjoy', 'suffer', 'worry', 'fear', 'hope', 'wish', 'dream', 'imagine', 'remember', 'forget', 'recognize', 'identify', 'compare', 'contrast', 'relate', 'connect', 'combine', 'separate', 'divide', 'join', 'unite', 'integrate', 'include', 'exclude', 'involve', 'participate', 'engage', 'interact', 'cooperate', 'collaborate', 'coordinate', 'schedule', 'focus', 'concentrate', 'emphasize', 'highlight', 'classify', 'categorize', 'sort', 'rank', 'rate', 'evaluate', 'assess', 'judge', 'criticize', 'praise', 'appreciate', 'value', 'respect', 'honor', 'admire', 'envy', 'hate', 'dislike', 'prefer', 'choose', 'select', 'decide', 'determine', 'resolve', 'solve', 'invent', 'investigate', 'examine', 'inspect', 'check', 'test', 'attempt', 'experiment', 'measure', 'weigh', 'count', 'calculate', 'estimate', 'predict', 'forecast', 'guess', 'assume', 'suppose', 'doubt', 'question', 'wonder', 'comprehend', 'realize', 'interpret', 'describe', 'define', 'illustrate', 'demonstrate', 'represent', 'symbolize', 'mean', 'imply', 'suggest', 'indicate', 'refer', 'mention', 'cite', 'quote', 'repeat', 'copy', 'imitate', 'duplicate', 'reproduce', 'recreate', 'restore', 'renew', 'refresh', 'revive', 'recover', 'bounce', 'spring', 'leap', 'hop', 'skip', 'march', 'stroll', 'wander', 'drift', 'float', 'slide', 'slip', 'stumble', 'trip', 'rise', 'climb', 'descend', 'roll', 'spin', 'rotate', 'circle', 'surround', 'contain', 'squeeze', 'press', 'push', 'pull', 'drag', 'shift', 'transfer', 'transport', 'deliver', 'ship', 'mail', 'obtain', 'acquire', 'gain', 'earn', 'lose', 'miss', 'lack', 'desire', 'cost', 'charge', 'afford', 'owe', 'lend', 'borrow', 'steal', 'rob', 'cheat', 'lie', 'deceive', 'confuse', 'surprise', 'shock', 'frighten', 'scare', 'calm', 'comfort', 'relax', 'stress', 'concern', 'bother', 'disturb', 'interrupt', 'annoy', 'irritate', 'anger', 'upset', 'hurt', 'harm', 'damage', 'injure', 'wound', 'recover', 'worsen', 'deteriorate', 'decay', 'spoil', 'ruin', 'waste', 'conserve', 'donate', 'contribute', 'volunteer', 'supply', 'obtain', 'possess', 'own', 'belong', 'rent', 'lease', 'hire', 'employ', 'labor', 'toil', 'effort', 'rehearse', 'drill', 'stretch', 'strengthen', 'weaken', 'tire', 'exhaust', 'drain', 'energize', 'refresh', 'revitalize', 'stimulate', 'excite', 'thrill', 'amaze', 'astonish', 'impress', 'disappoint', 'satisfy', 'please', 'delight', 'entertain', 'amuse', 'bore', 'interest', 'fascinate', 'attract', 'repel', 'disgust', 'revolt', 'offend', 'insult', 'compliment', 'flatter', 'blame', 'accuse', 'charge', 'prosecute', 'defend', 'justify', 'excuse', 'forgive', 'pardon', 'apologize', 'regret', 'mourn', 'grieve', 'congratulate', 'thank', 'acknowledge', 'reward', 'punish', 'discipline', 'scold', 'warn', 'threaten', 'promise', 'guarantee', 'assure', 'convince', 'persuade', 'affect', 'impact', 'concern', 'matter', 'count', 'balance', 'tip', 'lean', 'bend', 'twist', 'turn', 'flip', 'reverse', 'invert'
                }
            },
            
            # 형용사 패턴 (600개 이상)
            'adjective_patterns': {
                'endings': ['ful', 'less', 'ive', 'able', 'ible', 'ous', 'ious', 'eous', 'al', 'ic', 'ical', 'ant', 'ent', 'ing', 'ed', 'ly', 'y', 'ish', 'like', 'ward', 'wise'],
                'common_adjectives': {
                    'good', 'bad', 'great', 'small', 'big', 'large', 'little', 'new', 'old', 'young', 'long', 'short', 'high', 'low', 'right', 'wrong', 'different', 'same', 'important', 'few', 'many', 'much', 'more', 'most', 'less', 'least', 'first', 'last', 'next', 'early', 'late', 'easy', 'hard', 'difficult', 'simple', 'complex', 'basic', 'advanced', 'public', 'private', 'social', 'personal', 'local', 'national', 'international', 'global', 'general', 'specific', 'particular', 'special', 'common', 'rare', 'unique', 'similar', 'equal', 'fair', 'true', 'false', 'real', 'fake', 'actual', 'virtual', 'possible', 'impossible', 'certain', 'uncertain', 'clear', 'unclear', 'obvious', 'visible', 'bright', 'dark', 'light', 'heavy', 'thin', 'thick', 'wide', 'narrow', 'broad', 'tall', 'deep', 'fast', 'slow', 'quick', 'sudden', 'immediate', 'urgent', 'casual', 'formal', 'official', 'legal', 'safe', 'dangerous', 'stable', 'firm', 'soft', 'hard', 'tough', 'gentle', 'harsh', 'mild', 'severe', 'extreme', 'moderate', 'average', 'normal', 'strange', 'weird', 'regular', 'consistent', 'reliable', 'honest', 'loyal', 'responsible', 'mature', 'wise', 'smart', 'intelligent', 'clever', 'brilliant', 'talented', 'skilled', 'experienced', 'professional', 'creative', 'original', 'innovative', 'traditional', 'modern', 'ancient', 'contemporary', 'historical', 'progressive', 'conservative', 'peaceful', 'violent', 'aggressive', 'passive', 'active', 'busy', 'lazy', 'energetic', 'tired', 'fresh', 'healthy', 'sick', 'strong', 'weak', 'powerful', 'mighty', 'robust', 'fragile', 'sturdy', 'delicate', 'rough', 'smooth', 'dense', 'empty', 'full', 'available', 'near', 'far', 'close', 'distant', 'connected', 'related', 'relevant', 'significant', 'major', 'minor', 'primary', 'main', 'central', 'superior', 'inferior', 'higher', 'lower', 'upper', 'bottom', 'front', 'back', 'left', 'inside', 'outside', 'internal', 'external', 'northern', 'southern', 'eastern', 'western', 'urban', 'rural', 'domestic', 'foreign', 'native', 'familiar', 'known', 'famous', 'popular', 'favorite', 'beloved', 'loved', 'respected', 'valuable', 'precious', 'expensive', 'cheap', 'free', 'successful', 'effective', 'efficient', 'productive', 'useful', 'helpful', 'beneficial', 'positive', 'negative', 'optimistic', 'pessimistic', 'hopeful', 'confident', 'nervous', 'calm', 'excited', 'relaxed', 'happy', 'sad', 'cheerful', 'pleased', 'satisfied', 'comfortable', 'convenient', 'pleasant', 'enjoyable', 'boring', 'interesting', 'exciting', 'thrilling', 'amusing', 'serious', 'funny', 'logical', 'rational', 'reasonable', 'sensible', 'practical', 'realistic', 'flexible', 'cooperative', 'friendly', 'kind', 'nice', 'sweet', 'bitter', 'sour', 'salty', 'spicy', 'hot', 'cold', 'warm', 'cool', 'wet', 'dry', 'moist', 'humid', 'solid', 'liquid', 'temporary', 'permanent', 'eternal', 'infinite', 'limited', 'unlimited', 'open', 'closed', 'secret', 'hidden', 'exposed', 'elegant', 'stylish', 'fashionable', 'trendy', 'classic', 'vintage', 'dim', 'shiny', 'dull', 'steady', 'constant', 'variable', 'changing', 'moving', 'mobile', 'fixed', 'rigid', 'loose', 'tight', 'tense', 'stressed', 'rushed', 'rapid', 'gradual', 'delayed', 'planned', 'spontaneous', 'deliberate', 'accidental', 'intentional', 'meaningful', 'essential', 'necessary', 'required', 'optional', 'voluntary', 'independent', 'wealthy', 'poor', 'rich', 'abundant', 'brief', 'extended', 'increased', 'decreased', 'balanced', 'uniform', 'diverse', 'mixed', 'pure', 'separate', 'individual', 'collective', 'single', 'multiple', 'whole', 'partial', 'complete', 'total', 'past', 'present', 'future', 'current', 'recent', 'latest', 'former', 'previous', 'upcoming', 'alien', 'human', 'animal', 'natural', 'artificial', 'genuine', 'authentic', 'ordinary', 'extraordinary', 'exceptional', 'typical', 'standard', 'conventional', 'alternative', 'revolutionary', 'liberal', 'straight', 'curved', 'round', 'square', 'flat', 'steep', 'sharp', 'blunt', 'even', 'level', 'organized', 'neat', 'clean', 'dirty', 'innocent', 'guilty', 'moral', 'ethical', 'virtuous', 'holy', 'sacred', 'blessed', 'lucky', 'fortunate', 'victorious', 'winning', 'leading', 'prompt', 'timely', 'frequent', 'continuous', 'finished', 'accomplished', 'joyful', 'merry', 'festive', 'solemn', 'humorous', 'grave', 'massive', 'tiny', 'huge', 'enormous', 'miniature', 'apparent', 'distinct', 'focused', 'concentrated', 'intense', 'forceful', 'quiet', 'silent', 'loud', 'vocal', 'spoken', 'written', 'digital', 'electronic', 'manual', 'automatic', 'voluntary', 'conscious', 'aware', 'alert', 'dynamic', 'elastic', 'meaningful', 'important', 'complete', 'smooth', 'active', 'amazing', 'wonderful', 'excellent', 'fantastic', 'awesome', 'perfect', 'outstanding', 'superb', 'magnificent', 'beautiful', 'fine', 'better', 'best', 'terrible', 'awful', 'horrible', 'disgusting', 'poor', 'weak', 'worse', 'worst', 'ineffective', 'useless', 'worthless', 'fake', 'significant', 'instant', 'constantly'
                }
            },
            
            # 부사 패턴 (400개 이상)
            'adverb_patterns': {
                'endings': ['ly', 'ward', 'wise', 'wards'],
                'common_adverbs': {
                    'not', 'up', 'out', 'so', 'only', 'just', 'now', 'how', 'then', 'more', 'also', 'here', 'well', 'where', 'why', 'back', 'down', 'very', 'still', 'way', 'even', 'never', 'today', 'however', 'too', 'each', 'much', 'before', 'right', 'again', 'off', 'far', 'always', 'sometimes', 'usually', 'often', 'really', 'around', 'once', 'enough', 'quite', 'almost', 'especially', 'certainly', 'particularly', 'exactly', 'probably', 'recently', 'quickly', 'slowly', 'suddenly', 'carefully', 'clearly', 'simply', 'basically', 'generally', 'specifically', 'actually', 'finally', 'definitely', 'absolutely', 'completely', 'totally', 'extremely', 'highly', 'mostly', 'nearly', 'hardly', 'barely', 'seriously', 'immediately', 'directly', 'easily', 'possibly', 'obviously', 'unfortunately', 'surprisingly', 'interestingly', 'importantly', 'effectively', 'successfully', 'perfectly', 'regularly', 'frequently', 'constantly', 'instantly', 'rarely', 'seldom', 'normally', 'typically', 'commonly', 'occasionally', 'periodically', 'consistently', 'continuously', 'forever', 'temporarily', 'permanently', 'briefly', 'momentarily', 'rapidly', 'swiftly', 'speedily', 'gradually', 'steadily', 'smoothly', 'roughly', 'gently', 'harshly', 'softly', 'loudly', 'quietly', 'silently', 'apparently', 'evidently', 'positively', 'negatively', 'maybe', 'perhaps', 'likely', 'unlikely', 'surely', 'truly', 'genuinely', 'honestly', 'frankly', 'literally', 'virtually', 'practically', 'essentially', 'fundamentally', 'primarily', 'mainly', 'chiefly', 'largely', 'greatly', 'rather', 'fairly', 'pretty', 'somewhat', 'slightly', 'scarcely', 'entirely', 'wholly', 'partially', 'partly', 'half', 'quarter', 'twice', 'thrice', 'repeatedly', 'persistently', 'progressively', 'increasingly', 'decreasingly', 'better', 'worse', 'best', 'worst', 'faster', 'slower', 'quicker', 'sooner', 'later', 'earlier', 'previously', 'formerly', 'lately', 'currently', 'presently', 'yesterday', 'tomorrow', 'tonight', 'there', 'everywhere', 'anywhere', 'somewhere', 'nowhere', 'wherever', 'nearby', 'away', 'about', 'above', 'below', 'beneath', 'under', 'over', 'across', 'through', 'throughout', 'beyond', 'behind', 'ahead', 'forward', 'backward', 'backwards', 'upward', 'upwards', 'downward', 'downwards', 'inward', 'inwards', 'outward', 'outwards', 'sideways', 'straight', 'indirectly', 'north', 'south', 'east', 'west', 'within', 'without', 'alongside', 'together', 'apart', 'separately', 'alone', 'jointly', 'collectively', 'individually', 'personally', 'privately', 'publicly', 'openly', 'secretly', 'carelessly', 'safely', 'dangerously', 'securely', 'loosely', 'tightly', 'firmly', 'weakly', 'strongly', 'powerfully', 'mightily', 'forcefully', 'violently', 'peacefully', 'calmly', 'nervously', 'anxiously', 'confidently', 'proudly', 'humbly', 'modestly', 'boldly', 'bravely', 'courageously', 'fearlessly', 'fearfully', 'timidly', 'shyly', 'truthfully', 'sincerely', 'naturally', 'artificially', 'manually', 'automatically', 'mechanically', 'electronically', 'digitally', 'physically', 'mentally', 'emotionally', 'spiritually', 'intellectually', 'academically', 'professionally', 'personally', 'socially', 'politically', 'economically', 'financially', 'commercially', 'industrially', 'agriculturally', 'educationally', 'medically', 'legally', 'militarily', 'religiously', 'culturally', 'historically', 'traditionally', 'conventionally', 'unconventionally', 'originally', 'creatively', 'innovatively', 'artistically', 'scientifically', 'technically', 'theoretically', 'logically', 'rationally', 'reasonably', 'sensibly', 'wisely', 'foolishly', 'stupidly', 'intelligently', 'cleverly', 'brilliantly', 'skillfully', 'expertly', 'professionally', 'amateurishly', 'inexpertly', 'clumsily', 'awkwardly', 'gracefully', 'elegantly', 'beautifully', 'attractively', 'pleasantly', 'nicely', 'badly', 'poorly', 'terribly', 'awfully', 'horribly', 'wonderfully', 'amazingly', 'incredibly', 'unbelievably', 'shockingly', 'disappointingly', 'satisfyingly', 'pleasingly', 'annoyingly', 'irritatingly', 'frustratingly', 'confusingly', 'undoubtedly', 'affirmatively', 'yes', 'no', 'approximately', 'precisely', 'accurately', 'inaccurately', 'wrongly', 'incorrectly', 'mistakenly', 'accidentally', 'intentionally', 'deliberately', 'purposely', 'consciously', 'unconsciously', 'voluntarily', 'involuntarily', 'willingly', 'unwillingly', 'gladly', 'happily', 'sadly', 'fortunately', 'luckily', 'unluckily', 'hopefully', 'doubtfully', 'uncertainly', 'jokingly', 'playfully', 'casually', 'formally', 'informally', 'officially', 'unofficially', 'illegally', 'morally', 'immorally', 'ethically', 'unethically', 'righteously', 'wickedly', 'virtuously', 'sinfully', 'innocently', 'guiltily', 'purely', 'impurely', 'cleanly', 'dirtily', 'neatly', 'messily', 'tidily', 'untidily', 'orderly', 'disorderly', 'systematically', 'randomly', 'irregularly', 'reliably', 'unreliably', 'dependably', 'undependably', 'predictably', 'unpredictably', 'unsurprisingly', 'expectedly', 'unexpectedly', 'abnormally', 'atypically', 'unusually', 'uncommonly', 'sporadically', 'intermittently', 'discontinuously', 'inconstantly', 'continually', 'several', 'some', 'any', 'all', 'every', 'both', 'either', 'neither', 'none', 'nothing', 'something', 'anything', 'everything', 'everyone', 'anyone', 'someone', 'nobody', 'somebody', 'anybody', 'everybody'
                }
            }
        }

        # 단어 분리 및 분석
        words = re.findall(r'\b[a-zA-Z]+\b', cleaned_text.lower())
        
        manual_results = []
        pos_counts = {'명사': 0, '동사': 0, '형용사': 0, '부사': 0, '기타': 0}
        
        for word in words:
            if len(word) < 2:
                continue
                
            pos_tag = '기타'
            rule_applied = ''
            
            # 순서를 최적화하여 정확한 분류
            
            # 1. 일반적인 단어 목록 우선 확인 (가장 정확)
            if word in manual_rules['noun_patterns']['common_nouns']:
                pos_tag = '명사'
                rule_applied = "일반 명사 목록"
            elif word in manual_rules['verb_patterns']['common_verbs']:
                pos_tag = '동사'
                rule_applied = "일반 동사 목록"
            elif word in manual_rules['adjective_patterns']['common_adjectives']:
                pos_tag = '형용사'
                rule_applied = "일반 형용사 목록"
            elif word in manual_rules['adverb_patterns']['common_adverbs']:
                pos_tag = '부사'
                rule_applied = "일반 부사 목록"
            
            # 2. 어미 패턴 확인 (특별한 순서로 정확도 향상)
            # 부사는 -ly로 끝나는 경우가 많으므로 우선 확인
            elif word.endswith('ly') and len(word) > 3:
                pos_tag = '부사'
                rule_applied = "어미 패턴: -ly"
            
            # 형용사 어미들
            elif any(word.endswith(ending) for ending in ['ful', 'less', 'ive', 'able', 'ible', 'ous', 'ious']):
                pos_tag = '형용사'
                matching_ending = next(ending for ending in ['ful', 'less', 'ive', 'able', 'ible', 'ous', 'ious'] if word.endswith(ending))
                rule_applied = f"어미 패턴: -{matching_ending}"
            
            # 명사 어미들
            elif any(word.endswith(ending) for ending in ['tion', 'sion', 'ness', 'ment', 'ship', 'hood', 'ity']):
                pos_tag = '명사'
                matching_ending = next(ending for ending in ['tion', 'sion', 'ness', 'ment', 'ship', 'hood', 'ity'] if word.endswith(ending))
                rule_applied = f"어미 패턴: -{matching_ending}"
            
            # 동사 어미들 (과거형, 현재분사 등)
            elif word.endswith('ed') and len(word) > 3:
                pos_tag = '동사'
                rule_applied = "어미 패턴: -ed (과거형)"
            elif word.endswith('ing') and len(word) > 4:
                pos_tag = '동사'
                rule_applied = "어미 패턴: -ing (현재분사)"
            
            # 3. 기타 어미 패턴들
            elif any(word.endswith(ending) for ending in manual_rules['adjective_patterns']['endings']):
                pos_tag = '형용사'
                rule_applied = f"어미 패턴 (형용사)"
            elif any(word.endswith(ending) for ending in manual_rules['noun_patterns']['endings']):
                pos_tag = '명사'
                rule_applied = f"어미 패턴 (명사)"
            elif any(word.endswith(ending) for ending in manual_rules['verb_patterns']['endings']):
                pos_tag = '동사'
                rule_applied = f"어미 패턴 (동사)"
            elif any(word.endswith(ending) for ending in manual_rules['adverb_patterns']['endings']):
                pos_tag = '부사'
                rule_applied = f"어미 패턴 (부사)"
            
            pos_counts[pos_tag] += 1
            if len(manual_results) < 30:  # 처음 30개만 저장
                manual_results.append({
                    'word': word,
                    'pos': pos_tag,
                    'rule': rule_applied if rule_applied else '기타 분류'
                })
        
        # 비율 계산
        total_words = sum(pos_counts.values())
        pos_ratios = {}
        if total_words > 0:
            for pos, count in pos_counts.items():
                pos_ratios[pos] = (count / total_words) * 100
        
        # 단어 목록 크기 계산
        total_known_words = (
            len(manual_rules['noun_patterns']['common_nouns']) +
            len(manual_rules['verb_patterns']['common_verbs']) +
            len(manual_rules['adjective_patterns']['common_adjectives']) +
            len(manual_rules['adverb_patterns']['common_adverbs'])
        )
        
        return {
            'method': '수동 규칙 기반 (완전 개선)',
            'word_analysis': manual_results,
            'pos_counts': pos_counts,
            'pos_ratios': pos_ratios,
            'total_words': total_words,
            'known_words_count': total_known_words,
            'explanation': f"대폭 확장된 단어 목록 ({total_known_words}개) + 최적화된 어미 패턴으로 완전 개선. 총 {total_words}개 단어 중 기타: {pos_counts['기타']}개 ({pos_ratios.get('기타', 0):.1f}%)"
        }

    def educational_pos_analysis_step2_nltk_basic(self, text):
        """2단계: NLTK 기본 품사 태깅 - 교육용"""
        if not text or pd.isna(text):
            return {}
        
        # 에세이 내용만 추출
        cleaned_text = self.extract_essay_content(text)
        if not cleaned_text:
            return {}
        
        try:
            # NLTK 품사 태깅
            words = nltk.word_tokenize(cleaned_text)
            pos_tags = nltk.pos_tag(words)
            
            # 품사 태그 설명
            pos_tag_explanations = {
                'NN': '명사(단수)', 'NNS': '명사(복수)', 'NNP': '고유명사(단수)', 'NNPS': '고유명사(복수)',
                'VB': '동사(원형)', 'VBD': '동사(과거)', 'VBG': '동사(현재분사)', 'VBN': '동사(과거분사)', 
                'VBP': '동사(현재)', 'VBZ': '동사(3인칭단수)',
                'JJ': '형용사', 'JJR': '형용사(비교급)', 'JJS': '형용사(최상급)',
                'RB': '부사', 'RBR': '부사(비교급)', 'RBS': '부사(최상급)',
                'DT': '관사', 'IN': '전치사', 'CC': '접속사', 'PRP': '대명사'
            }
            
            # 주요 품사별 분류
            pos_categories = {
                '명사': ['NN', 'NNS', 'NNP', 'NNPS'],
                '동사': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
                '형용사': ['JJ', 'JJR', 'JJS'],
                '부사': ['RB', 'RBR', 'RBS'],
                '기능어': ['DT', 'IN', 'CC', 'PRP', 'TO', 'WDT', 'WP', 'WRB']
            }
            
            # 결과 분석
            detailed_analysis = []
            category_counts = {'명사': 0, '동사': 0, '형용사': 0, '부사': 0, '기능어': 0, '기타': 0}
            
            for word, pos in pos_tags:
                if word.isalpha() and len(word) > 1:
                    category = '기타'
                    for cat, tags in pos_categories.items():
                        if pos in tags:
                            category = cat
                            break
                    
                    category_counts[category] += 1
                    
                    detailed_analysis.append({
                        'word': word.lower(),
                        'pos_tag': pos,
                        'pos_explanation': pos_tag_explanations.get(pos, '기타'),
                        'category': category
                    })
            
            # 비율 계산
            total_words = sum(category_counts.values())
            category_ratios = {}
            if total_words > 0:
                for category, count in category_counts.items():
                    category_ratios[category] = (count / total_words) * 100
            
            return {
                'method': 'NLTK 품사 태깅',
                'detailed_analysis': detailed_analysis[:30],  # 처음 30개만
                'category_counts': category_counts,
                'category_ratios': category_ratios,
                'total_words': total_words,
                'pos_tag_explanations': pos_tag_explanations,
                'pos_tagged_words': pos_tags,  # main.py에서 필요한 키 추가
                'explanation': "NLTK의 기계학습 기반 품사 태거를 사용한 정확한 분류"
            }
            
        except Exception as e:
            return {
                'method': 'NLTK 품사 태깅',
                'error': f'NLTK 분석 중 오류: {str(e)}',
                'explanation': 'NLTK 라이브러리 문제로 분석을 수행할 수 없습니다.'
            }

    def educational_pos_analysis_step3_pattern_discovery(self, text):
        """3단계: 패턴 발견 및 언어적 특성 분석 - 교육용"""
        if not text or pd.isna(text):
            return {}
        
        # 에세이 내용만 추출
        cleaned_text = self.extract_essay_content(text)
        if not cleaned_text:
            return {}
        
        try:
            # 문장별 분석
            sentences = nltk.sent_tokenize(cleaned_text)
            
            sentence_analysis = []
            overall_patterns = {
                'avg_sentence_length': 0,
                'noun_density': [],
                'verb_density': [],
                'adj_density': [],
                'complex_sentences': 0,
                'simple_sentences': 0
            }
            
            for i, sentence in enumerate(sentences[:5]):  # 처음 5문장만
                words = nltk.word_tokenize(sentence)
                pos_tags = nltk.pos_tag(words)
                
                # 품사별 카운트
                nouns = sum(1 for _, pos in pos_tags if pos.startswith('NN'))
                verbs = sum(1 for _, pos in pos_tags if pos.startswith('VB'))
                adjectives = sum(1 for _, pos in pos_tags if pos.startswith('JJ'))
                total_content_words = nouns + verbs + adjectives
                
                if total_content_words > 0:
                    noun_ratio = (nouns / total_content_words) * 100
                    verb_ratio = (verbs / total_content_words) * 100
                    adj_ratio = (adjectives / total_content_words) * 100
                else:
                    noun_ratio = verb_ratio = adj_ratio = 0
                
                # 문장 복잡도 판단
                is_complex = len(words) > 15 or any(word.lower() in ['because', 'although', 'since', 'while', 'if'] for word, _ in pos_tags)
                
                if is_complex:
                    overall_patterns['complex_sentences'] += 1
                else:
                    overall_patterns['simple_sentences'] += 1
                
                overall_patterns['noun_density'].append(noun_ratio)
                overall_patterns['verb_density'].append(verb_ratio)
                overall_patterns['adj_density'].append(adj_ratio)
                
                sentence_analysis.append({
                    'sentence_num': i + 1,
                    'sentence': sentence[:100] + "..." if len(sentence) > 100 else sentence,
                    'word_count': len(words),
                    'nouns': nouns,
                    'verbs': verbs,
                    'adjectives': adjectives,
                    'noun_ratio': noun_ratio,
                    'verb_ratio': verb_ratio,
                    'adj_ratio': adj_ratio,
                    'complexity': 'Complex' if is_complex else 'Simple',
                    'content_words': total_content_words
                })
            
            # 전체 패턴 분석
            if overall_patterns['noun_density']:
                avg_noun_density = sum(overall_patterns['noun_density']) / len(overall_patterns['noun_density'])
                avg_verb_density = sum(overall_patterns['verb_density']) / len(overall_patterns['verb_density'])
                avg_adj_density = sum(overall_patterns['adj_density']) / len(overall_patterns['adj_density'])
            else:
                avg_noun_density = avg_verb_density = avg_adj_density = 0
            
            # 글쓰기 스타일 분석
            writing_style = {}
            
            if avg_noun_density > 50:
                writing_style['noun_heavy'] = "명사 중심의 정보 전달형 글쓰기"
            elif avg_verb_density > 30:
                writing_style['verb_heavy'] = "동사 중심의 동적인 글쓰기"
            elif avg_adj_density > 20:
                writing_style['descriptive'] = "형용사가 풍부한 묘사적 글쓰기"
            else:
                writing_style['balanced'] = "균형잡힌 글쓰기"
            
            complexity_ratio = overall_patterns['complex_sentences'] / (overall_patterns['complex_sentences'] + overall_patterns['simple_sentences']) * 100 if (overall_patterns['complex_sentences'] + overall_patterns['simple_sentences']) > 0 else 0
            
            # 공통 패턴 생성 (main.py에서 필요)
            common_patterns = []
            if avg_noun_density > 50:
                common_patterns.append("NN-DT-NN (명사-관사-명사 패턴 빈발)")
            if avg_verb_density > 30:
                common_patterns.append("VB-DT-NN (동사-관사-명사 패턴)")
            if complexity_ratio > 50:
                common_patterns.append("IN-PRP-VB (전치사-대명사-동사 복합 패턴)")
            if len(sentence_analysis) > 3:
                common_patterns.append("PRP-VB-JJ (대명사-동사-형용사 패턴)")
            common_patterns.append("DT-JJ-NN (관사-형용사-명사 패턴)")

            # 언어적 특성 정리 (main.py에서 필요)
            linguistic_features = {
                'complexity_score': complexity_ratio / 100,  # 0-1 범위로 정규화
                'lexical_diversity': min(1.0, avg_adj_density / 25.0),  # 어휘 다양성 추정
                'avg_sentence_length': sum(sa['word_count'] for sa in sentence_analysis) / len(sentence_analysis) if sentence_analysis else 0
            }

            return {
                'method': '패턴 발견 및 언어적 특성',
                'sentence_analysis': sentence_analysis,
                'overall_patterns': {
                    'avg_noun_density': avg_noun_density,
                    'avg_verb_density': avg_verb_density,
                    'avg_adj_density': avg_adj_density,
                    'complexity_ratio': complexity_ratio,
                    'total_sentences': len(sentence_analysis)
                },
                'writing_style': writing_style,
                'common_patterns': common_patterns,  # main.py에서 필요한 키 추가
                'linguistic_features': linguistic_features,  # main.py에서 필요한 키 추가
                'explanation': "문장별 품사 분포를 분석하여 글쓰기 패턴과 스타일을 발견"
            }
            
        except Exception as e:
            return {
                'method': '패턴 발견 및 언어적 특성',
                'error': f'패턴 분석 중 오류: {str(e)}',
                'explanation': '패턴 분석을 수행할 수 없습니다.'
            }

    def educational_pos_comparison(self, essay_data, essay_index=0):
        """세 가지 품사 분석 방법 비교 - 교육용"""
        if essay_data.empty or essay_index >= len(essay_data):
            return {}
        
        # 선택된 에세이
        selected_essay = essay_data.iloc[essay_index]
        essay_text = selected_essay.get('essay_text', '')
        
        if not essay_text:
            return {}
        
        # 세 가지 방법으로 분석
        result1 = self.educational_pos_analysis_step1_manual_rules(essay_text)
        result2 = self.educational_pos_analysis_step2_nltk_basic(essay_text)
        result3 = self.educational_pos_analysis_step3_pattern_discovery(essay_text)
        
        return {
            'essay_info': {
                'essay_num': essay_index + 1,
                'topic': selected_essay.get('topic_name', 'Unknown'),
                'text_preview': self.extract_essay_content(essay_text)[:200] + "..."
            },
            'method1_manual': result1,
            'method2_nltk': result2,
            'method3_patterns': result3
        }
    
    def educational_pos_analysis_step4_benchmarking(self, text):
        """4단계: 벤치마킹 및 글쓰기 수준 진단"""
        
        # 벤치마크 데이터
        BENCHMARK_DATA = {
            "middle_school_excellent": {
                "name": "🎓 중학생 우수작",
                "noun_ratio": 42.1,
                "verb_ratio": 22.8,
                "adj_ratio": 15.3,
                "complexity_ratio": 58.7,
                "vocabulary_diversity": 0.72
            },
            "academic_essay": {
                "name": "📚 학술 에세이",
                "noun_ratio": 45.2,
                "verb_ratio": 18.3,
                "adj_ratio": 12.1,
                "complexity_ratio": 78.5,
                "vocabulary_diversity": 0.85
            },
            "creative_writing": {
                "name": "🎨 창의적 글쓰기",
                "noun_ratio": 38.7,
                "verb_ratio": 25.4,
                "adj_ratio": 18.2,
                "complexity_ratio": 65.2,
                "vocabulary_diversity": 0.78
            }
        }
        
        try:
            # 사용자 텍스트 분석 (기존 메서드들 활용)
            step1_result = self.educational_pos_analysis_step1_manual_rules(text)
            step2_result = self.educational_pos_analysis_step2_nltk_basic(text)
            step3_result = self.educational_pos_analysis_step3_pattern_discovery(text)
            
            # 사용자 통계 추출
            user_stats = self._extract_user_statistics(step1_result, step2_result, step3_result, text)
            
            # 각 벤치마크와 비교
            comparisons = {}
            for benchmark_key, benchmark_data in BENCHMARK_DATA.items():
                comparison = self._compare_with_benchmark(user_stats, benchmark_data)
                comparisons[benchmark_key] = comparison
            
            # 종합 평가
            overall_assessment = self._calculate_writing_level(user_stats, BENCHMARK_DATA)
            
            # 개선 제안
            improvement_suggestions = self._generate_improvement_suggestions(user_stats, BENCHMARK_DATA)
            
            return {
                'user_stats': user_stats,
                'benchmark_comparisons': comparisons,
                'overall_assessment': overall_assessment,
                'improvement_suggestions': improvement_suggestions,
                'writing_strengths': self._identify_strengths(user_stats, BENCHMARK_DATA),
                'growth_areas': self._identify_growth_areas(user_stats, BENCHMARK_DATA)
            }
            
        except Exception as e:
            return {'error': f"벤치마킹 분석 중 오류: {str(e)}"}

    def _extract_user_statistics(self, step1_result, step2_result, step3_result, text):
        """사용자 텍스트 통계 추출"""
        
        # Step1에서 기본 품사 비율
        pos_ratios = step1_result.get('pos_ratios', {})
        
        # Step3에서 복잡도와 패턴
        patterns = step3_result.get('overall_patterns', {})
        
        # 어휘 다양성 계산
        words = text.lower().split()
        unique_words = len(set(words))
        total_words = len(words)
        vocabulary_diversity = unique_words / total_words if total_words > 0 else 0
        
        # 문장 수 계산
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_length = total_words / len(sentences) if sentences else 0
        
        return {
            'noun_ratio': pos_ratios.get('명사', 0),
            'verb_ratio': pos_ratios.get('동사', 0),
            'adj_ratio': pos_ratios.get('형용사', 0),
            'complexity_ratio': patterns.get('complexity_ratio', 0),
            'vocabulary_diversity': vocabulary_diversity,
            'total_words': total_words,
            'total_sentences': len(sentences),
            'avg_sentence_length': avg_sentence_length
        }

    def _compare_with_benchmark(self, user_stats, benchmark_data):
        """벤치마크와 비교 분석"""
        
        comparison = {
            'benchmark_name': benchmark_data['name'],
            'scores': {},
            'total_score': 0,
            'strengths': [],
            'improvements': []
        }
        
        # 각 지표별 비교 (100점 만점)
        metrics = ['noun_ratio', 'verb_ratio', 'adj_ratio', 'complexity_ratio', 'vocabulary_diversity']
        
        total_score = 0
        for metric in metrics:
            user_value = user_stats.get(metric, 0)
            benchmark_value = benchmark_data.get(metric, 0)
            
            # 차이를 점수로 변환 (차이가 적을수록 높은 점수)
            if benchmark_value > 0:
                difference = abs(user_value - benchmark_value) / benchmark_value
                score = max(0, 100 - (difference * 100))
            else:
                score = 50  # 기본 점수
            
            comparison['scores'][metric] = score
            total_score += score
        
        comparison['total_score'] = total_score / len(metrics)
        
        return comparison

    def _calculate_writing_level(self, user_stats, benchmark_data):
        """종합 글쓰기 수준 계산"""
        
        # 중학생 우수작 기준으로 평가
        middle_school_benchmark = benchmark_data['middle_school_excellent']
        comparison = self._compare_with_benchmark(user_stats, middle_school_benchmark)
        
        score = comparison['total_score']
        
        if score >= 85:
            level = "🏆 우수 (Excellent)"
            level_desc = "중학생 최상위 수준의 글쓰기 실력!"
            color = "success"
        elif score >= 70:
            level = "🥈 양호 (Good)"
            level_desc = "중학생 상위 수준의 안정적인 글쓰기"
            color = "info"
        elif score >= 55:
            level = "🥉 보통 (Average)"
            level_desc = "중학생 평균 수준, 꾸준한 연습이 필요"
            color = "warning"
        else:
            level = "📚 개선 필요 (Needs Improvement)"
            level_desc = "기초를 다지며 단계적으로 향상해보세요"
            color = "error"
        
        return {
            'level': level,
            'description': level_desc,
            'score': round(score, 1),
            'color': color
        }

    def _generate_improvement_suggestions(self, user_stats, benchmark_data):
        """개선 제안 생성"""
        
        suggestions = []
        middle_school = benchmark_data['middle_school_excellent']
        
        # 명사 사용 분석
        noun_diff = user_stats['noun_ratio'] - middle_school['noun_ratio']
        if noun_diff > 8:
            suggestions.append({
                'type': '명사 사용',
                'icon': '📝',
                'suggestion': '명사 사용을 줄이고 동사와 형용사로 더 생동감 있게 표현해보세요',
                'priority': 'high'
            })
        elif noun_diff < -8:
            suggestions.append({
                'type': '명사 사용',
                'icon': '🎯',
                'suggestion': '더 구체적이고 정확한 명사를 사용해 내용을 명확하게 전달해보세요',
                'priority': 'medium'
            })
        
        # 동사 활용 분석
        verb_diff = user_stats['verb_ratio'] - middle_school['verb_ratio']
        if verb_diff < -5:
            suggestions.append({
                'type': '동사 활용',
                'icon': '🎬',
                'suggestion': '다양한 동작 동사를 활용해 글에 생동감을 더해보세요',
                'priority': 'high'
            })
        
        # 형용사 표현 분석
        adj_diff = user_stats['adj_ratio'] - middle_school['adj_ratio']
        if adj_diff < -3:
            suggestions.append({
                'type': '형용사 표현',
                'icon': '🎨',
                'suggestion': '적절한 형용사를 사용해 묘사를 더 풍부하고 생생하게 만들어보세요',
                'priority': 'medium'
            })
        elif adj_diff > 8:
            suggestions.append({
                'type': '형용사 표현',
                'icon': '✂️',
                'suggestion': '과도한 형용사 사용을 줄이고 핵심 내용에 집중해보세요',
                'priority': 'low'
            })
        
        # 어휘 다양성 분석
        diversity_diff = user_stats['vocabulary_diversity'] - middle_school['vocabulary_diversity']
        if diversity_diff < -0.1:
            suggestions.append({
                'type': '어휘 다양성',
                'icon': '📚',
                'suggestion': '같은 단어 반복을 피하고 다양한 어휘를 사용해보세요',
                'priority': 'high'
            })
        
        # 문장 복잡도 분석
        complexity_diff = user_stats['complexity_ratio'] - middle_school['complexity_ratio']
        if complexity_diff < -15:
            suggestions.append({
                'type': '문장 구조',
                'icon': '🧠',
                'suggestion': '단순 문장과 복합 문장을 적절히 조합해 글의 흐름을 개선해보세요',
                'priority': 'medium'
            })
        elif complexity_diff > 20:
            suggestions.append({
                'type': '문장 구조',
                'icon': '✨',
                'suggestion': '너무 복잡한 문장보다는 명확하고 간결한 표현을 연습해보세요',
                'priority': 'medium'
            })
        
        # 기본 제안이 없으면 격려 메시지
        if not suggestions:
            suggestions.append({
                'type': '전체적 평가',
                'icon': '🌟',
                'suggestion': '균형 잡힌 좋은 글쓰기를 하고 있습니다! 창의적 표현에 도전해보세요',
                'priority': 'low'
            })
        
        return suggestions

    def _identify_strengths(self, user_stats, benchmark_data):
        """글쓰기 강점 식별"""
        
        strengths = []
        middle_school = benchmark_data['middle_school_excellent']
        
        # 각 영역별 강점 체크
        if abs(user_stats['noun_ratio'] - middle_school['noun_ratio']) <= 5:
            strengths.append("명사 사용이 적절하고 균형잡혀 있습니다")
        
        if abs(user_stats['verb_ratio'] - middle_school['verb_ratio']) <= 3:
            strengths.append("동사 활용이 우수하여 글에 생동감이 있습니다")
        
        if abs(user_stats['adj_ratio'] - middle_school['adj_ratio']) <= 3:
            strengths.append("형용사 표현이 적절하여 묘사가 풍부합니다")
        
        if user_stats['vocabulary_diversity'] >= middle_school['vocabulary_diversity']:
            strengths.append("어휘 사용이 다양하고 풍부합니다")
        
        if abs(user_stats['complexity_ratio'] - middle_school['complexity_ratio']) <= 10:
            strengths.append("문장 구조가 적절히 복합적입니다")
        
        if user_stats['avg_sentence_length'] >= 8 and user_stats['avg_sentence_length'] <= 15:
            strengths.append("문장 길이가 읽기 좋게 조절되어 있습니다")
        
        return strengths if strengths else ["꾸준한 연습으로 실력이 향상되고 있습니다"]

    def _identify_growth_areas(self, user_stats, benchmark_data):
        """성장이 필요한 영역 식별"""
        
        growth_areas = []
        middle_school = benchmark_data['middle_school_excellent']
        
        # 개선이 가장 필요한 영역 식별
        improvements_needed = []
        
        noun_diff = abs(user_stats['noun_ratio'] - middle_school['noun_ratio'])
        if noun_diff > 8:
            improvements_needed.append(('명사 사용 균형', noun_diff))
        
        verb_diff = abs(user_stats['verb_ratio'] - middle_school['verb_ratio'])
        if verb_diff > 5:
            improvements_needed.append(('동사 활용 다양성', verb_diff))
        
        adj_diff = abs(user_stats['adj_ratio'] - middle_school['adj_ratio'])
        if adj_diff > 5:
            improvements_needed.append(('형용사 표현 조절', adj_diff))
        
        if user_stats['vocabulary_diversity'] < middle_school['vocabulary_diversity'] - 0.1:
            improvements_needed.append(('어휘 다양성 향상', 
                                    (middle_school['vocabulary_diversity'] - user_stats['vocabulary_diversity']) * 100))
        
        complexity_diff = abs(user_stats['complexity_ratio'] - middle_school['complexity_ratio'])
        if complexity_diff > 15:
            improvements_needed.append(('문장 구조 개선', complexity_diff))
        
        # 가장 큰 차이를 보이는 영역부터 정렬
        improvements_needed.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 3개 영역만 선택
        for area, _ in improvements_needed[:3]:
            growth_areas.append(area)
        
        return growth_areas if growth_areas else ["현재 수준을 유지하며 창의성 개발에 집중"]



    def comprehensive_writing_analysis(self, text):
        """통합 글쓰기 수준 종합 진단"""
        
        
        try:
            # 1단계: 통계적 벤치마킹 분석
            step1_result = self._statistical_benchmarking_analysis(text)
            
            # 2단계: 어휘 수준 분석
            step2_result = self._vocabulary_level_analysis(text)
            
            # 3단계: 문법 오류 패턴 분석
            step3_result = self.analyze_grammar_patterns(text)
            
            # 4단계: 문장 유사도 분석
            step4_result = self._sentence_similarity_analysis(text)
            
            # 5단계: 종합 진단
            step5_result = self._comprehensive_assessment(text, step1_result, step2_result, step3_result, step4_result)
            
            return {
                'step1_stats': step1_result,
                'step2_vocabulary': step2_result,
                'step3_grammar': step3_result,
                'step4_similarity': step4_result,
                'step5_comprehensive': step5_result,
                'overall_score': step5_result['overall_score'],
                'final_level': step5_result['final_level'],
                'improvement_roadmap': step5_result['improvement_roadmap']
            }
            
        except Exception as e:
            return {'error': f"통합 글쓰기 진단 중 오류: {str(e)}"}

    def _statistical_benchmarking_analysis(self, text):
        """1단계: 통계적 벤치마킹 분석"""
        
        
        # 벤치마크 데이터
        EXPERT_BENCHMARKS = {
            "middle_school_excellent": {
                "name": "🎓 중학생 우수작",
                "noun_ratio": 42.1,
                "verb_ratio": 22.8,
                "adj_ratio": 15.3,
                "complexity_ratio": 58.7,
                "vocabulary_diversity": 0.72,
                "avg_sentence_length": 12.5
            },
            "academic_essay": {
                "name": "📚 학술 에세이",
                "noun_ratio": 45.2,
                "verb_ratio": 18.3,
                "adj_ratio": 12.1,
                "complexity_ratio": 78.5,
                "vocabulary_diversity": 0.85,
                "avg_sentence_length": 16.2
            },
            "creative_writing": {
                "name": "🎨 창의적 글쓰기",
                "noun_ratio": 38.7,
                "verb_ratio": 25.4,
                "adj_ratio": 18.2,
                "complexity_ratio": 65.2,
                "vocabulary_diversity": 0.78,
                "avg_sentence_length": 14.1
            }
        }
        
        # 사용자 텍스트 통계 계산
        user_stats = self._calculate_text_statistics(text)
        
        # 각 벤치마크와 비교
        benchmark_scores = {}
        for key, benchmark in EXPERT_BENCHMARKS.items():
            similarity_score = self._calculate_similarity_score(user_stats, benchmark)
            benchmark_scores[key] = {
                'benchmark_name': benchmark['name'],
                'similarity_score': similarity_score,
                'individual_scores': self._get_individual_scores(user_stats, benchmark)
            }
        
        # 최고 유사도 벤치마크 선정
        best_match = max(benchmark_scores.items(), key=lambda x: x[1]['similarity_score'])
        
        return {
            'user_statistics': user_stats,
            'benchmark_comparisons': benchmark_scores,
            'best_match': best_match,
            'statistical_score': best_match[1]['similarity_score'],
            'insights': self._generate_statistical_insights(user_stats, EXPERT_BENCHMARKS)
        }

    def _vocabulary_level_analysis(self, text):
        """2단계: 어휘 수준 분석 (Word Embedding 시뮬레이션)"""
        
        words = text.lower().split()
        
        # 고급 어휘 사전 (실제로는 Word2Vec/GloVe 사용)
        ADVANCED_VOCABULARY = {
            # 학술적 어휘
            'academic': ['analyze', 'synthesize', 'evaluate', 'critique', 'demonstrate', 
                        'illustrate', 'establish', 'determine', 'investigate', 'examine',
                        'significant', 'substantial', 'comprehensive', 'fundamental', 'crucial'],
            
            # 고급 형용사
            'descriptive': ['magnificent', 'extraordinary', 'remarkable', 'exceptional', 
                        'profound', 'intricate', 'sophisticated', 'elaborate', 'vivid',
                        'compelling', 'fascinating', 'intriguing', 'captivating'],
            
            # 연결 어구
            'transitions': ['furthermore', 'consequently', 'nevertheless', 'moreover', 
                        'therefore', 'however', 'subsequently', 'additionally', 'ultimately',
                        'specifically', 'particularly', 'essentially', 'significantly'],
            
            # 고급 동사
            'advanced_verbs': ['enhance', 'facilitate', 'demonstrate', 'implement', 'establish',
                            'contribute', 'emphasize', 'illustrate', 'represent', 'indicate',
                            'reflect', 'reveal', 'suggest', 'imply', 'encompass']
        }
        
        # 어휘 분석
        total_words = len(words)
        unique_words = len(set(words))
        vocabulary_diversity = unique_words / total_words if total_words > 0 else 0
        
        # 고급 어휘 사용률 계산
        advanced_count = 0
        category_usage = {}
        
        for category, vocab_list in ADVANCED_VOCABULARY.items():
            category_count = sum(1 for word in words if word in vocab_list)
            category_usage[category] = {
                'count': category_count,
                'ratio': (category_count / total_words * 100) if total_words > 0 else 0
            }
            advanced_count += category_count
        
        advanced_vocabulary_ratio = (advanced_count / total_words * 100) if total_words > 0 else 0
        
        # 어휘 수준 점수 계산 (100점 만점)
        diversity_score = min(vocabulary_diversity * 100, 50)  # 다양성 최대 50점
        advanced_score = min(advanced_vocabulary_ratio * 2, 50)  # 고급어휘 최대 50점
        vocabulary_score = diversity_score + advanced_score
        
        # 어휘 수준 평가
        if vocabulary_score >= 80:
            level = "🏆 고급 (Advanced)"
            level_desc = "풍부하고 정교한 어휘 사용"
        elif vocabulary_score >= 60:
            level = "🥈 중급 (Intermediate)"
            level_desc = "적절한 어휘 구사력"
        elif vocabulary_score >= 40:
            level = "🥉 기초 (Basic)"
            level_desc = "기본적인 어휘 사용"
        else:
            level = "📚 입문 (Beginner)"
            level_desc = "어휘 확장이 필요"
        
        # 어휘 복잡도 분석 추가
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        unique_word_ratio = (unique_words / total_words * 100) if total_words > 0 else 0
        
        # 수준별 어휘 분석
        basic_words = sum(1 for word in words if len(word) <= 4)
        intermediate_words = sum(1 for word in words if 5 <= len(word) <= 7)
        advanced_words = sum(1 for word in words if len(word) > 7)
        academic_words = advanced_count
        
        return {
            'overall_level': level,
            'advanced_vocabulary_ratio': advanced_vocabulary_ratio,
            'academic_vocabulary_score': advanced_score,
            'level_analysis': {
                'basic_words': basic_words,
                'intermediate_words': intermediate_words,
                'advanced_words': advanced_words,
                'academic_words': academic_words
            },
            'complexity_analysis': {
                'avg_word_length': avg_word_length,
                'unique_word_ratio': unique_word_ratio,
                'sophistication_score': vocabulary_score,
                'level_description': level_desc
            },
            'vocabulary_recommendations': self._generate_vocabulary_recommendations(advanced_vocabulary_ratio, vocabulary_score),
            'category_usage': category_usage,
            'vocabulary_diversity': vocabulary_diversity
        }

    def _sentence_similarity_analysis(self, text):
        """3단계: 문장 유사도 분석 (전문가 글과의 논리성 비교)"""
        
        try:
            import nltk
            sentences = nltk.sent_tokenize(text)
        except:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if not sentences:
            return {
                'average_similarity': 0,
                'coherence_score': 0,
                'logical_flow_level': 'Unknown',
                'sentence_pair_analysis': [],
                'topic_consistency': {}
            }
        
        # 논리적 연결어 분석
        logical_connectors = ['however', 'therefore', 'furthermore', 'moreover', 'consequently',
                            'nevertheless', 'additionally', 'specifically', 'ultimately', 'initially',
                            'firstly', 'secondly', 'finally', 'in conclusion', 'for example', 'such as']
        
        transition_words = ['but', 'and', 'or', 'so', 'because', 'since', 'while', 'although', 
                          'unless', 'before', 'after', 'when', 'if', 'thus', 'hence']
        
        # 문장 간 유사도 계산 (단어 기반 Jaccard 유사도)
        sentence_similarities = []
        sentence_pairs = []
        
        for i in range(len(sentences)-1):
            sentence1 = sentences[i].lower()
            sentence2 = sentences[i+1].lower()
            
            words1 = set(word.strip('.,!?;:') for word in sentence1.split() if word.isalpha())
            words2 = set(word.strip('.,!?;:') for word in sentence2.split() if word.isalpha())
            
            if words1 and words2:
                intersection = words1.intersection(words2)
                union = words1.union(words2)
                similarity = len(intersection) / len(union) if union else 0
                sentence_similarities.append(similarity)
                
                if len(sentence_pairs) < 5:  # 상위 5개만 저장
                    sentence_pairs.append({
                        'similarity': similarity,
                        'sentence1_preview': sentences[i][:100],
                        'sentence2_preview': sentences[i+1][:100]
                    })
        
        # 평균 유사도
        avg_similarity = sum(sentence_similarities) / len(sentence_similarities) if sentence_similarities else 0
        
        # 논리적 흐름 분석
        connector_count = 0
        for sentence in sentences:
            sentence_lower = sentence.lower()
            connector_count += sum(1 for conn in logical_connectors + transition_words if conn in sentence_lower)
        
        connector_ratio = connector_count / len(sentences) if sentences else 0
        
        # 일관성 점수 계산
        coherence_score = min(100, (avg_similarity * 50) + (connector_ratio * 30) + 20)
        
        # 논리적 흐름 수준 결정
        if coherence_score >= 80:
            logical_flow = "🟢 매우 우수"
        elif coherence_score >= 60:
            logical_flow = "🟡 양호"
        elif coherence_score >= 40:
            logical_flow = "🟠 보통"
        else:
            logical_flow = "🔴 개선 필요"
        
        # 주제 일관성 분석 (키워드 기반)
        all_words = []
        for sentence in sentences:
            words = [word.lower().strip('.,!?;:') for word in sentence.split() if word.isalpha() and len(word) > 3]
            all_words.extend(words)
        
        if all_words:
            from collections import Counter
            word_freq = Counter(all_words)
            top_words = word_freq.most_common(5)
            
            # 주요 키워드의 분포
            total_occurrences = sum(freq for word, freq in top_words)
            if total_occurrences > 0:
                main_theme_strength = (top_words[0][1] / total_occurrences * 100) if top_words else 0
                topic_drift_score = max(0, 100 - main_theme_strength)
            else:
                main_theme_strength = 0
                topic_drift_score = 100
        else:
            main_theme_strength = 0
            topic_drift_score = 100
        
        return {
            'average_similarity': avg_similarity,
            'coherence_score': coherence_score,
            'logical_flow_level': logical_flow,
            'sentence_pair_analysis': sentence_pairs,
            'topic_consistency': {
                'topic_drift_score': topic_drift_score,
                'main_theme_strength': main_theme_strength
            }
        }
    
    def _generate_vocabulary_recommendations(self, advanced_ratio, vocab_score):
        """어휘 개선 추천사항 생성"""
        recommendations = []
        
        if advanced_ratio < 10:
            recommendations.append("고급 어휘 사용을 늘려보세요 (현재 {:.1f}%)".format(advanced_ratio))
            recommendations.append("학술적 동사(analyze, evaluate, demonstrate) 활용 연습")
        
        if vocab_score < 60:
            recommendations.append("어휘 다양성을 높이기 위해 동의어 활용을 늘려보세요")
            recommendations.append("전문 분야별 어휘 학습을 권장합니다")
        
        if advanced_ratio > 20:
            recommendations.append("우수한 고급 어휘 사용력을 보입니다")
        
        return recommendations[:3] if recommendations else ["현재 어휘 수준이 적절합니다"]

    def _comprehensive_assessment(self, text, step1_result, step2_result, step3_result, step4_result):
        """5단계: 종합 진단 및 개선 로드맵"""
        
        # 각 단계별 점수 (100점 만점으로 정규화)
        statistical_score = step1_result.get('statistical_score', 0)
        vocabulary_score = step2_result.get('complexity_analysis', {}).get('sophistication_score', 0)
        grammar_score = step3_result.get('grammar_score', 100)  # 새로 추가된 문법 점수
        similarity_score = step4_result.get('coherence_score', 0)
        
        # 가중치 적용 종합 점수 (4단계로 확장)
        overall_score = (
            statistical_score * 0.25 +  # 통계적 특성 25%
            vocabulary_score * 0.25 +   # 어휘 수준 25%
            grammar_score * 0.25 +      # 문법 정확성 25%
            similarity_score * 0.25     # 논리적 구성 25%
        )
        
        # 최종 등급 판정
        if overall_score >= 85:
            final_level = "🏆 우수 (Excellent)"
            level_desc = "전문가 수준의 뛰어난 글쓰기 실력"
            level_color = "success"
        elif overall_score >= 70:
            final_level = "🥈 양호 (Good)"
            level_desc = "상급자 수준의 안정적인 글쓰기"
            level_color = "info"
        elif overall_score >= 55:
            final_level = "🥉 보통 (Average)"
            level_desc = "중급자 수준, 꾸준한 발전이 필요"
            level_color = "warning"
        else:
            final_level = "📚 향상 필요 (Developing)"
            level_desc = "기초 실력 향상에 집중 필요"
            level_color = "error"
        
        # 맞춤형 개선 로드맵 생성
        improvement_roadmap = self._generate_improvement_roadmap(
            statistical_score, vocabulary_score, similarity_score, overall_score
        )
        
        # 강점과 약점 분석
        strengths = []
        weaknesses = []
        
        if statistical_score >= 70:
            strengths.append("통계적 글쓰기 패턴이 우수함")
        else:
            weaknesses.append("품사 사용과 문장 구성의 균형 개선 필요")
        
        if vocabulary_score >= 70:
            strengths.append("어휘 사용이 풍부하고 다양함")
        else:
            weaknesses.append("어휘 다양성과 고급 표현 확장 필요")
        
        if similarity_score >= 70:
            strengths.append("논리적이고 일관성 있는 구성")
        else:
            weaknesses.append("문장 간 연결성과 논리적 흐름 개선 필요")
        
        return {
            'overall_score': round(overall_score, 1),
            'final_level': final_level,
            'level_description': level_desc,
            'level_color': level_color,
            'component_scores': {
                'statistical': round(statistical_score, 1),
                'vocabulary': round(vocabulary_score, 1),
                'similarity': round(similarity_score, 1)
            },
            'strengths': strengths,
            'weaknesses': weaknesses,
            'improvement_roadmap': improvement_roadmap
        }

    # 보조 메서드들
    def _calculate_text_statistics(self, text):
        """영어 텍스트 통계 계산"""
        
        if not text or not text.strip():
            return {
                'total_words': 0,
                'total_sentences': 0, 
                'unique_words': 0,
                'vocabulary_diversity': 0,
                'avg_sentence_length': 0,
                'noun_ratio': 0,
                'verb_ratio': 0,
                'adj_ratio': 0,
                'complexity_ratio': 0
            }
        
        try:
            import nltk
            
            # 문장과 단어 토큰화
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text.lower())
            
            # 알파벳 단어만 필터링 (길이 2 이상)
            words = [word for word in words if word.isalpha() and len(word) >= 2]
            total_words = len(words)
            
            
            if total_words == 0:
                return {
                    'total_words': 0,
                    'total_sentences': 0,
                    'unique_words': 0,
                    'vocabulary_diversity': 0,
                    'avg_sentence_length': 0,
                    'noun_ratio': 0,
                    'verb_ratio': 0,
                    'adj_ratio': 0,
                    'complexity_ratio': 0
                }
            
            # POS 태깅
            pos_tagged = nltk.pos_tag(words)
            
            # 품사별 개수 계산
            noun_count = sum(1 for word, pos in pos_tagged if pos.startswith('NN'))
            verb_count = sum(1 for word, pos in pos_tagged if pos.startswith('VB'))
            adj_count = sum(1 for word, pos in pos_tagged if pos.startswith('JJ'))
            
            # 복잡도 계산 (복합문/복문의 비율 추정)
            complex_indicators = sum(1 for word in words if word in ['however', 'therefore', 'furthermore', 'although', 'because', 'since', 'while', 'whereas', 'unless', 'though'])
            complexity_ratio = min(100, (complex_indicators / len(sentences)) * 50) if sentences else 0
            
            unique_words_count = len(set(words))
            diversity = unique_words_count / total_words
            
            return {
                'total_words': total_words,
                'total_sentences': len(sentences),
                'unique_words': unique_words_count,
                'vocabulary_diversity': diversity,
                'avg_sentence_length': total_words / len(sentences) if sentences else 0,
                'noun_ratio': (noun_count / total_words * 100),
                'verb_ratio': (verb_count / total_words * 100),
                'adj_ratio': (adj_count / total_words * 100),
                'complexity_ratio': complexity_ratio
            }
            
        except Exception as e:
            # NLTK가 없거나 오류 시 기본 분석
            import re
            
            # 단어 추출 (구두점 제거)
            words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            total_words = len(words)
            
            if total_words == 0:
                return {
                    'total_words': 0,
                    'total_sentences': 0,
                    'unique_words': 0,
                    'vocabulary_diversity': 0,
                    'avg_sentence_length': 0,
                    'noun_ratio': 0,
                    'verb_ratio': 0,
                    'adj_ratio': 0,
                    'complexity_ratio': 0
                }
            
            # 간단한 영어 규칙 기반 분석
            noun_count = sum(1 for word in words if word.endswith(('tion', 'sion', 'ment', 'ness', 'ity', 'ty', 'ence', 'ance')))
            verb_count = sum(1 for word in words if word.endswith(('ed', 'ing', 'ize', 'ise', 'ate')))
            adj_count = sum(1 for word in words if word.endswith(('ful', 'less', 'ous', 'ive', 'able', 'ible', 'al', 'ic')))
            
            unique_count = len(set(words))
            diversity = unique_count / total_words
            
            return {
                'total_words': total_words,
                'total_sentences': len(sentences),
                'unique_words': unique_count,
                'vocabulary_diversity': diversity,
                'avg_sentence_length': total_words / len(sentences) if sentences else 0,
                'noun_ratio': (noun_count / total_words * 100),
                'verb_ratio': (verb_count / total_words * 100),
                'adj_ratio': (adj_count / total_words * 100),
                'complexity_ratio': 50  # 기본값
            }

    def _calculate_similarity_score(self, user_stats, benchmark):
        """벤치마크와의 유사도 점수 계산"""
        
        metrics = ['noun_ratio', 'verb_ratio', 'adj_ratio', 'vocabulary_diversity', 'avg_sentence_length']
        
        total_score = 0
        valid_metrics = 0
        
        for metric in metrics:
            user_value = user_stats.get(metric, 0)
            benchmark_value = benchmark.get(metric, 0)
            
            if benchmark_value > 0:
                # 차이를 백분율로 계산
                difference = abs(user_value - benchmark_value) / benchmark_value
                # 차이가 적을수록 높은 점수 (최대 100점)
                score = max(0, 100 - (difference * 100))
                total_score += score
                valid_metrics += 1
        
        return total_score / valid_metrics if valid_metrics > 0 else 0

    def _get_individual_scores(self, user_stats, benchmark):
        """개별 지표별 점수"""
        
        metrics = {
            'noun_ratio': 'noun_score',
            'verb_ratio': 'verb_score', 
            'adj_ratio': 'adj_score',
            'vocabulary_diversity': 'diversity_score',
            'avg_sentence_length': 'length_score',
            'complexity_ratio': 'complexity_score'
        }
        
        scores = {}
        
        for metric, score_key in metrics.items():
            user_value = user_stats.get(metric, 0)
            benchmark_value = benchmark.get(metric, 0)
            
            if benchmark_value > 0:
                difference = abs(user_value - benchmark_value) / benchmark_value
                score = max(0, 100 - (difference * 100))
                scores[score_key] = score
            else:
                scores[score_key] = 50
        
        return scores

    def _generate_statistical_insights(self, user_stats, benchmarks):
        """통계적 분석 인사이트 생성"""
        
        insights = []
        
        # 문장 길이 분석
        avg_length = user_stats.get('avg_sentence_length', 0)
        if avg_length > 15:
            insights.append("문장이 길어 복잡한 내용을 다루고 있습니다")
        elif avg_length < 8:
            insights.append("문장이 짧아 간결한 표현을 선호합니다")
        else:
            insights.append("적절한 문장 길이로 읽기 좋은 글입니다")
        
        # 어휘 다양성 분석
        diversity = user_stats.get('vocabulary_diversity', 0)
        if diversity > 0.7:
            insights.append("어휘 사용이 매우 다양합니다")
        elif diversity < 0.5:
            insights.append("어휘 다양성 개선이 필요합니다")
        
        return insights

    def _generate_vocabulary_insights(self, category_usage, advanced_ratio):
        """어휘 분석 인사이트 생성"""
        
        insights = []
        
        if advanced_ratio > 5:
            insights.append("고급 어휘를 적절히 활용하고 있습니다")
        elif advanced_ratio < 2:
            insights.append("고급 어휘 사용을 늘려보세요")
        
        # 카테고리별 인사이트
        for category, usage in category_usage.items():
            if usage['ratio'] > 2:
                if category == 'academic':
                    insights.append("학술적 표현이 풍부합니다")
                elif category == 'transitions':
                    insights.append("논리적 연결이 우수합니다")
        
        return insights

    def _generate_similarity_insights(self, sentence_analysis, avg_coherence):
        """문장 유사도 인사이트 생성"""
        
        insights = []
        
        if avg_coherence > 70:
            insights.append("문장 구성이 논리적이고 일관성이 있습니다")
        elif avg_coherence < 50:
            insights.append("문장 간 연결성 개선이 필요합니다")
        
        # 논리적 연결어 사용 분석
        total_connectors = sum(s.get('logical_connectors', 0) for s in sentence_analysis)
        if total_connectors > len(sentence_analysis):
            insights.append("논리적 연결어를 잘 활용하고 있습니다")
        else:
            insights.append("논리적 연결어 사용을 늘려보세요")
        
        return insights

    def _generate_improvement_roadmap(self, stat_score, vocab_score, sim_score, overall_score):
        """맞춤형 개선 로드맵 생성"""
        
        roadmap = {
            'immediate_actions': [],
            'short_term_goals': [],
            'long_term_goals': []
        }
        
        # 즉시 실행 가능한 개선 사항
        if stat_score < 60:
            roadmap['immediate_actions'].append({
                'action': '문장 길이 조절하기',
                'description': '너무 긴 문장은 나누고, 짧은 문장은 연결해보세요',
                'priority': 'high'
            })
        
        if vocab_score < 60:
            roadmap['immediate_actions'].append({
                'action': '고급 어휘 학습',
                'description': '매일 새로운 어휘 3개씩 학습하고 글에 적용해보세요',
                'priority': 'high'
            })
        
        if sim_score < 60:
            roadmap['immediate_actions'].append({
                'action': '논리적 연결어 활용',
                'description': '"그러므로", "따라서", "하지만" 등을 활용해 문장을 연결해보세요',
                'priority': 'high'
            })
        
        # 단기 목표 (1-3개월)
        if overall_score < 70:
            roadmap['short_term_goals'].append('매주 에세이 1편씩 작성하며 체계적 연습')
            roadmap['short_term_goals'].append('다양한 장르의 우수작 읽기 및 분석')
            roadmap['short_term_goals'].append('글쓰기 피드백 받기 및 개선점 적용')
        
        # 장기 목표 (3-6개월)
        roadmap['long_term_goals'].append('개인만의 글쓰기 스타일 확립')
        roadmap['long_term_goals'].append('창의적이고 독창적인 표현력 개발')
        roadmap['long_term_goals'].append('복합적 주제에 대한 깊이 있는 글쓰기')
        
        return roadmap

    def analyze_grammar_patterns(self, text):
        """문법 오류 패턴 분석"""
        import re
        import nltk
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.tag import pos_tag
        
        try:
            sentences = sent_tokenize(text)
            grammar_analysis = {
                'total_sentences': len(sentences),
                'potential_errors': [],
                'error_patterns': {},
                'error_count_by_type': {},
                'sentences_with_issues': [],
                'grammar_score': 0
            }
            
            error_count = 0
            total_issues = 0
            
            for i, sentence in enumerate(sentences):
                sentence_issues = []
                words = word_tokenize(sentence)
                pos_tags = pos_tag(words)
                
                # 1. 주어-동사 수일치 검사 (간단한 패턴)
                subject_verb_issues = self._check_subject_verb_agreement(sentence, pos_tags)
                if subject_verb_issues:
                    sentence_issues.extend(subject_verb_issues)
                    error_count += len(subject_verb_issues)
                
                # 2. 시제 일관성 검사
                tense_issues = self._check_tense_consistency(sentence, pos_tags)
                if tense_issues:
                    sentence_issues.extend(tense_issues)
                    error_count += len(tense_issues)
                
                # 3. 관사 사용 검사 (기초 패턴)
                article_issues = self._check_article_usage(sentence, pos_tags)
                if article_issues:
                    sentence_issues.extend(article_issues)
                    error_count += len(article_issues)
                
                # 4. 전치사 사용 검사
                preposition_issues = self._check_preposition_patterns(sentence, pos_tags)
                if preposition_issues:
                    sentence_issues.extend(preposition_issues)
                    error_count += len(preposition_issues)
                
                # 5. 문장 구조 검사
                structure_issues = self._check_sentence_structure(sentence, pos_tags)
                if structure_issues:
                    sentence_issues.extend(structure_issues)
                    error_count += len(structure_issues)
                
                if sentence_issues:
                    grammar_analysis['sentences_with_issues'].append({
                        'sentence_number': i + 1,
                        'sentence': sentence,
                        'issues': sentence_issues
                    })
                    total_issues += len(sentence_issues)
            
            # 오류 패턴별 분류
            for sentence_data in grammar_analysis['sentences_with_issues']:
                for issue in sentence_data['issues']:
                    error_type = issue['type']
                    if error_type not in grammar_analysis['error_patterns']:
                        grammar_analysis['error_patterns'][error_type] = []
                    grammar_analysis['error_patterns'][error_type].append({
                        'sentence': sentence_data['sentence'],
                        'description': issue['description'],
                        'suggestion': issue.get('suggestion', '')
                    })
                    
                    # 오류 유형별 카운트
                    if error_type not in grammar_analysis['error_count_by_type']:
                        grammar_analysis['error_count_by_type'][error_type] = 0
                    grammar_analysis['error_count_by_type'][error_type] += 1
            
            # 문법 점수 계산 (100점 만점)
            if len(sentences) > 0:
                error_rate = total_issues / len(sentences)
                grammar_analysis['grammar_score'] = max(0, 100 - (error_rate * 20))
            else:
                grammar_analysis['grammar_score'] = 100
            
            # 주요 개선 영역 식별
            grammar_analysis['improvement_areas'] = self._identify_improvement_areas(grammar_analysis['error_count_by_type'])
            
            return grammar_analysis
            
        except Exception as e:
            return {
                'total_sentences': 0,
                'potential_errors': [],
                'error_patterns': {},
                'error_count_by_type': {},
                'sentences_with_issues': [],
                'grammar_score': 100,
                'improvement_areas': [],
                'error': f"문법 분석 중 오류 발생: {str(e)}"
            }
    
    def _check_subject_verb_agreement(self, sentence, pos_tags):
        """주어-동사 수일치 검사"""
        issues = []
        
        # 간단한 패턴 검사 (I/He/She + 동사 형태)
        sentence_lower = sentence.lower()
        
        # 자주 틀리는 패턴들
        error_patterns = [
            (r'\bi am\s+\w+ing\b', '현재진행형이 맞나요?'),
            (r'\bhe are\b|\bshe are\b', 'He/She는 is를 써야 합니다'),
            (r'\bthey is\b', 'They는 are를 써야 합니다'),
            (r'\bi are\b', 'I는 am을 써야 합니다')
        ]
        
        for pattern, description in error_patterns:
            if re.search(pattern, sentence_lower):
                issues.append({
                    'type': 'subject_verb_agreement',
                    'description': description,
                    'suggestion': '주어와 동사의 수를 맞춰보세요'
                })
        
        return issues
    
    def _check_tense_consistency(self, sentence, pos_tags):
        """시제 일관성 검사"""
        issues = []
        
        # 과거형과 현재형이 혼재된 경우 체크
        past_verbs = [word for word, pos in pos_tags if pos in ['VBD']]  # 과거형
        present_verbs = [word for word, pos in pos_tags if pos in ['VBZ', 'VBP']]  # 현재형
        
        if len(past_verbs) > 0 and len(present_verbs) > 0:
            issues.append({
                'type': 'tense_consistency',
                'description': '한 문장에서 과거형과 현재형이 혼재되어 있습니다',
                'suggestion': '문장 전체의 시제를 일치시켜보세요'
            })
        
        return issues
    
    def _check_article_usage(self, sentence, pos_tags):
        """관사 사용 검사"""
        issues = []
        
        # 관사가 빠진 패턴 (간단한 체크)
        sentence_lower = sentence.lower()
        
        # 자주 실수하는 패턴
        if re.search(r'\b(go to school|at home|in bed)\b', sentence_lower):
            # 이런 경우는 관사가 없는 게 맞음
            pass
        elif re.search(r'\b[a-z]+ (cat|dog|book|house|car)\b', sentence_lower):
            if not re.search(r'\b(a|an|the) (cat|dog|book|house|car)\b', sentence_lower):
                issues.append({
                    'type': 'article_usage',
                    'description': '셀 수 있는 명사 앞에는 관사가 필요할 수 있습니다',
                    'suggestion': 'a/an/the 중 적절한 관사를 추가해보세요'
                })
        
        return issues
    
    def _check_preposition_patterns(self, sentence, pos_tags):
        """전치사 사용 검사"""
        issues = []
        
        sentence_lower = sentence.lower()
        
        # 자주 틀리는 전치사 패턴
        error_patterns = [
            (r'\bin the morning\b.*\bin the afternoon\b', '시간 전치사 사용을 확인해보세요'),
            (r'\bgo to home\b', 'go home이 맞습니다 (to 불필요)'),
            (r'\blisten music\b', 'listen to music이 맞습니다'),
        ]
        
        for pattern, description in error_patterns:
            if re.search(pattern, sentence_lower):
                issues.append({
                    'type': 'preposition_usage',
                    'description': description,
                    'suggestion': '전치사 사용 규칙을 확인해보세요'
                })
        
        return issues
    
    def _check_sentence_structure(self, sentence, pos_tags):
        """문장 구조 검사"""
        issues = []
        
        # 문장이 너무 길거나 짧은 경우
        word_count = len([word for word, pos in pos_tags if pos not in ['.',  ',', ':', ';']])
        
        if word_count > 25:
            issues.append({
                'type': 'sentence_length',
                'description': '문장이 너무 길어 읽기 어려울 수 있습니다',
                'suggestion': '문장을 나누어보세요'
            })
        elif word_count < 3:
            issues.append({
                'type': 'sentence_length', 
                'description': '문장이 너무 짧습니다',
                'suggestion': '좀 더 자세한 설명을 추가해보세요'
            })
        
        # 주어나 동사가 없는 경우 (간단 체크)
        has_subject = any(pos in ['PRP', 'NN', 'NNS', 'NNP', 'NNPS'] for word, pos in pos_tags)
        has_verb = any(pos.startswith('VB') for word, pos in pos_tags)
        
        if not has_subject:
            issues.append({
                'type': 'sentence_structure',
                'description': '주어가 없는 것 같습니다',
                'suggestion': '문장의 주어를 명확히 해보세요'
            })
        
        if not has_verb:
            issues.append({
                'type': 'sentence_structure',
                'description': '동사가 없는 것 같습니다',
                'suggestion': '문장에 동사를 추가해보세요'
            })
        
        return issues
    
    def _identify_improvement_areas(self, error_count_by_type):
        """주요 개선 영역 식별"""
        if not error_count_by_type:
            return ['문법 사용이 우수합니다']
        
        # 가장 많이 발생한 오류 유형 순으로 정렬
        sorted_errors = sorted(error_count_by_type.items(), key=lambda x: x[1], reverse=True)
        
        improvement_map = {
            'subject_verb_agreement': '주어-동사 수일치 연습이 필요합니다',
            'tense_consistency': '시제 일관성 유지 연습이 필요합니다',
            'article_usage': '관사(a, an, the) 사용법 학습이 필요합니다',
            'preposition_usage': '전치사 사용법 연습이 필요합니다',
            'sentence_structure': '문장 구조 개선이 필요합니다',
            'sentence_length': '적절한 문장 길이 조절이 필요합니다'
        }
        
        areas = []
        for error_type, count in sorted_errors[:3]:  # 상위 3개만
            if error_type in improvement_map:
                areas.append(improvement_map[error_type])
        
        return areas if areas else ['전반적인 문법 검토가 필요합니다']

    def educational_sentiment_analysis_step4_emotions(self, text):
        """4단계: 다중 감성 분석 - 8가지 기본 감정 탐지"""
        if not text or pd.isna(text):
            return {}
        
        try:
            # 감정별 키워드 사전 (중학생 수준의 영어 단어)
            emotion_keywords = {
                '기쁨 (Joy)': {
                    'keywords': ['happy', 'joy', 'excited', 'wonderful', 'amazing', 'great', 'awesome', 
                                'fantastic', 'excellent', 'love', 'enjoy', 'smile', 'laugh', 'fun', 'cheerful'],
                    'emoji': '😊',
                    'color': '#FFD700'
                },
                '슬픔 (Sadness)': {
                    'keywords': ['sad', 'cry', 'tears', 'disappointed', 'sorry', 'hurt', 'pain', 
                                'lonely', 'depressed', 'upset', 'broken', 'miss', 'lost', 'tragic'],
                    'emoji': '😢',
                    'color': '#4169E1'
                },
                '분노 (Anger)': {
                    'keywords': ['angry', 'mad', 'hate', 'furious', 'annoyed', 'frustrated', 'rage', 
                                'irritated', 'upset', 'disgusted', 'terrible', 'awful', 'stupid', 'worst'],
                    'emoji': '😠',
                    'color': '#DC143C'
                },
                '두려움 (Fear)': {
                    'keywords': ['scared', 'afraid', 'fear', 'worried', 'nervous', 'anxious', 'panic', 
                                'terrified', 'frightened', 'concerned', 'stress', 'dangerous', 'risky'],
                    'emoji': '😨',
                    'color': '#9932CC'
                },
                '놀람 (Surprise)': {
                    'keywords': ['surprised', 'shocked', 'amazed', 'incredible', 'unexpected', 'sudden', 
                                'wow', 'unbelievable', 'astonishing', 'remarkable', 'stunning'],
                    'emoji': '😲',
                    'color': '#FF69B4'
                },
                '혐오 (Disgust)': {
                    'keywords': ['disgusting', 'gross', 'yuck', 'nasty', 'horrible', 'disgusted', 
                                'revolting', 'sickening', 'repulsive', 'unpleasant'],
                    'emoji': '🤢',
                    'color': '#8B4513'
                },
                '신뢰 (Trust)': {
                    'keywords': ['trust', 'believe', 'reliable', 'honest', 'faithful', 'loyal', 
                                'dependable', 'confident', 'secure', 'certain', 'sure', 'respect'],
                    'emoji': '🤝',
                    'color': '#228B22'
                },
                '기대 (Anticipation)': {
                    'keywords': ['hope', 'expect', 'anticipate', 'excited', 'eager', 'looking forward', 
                                'ready', 'prepared', 'future', 'plan', 'dream', 'wish', 'goal'],
                    'emoji': '🌟',
                    'color': '#FF8C00'
                }
            }
            
            # 텍스트 정제
            cleaned_text = self.extract_essay_content(text)
            if not cleaned_text:
                return {}
            
            # 단어 단위로 분리하고 소문자 변환
            import re
            words = re.findall(r'\b[a-zA-Z]+\b', cleaned_text.lower())
            
            # 각 감정별 점수 계산
            emotion_scores = {}
            emotion_details = {}
            total_emotional_words = 0
            
            for emotion, data in emotion_keywords.items():
                keywords = data['keywords']
                found_words = [word for word in words if word in keywords]
                score = len(found_words)
                
                emotion_scores[emotion] = score
                emotion_details[emotion] = {
                    'score': score,
                    'found_words': found_words[:10],  # 최대 10개까지만 저장
                    'emoji': data['emoji'],
                    'color': data['color']
                }
                total_emotional_words += score
            
            # 주도적 감정 찾기 (점수가 가장 높은 감정)
            if total_emotional_words > 0:
                dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
                dominant_name = dominant_emotion[0]
                dominant_score = dominant_emotion[1]
                
                # 감정 강도 계산 (전체 단어 대비 감정 단어 비율)
                emotion_intensity = (total_emotional_words / len(words)) * 100 if words else 0
                
                # 감정 다양성 계산 (0이 아닌 감정 종류 수)
                emotion_variety = len([score for score in emotion_scores.values() if score > 0])
                
            else:
                dominant_name = "중립 (Neutral)"
                dominant_score = 0
                emotion_intensity = 0
                emotion_variety = 0
                emotion_details["중립 (Neutral)"] = {
                    'score': 0,
                    'found_words': [],
                    'emoji': '😐',
                    'color': '#808080'
                }
            
            # 문장별 감정 분석 (처음 3문장)
            sentences = [s.strip() for s in cleaned_text.split('.') if s.strip()]
            sentence_emotions = []
            
            for i, sentence in enumerate(sentences[:3]):
                sentence_words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
                sentence_emotion_scores = {}
                
                for emotion, data in emotion_keywords.items():
                    keywords = data['keywords']
                    found = [word for word in sentence_words if word in keywords]
                    sentence_emotion_scores[emotion] = len(found)
                
                if any(score > 0 for score in sentence_emotion_scores.values()):
                    sentence_dominant = max(sentence_emotion_scores.items(), key=lambda x: x[1])
                    sentence_emotions.append({
                        'sentence': sentence[:80] + "..." if len(sentence) > 80 else sentence,
                        'emotion': sentence_dominant[0],
                        'score': sentence_dominant[1],
                        'emoji': emotion_keywords[sentence_dominant[0]]['emoji']
                    })
                else:
                    sentence_emotions.append({
                        'sentence': sentence[:80] + "..." if len(sentence) > 80 else sentence,
                        'emotion': '중립 (Neutral)',
                        'score': 0,
                        'emoji': '😐'
                    })
            
            return {
                'method': '다중 감성 분석 (8가지 기본 감정)',
                'dominant_emotion': dominant_name,
                'dominant_score': dominant_score,
                'dominant_emoji': emotion_details.get(dominant_name, {}).get('emoji', '😐'),
                'emotion_intensity': round(emotion_intensity, 2),
                'emotion_variety': emotion_variety,
                'total_emotional_words': total_emotional_words,
                'total_words': len(words),
                'emotion_scores': emotion_scores,
                'emotion_details': emotion_details,
                'sentence_emotions': sentence_emotions,
                'interpretation': self._interpret_multi_emotions(dominant_name, emotion_intensity, emotion_variety)
            }
            
        except Exception as e:
            return {
                'error': f'다중 감성 분석 중 오류: {str(e)}',
                'dominant_emotion': '분석 불가',
                'dominant_emoji': '❓'
            }
    
    def _interpret_multi_emotions(self, dominant_emotion, intensity, variety):
        """다중 감성 분석 결과 해석"""
        interpretations = []
        
        # 주도적 감정에 따른 해석
        emotion_meanings = {
            '기쁨 (Joy)': "글에 긍정적이고 밝은 에너지가 담겨 있어 읽는 이에게 좋은 인상을 줍니다.",
            '슬픔 (Sadness)': "감정적 깊이가 있는 글로, 독자의 공감을 불러일으킬 수 있습니다.",
            '분노 (Anger)': "강한 의견이나 불만이 표현되어 있어 주장이 명확합니다.",
            '두려움 (Fear)': "우려나 걱정이 드러나 있어 신중한 사고 과정을 보여줍니다.",
            '놀람 (Surprise)': "새로운 발견이나 깨달음이 담긴 흥미로운 내용입니다.",
            '혐오 (Disgust)': "강한 반감이나 비판 의식이 나타나 있습니다.",
            '신뢰 (Trust)': "믿음과 확신이 담긴 신뢰할 만한 내용입니다.",
            '기대 (Anticipation)': "미래에 대한 희망과 계획이 잘 드러나 있습니다.",
            '중립 (Neutral)': "객관적이고 중립적인 톤으로 작성되었습니다."
        }
        
        if dominant_emotion in emotion_meanings:
            interpretations.append(emotion_meanings[dominant_emotion])
        
        # 감정 강도에 따른 해석
        if intensity > 15:
            interpretations.append("매우 강한 감정이 표현되어 있어 독자에게 강한 인상을 줄 것입니다.")
        elif intensity > 8:
            interpretations.append("적절한 수준의 감정 표현으로 균형잡힌 글입니다.")
        elif intensity > 3:
            interpretations.append("감정 표현이 절제되어 있어 차분한 느낌을 줍니다.")
        else:
            interpretations.append("감정 표현을 조금 더 풍부하게 하면 글이 더욱 생동감 있어집니다.")
        
        # 감정 다양성에 따른 해석
        if variety >= 4:
            interpretations.append("다양한 감정이 어우러져 풍부한 감정적 표현을 보여줍니다.")
        elif variety >= 2:
            interpretations.append("여러 감정이 조화롭게 표현되어 있습니다.")
        else:
            interpretations.append("감정 표현의 다양성을 높이면 더 흥미로운 글이 될 것입니다.")
        
        return interpretations