import streamlit as st
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
import json

# Google Sheets 설정
SCOPES = ['https://www.googleapis.com/auth/spreadsheets',
          'https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'credentials.json'
SHEET_ID = '1_HkNcnWX_31GhJwDcT3a2D41BJvbF9Njmwi5d5T8pWQ'

class DataLoader:
    def __init__(self):
        self.sheet = self._get_google_sheets()
    
    @st.cache_resource
    def _get_google_sheets(_self):
        """Google Sheets 연결"""
        try:
            # Streamlit Cloud에서 secrets 사용
            try:
                credentials = Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"], scopes=SCOPES)
                print("Using Streamlit secrets for authentication")
            except Exception as e:
                print(f"Failed to use Streamlit secrets: {e}")
                # 로컬에서는 파일 사용
                credentials = Credentials.from_service_account_file(
                    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
                print("Using local credentials file for authentication")
            
            gc = gspread.authorize(credentials)
            sheet = gc.open_by_key(SHEET_ID)
            print(f"Successfully connected to Google Sheets: {SHEET_ID}")
            return sheet
        except Exception as e:
            error_msg = f"Google Sheets 연결 실패: {e}"
            print(error_msg)
            st.error(error_msg)
            return None
    
    def get_student_essays(self, username):
        """특정 학생의 모든 에세이 데이터 가져오기"""
        try:
            if not self.sheet:
                st.error("Google Sheets 연결이 되지 않았습니다.")
                return pd.DataFrame()
            
            # 논술데이터 시트에서 데이터 가져오기
            essays_sheet = self.sheet.worksheet("논술데이터")
            data = essays_sheet.get_all_records()
            
            # 해당 학생의 데이터만 필터링
            student_data = []
            for row in data:
                row_username = row.get('아이디') or row.get('username')
                if row_username == username:
                    # 실제 시트 구조에 맞게 매핑
                    converted_row = {
                        'username': row_username,
                        'topic_name': row.get('이름', ''),  # 주제명
                        'created_at': row.get('날짜', ''),  # 작성일
                        'topic_description': row.get('주제', ''),  # 주제 설명
                        'essay_text': row.get('논술문', ''),  # 실제 에세이 내용
                        'total_score': row.get('점수', 0),  # 점수
                        'feedback': row.get('피드백', '')  # 피드백
                    }
                    student_data.append(converted_row)
            
            if not student_data:
                st.warning(f"{username}의 에세이 데이터가 없습니다.")
                return pd.DataFrame()
            
            df = pd.DataFrame(student_data)
            
            # 데이터 타입 변환
            if 'total_score' in df.columns:
                # 점수에서 숫자만 추출
                df['total_score'] = df['total_score'].astype(str).str.extract('(\d+)').astype(float)
            
            # 날짜 컬럼 변환
            if 'created_at' in df.columns:
                df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            
            return df
            
        except Exception as e:
            st.error(f"데이터 로딩 오류: {e}")
            return pd.DataFrame()
    
    def get_combined_essay_text(self, username):
        """학생의 모든 에세이를 하나의 텍스트로 합치기"""
        try:
            essay_data = self.get_student_essays(username)
            
            if essay_data.empty:
                return ""
            
            # 모든 에세이 텍스트 합치기
            all_texts = []
            for _, row in essay_data.iterrows():
                essay_text = row.get('essay_text', '')
                if essay_text and not pd.isna(essay_text):
                    all_texts.append(str(essay_text))
            
            return " ".join(all_texts)
            
        except Exception as e:
            st.error(f"텍스트 합치기 오류: {e}")
            return ""
    
    def get_all_students_list(self):
        """전체 학생 목록 가져오기"""
        try:
            if not self.sheet:
                return []
            
            # 사용자정보 시트에서 학생 목록 가져오기
            users_sheet = self.sheet.worksheet("사용자정보")
            
            # 모든 데이터 가져오기
            all_values = users_sheet.get_all_values()
            
            if len(all_values) < 2:
                return []
            
            # 학생 ID만 추출 (첫 번째 컬럼이 아이디)
            students = []
            for row in all_values[1:]:  # 헤더 제외
                if row and len(row) > 0:  # 빈 행이 아닌 경우
                    username = row[0].strip()  # 첫 번째 컬럼 (아이디)
                    if username and username != 'teachertest1':
                        students.append(username)
            
            return sorted(students)
            
        except Exception as e:
            st.error(f"학생 목록 로딩 오류: {e}")
            return []
    
    def test_connection(self):
        """연결 테스트"""
        try:
            if not self.sheet:
                return False, "Google Sheets 연결 실패"
            
            # 시트에 접근해보기
            worksheets = self.sheet.worksheets()
            
            return True, f"연결 성공! 시트 수: {len(worksheets)}"
            
        except Exception as e:
            return False, f"연결 테스트 실패: {e}"