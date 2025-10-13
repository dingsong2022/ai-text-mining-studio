import redis
import json
import os
import streamlit as st
from typing import Any, Optional

class RedisCache:
    """Redis 캐시 관리 클래스"""

    def __init__(self):
        """Redis 연결 초기화"""
        self.client = None
        self._connect()

    def _connect(self):
        """Redis 서버에 연결"""
        try:
            # Streamlit Cloud에서는 secrets 사용
            redis_url = None
            try:
                redis_url = st.secrets["REDIS_URL"]
                print(f"Using Redis URL from Streamlit secrets")
            except:
                # 로컬에서는 환경 변수 사용
                redis_url = os.getenv("REDIS_URL")
                if redis_url:
                    print(f"Using Redis URL from environment variable")

            if not redis_url:
                print("Warning: REDIS_URL not found. Cache disabled.")
                return

            # Redis 연결
            self.client = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )

            # 연결 테스트
            self.client.ping()
            print("Redis cache connected successfully!")

        except Exception as e:
            print(f"Redis connection failed: {e}")
            self.client = None

    def get(self, key: str) -> Optional[Any]:
        """캐시에서 데이터 가져오기

        Args:
            key: 캐시 키

        Returns:
            캐시된 데이터 또는 None
        """
        if not self.client:
            return None

        try:
            data = self.client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            print(f"Redis GET error for key '{key}': {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = 300):
        """캐시에 데이터 저장

        Args:
            key: 캐시 키
            value: 저장할 데이터 (JSON 직렬화 가능해야 함)
            ttl: 캐시 유지 시간 (초, 기본 5분)
        """
        if not self.client:
            return

        try:
            json_data = json.dumps(value, ensure_ascii=False, default=str)
            self.client.setex(key, ttl, json_data)
            print(f"Redis SET: key='{key}', ttl={ttl}s")
        except Exception as e:
            print(f"Redis SET error for key '{key}': {e}")

    def delete(self, key: str):
        """캐시 삭제

        Args:
            key: 삭제할 캐시 키
        """
        if not self.client:
            return

        try:
            self.client.delete(key)
            print(f"Redis DELETE: key='{key}'")
        except Exception as e:
            print(f"Redis DELETE error for key '{key}': {e}")

    def clear_all(self):
        """모든 캐시 삭제 (주의: 전체 데이터베이스 초기화)"""
        if not self.client:
            return

        try:
            self.client.flushdb()
            print("Redis: All cache cleared")
        except Exception as e:
            print(f"Redis CLEAR error: {e}")

    def get_keys(self, pattern: str = "*") -> list:
        """패턴과 일치하는 모든 키 반환

        Args:
            pattern: 검색 패턴 (기본: 모든 키)

        Returns:
            키 리스트
        """
        if not self.client:
            return []

        try:
            return self.client.keys(pattern)
        except Exception as e:
            print(f"Redis KEYS error: {e}")
            return []

    def is_connected(self) -> bool:
        """Redis 연결 상태 확인

        Returns:
            연결 여부
        """
        if not self.client:
            return False

        try:
            self.client.ping()
            return True
        except:
            return False
