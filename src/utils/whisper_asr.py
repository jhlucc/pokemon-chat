# src/utils/whisper_asr.py
import requests
import json
import re

class WhisperClient:
    def __init__(self, url="http://localhost:9000/asr?output=json"):
        self.url = url

    def transcribe(self, byte_array) -> str:
        files = {'audio_file': ('audio.wav', byte_array, 'audio/wav')}
        try:
            response = requests.post(self.url, files=files)
            response.raise_for_status()
            result = response.json()
            # print("[Whisper 返回结果]:", result)

            def clean_text(text: str) -> str:
                # ✅ 去除换行和多余空格，可扩展标点清洗等
                return re.sub(r'\s+', ' ', text).strip()

            if isinstance(result, dict) and "text" in result:
                return clean_text(result["text"])
            if isinstance(result, list) and "text" in result[0]:
                return clean_text(result[0]["text"])

            return ""
        except Exception as e:
            print("WhisperClient error:", e)
            return ""
