# src/utils/funasr_client.py
"""
FunASR (阿里达摩院) 语音识别客户端

支持通过 WebSocket 连接 FunASR 服务进行语音转文字。
"""
import json
import re
from websocket import create_connection


class FunASRClient:
    """FunASR WebSocket 客户端"""
    
    def __init__(self, url: str = "ws://localhost:10095"):
        """
        初始化 FunASR 客户端
        
        Args:
            url: FunASR WebSocket 服务地址，默认 ws://localhost:10095
        """
        self.url = url

    def transcribe(self, audio_bytes: bytes) -> str:
        """
        将音频数据转换为文字
        
        Args:
            audio_bytes: WAV 格式的音频字节数据 (16kHz, 单声道)
        
        Returns:
            识别出的文字内容
        """
        try:
            # 建立 WebSocket 连接
            ws = create_connection(self.url)
            
            # 发送开始信号
            start_msg = json.dumps({
                "mode": "offline",
                "chunk_size": [5, 10, 5],
                "wav_name": "audio",
                "is_speaking": True,
                "wav_format": "wav",
                "audio_fs": 16000,
            })
            ws.send(start_msg)
            
            # 发送音频数据
            ws.send_binary(audio_bytes)
            
            # 发送结束信号
            end_msg = json.dumps({"is_speaking": False})
            ws.send(end_msg)
            
            # 接收识别结果
            result_text = ""
            while True:
                response = ws.recv()
                if not response:
                    break
                    
                result = json.loads(response)
                if "text" in result:
                    result_text = self._clean_text(result["text"])
                    
                # 检查是否结束
                if result.get("is_final", False) or result.get("mode") == "offline":
                    break
            
            ws.close()
            return result_text
            
        except Exception as e:
            print(f"FunASRClient error: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """清理识别结果中的多余空格和换行"""
        return re.sub(r'\s+', ' ', text).strip()
