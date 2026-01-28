#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import re

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None

from src.plugins.parser.utils import get_text
from src.utils.logger import get_logger

_log = get_logger(__name__)

# encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
_encoder = None
_encoder_ready = False


def _get_encoder():
    global _encoder, _encoder_ready
    if _encoder_ready:
        return _encoder
    if tiktoken is None:
        _encoder = None
    else:
        try:
            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:
            _encoder = None
            _log.warning("tiktoken encoding unavailable; falling back to char length. (%s)", exc)
    _encoder_ready = True
    return _encoder


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    try:
        encoder = _get_encoder()
        if encoder is None:
            return len(string)
        return len(encoder.encode(string))
    except Exception:
        return len(string)


class RAGFlowTxtParser:
    def __call__(self, fnm, binary=None, chunk_token_num=128, delimiter="\n!?;。；！？"):
        txt = get_text(fnm, binary)
        return self.parser_txt(txt, chunk_token_num, delimiter)

    @classmethod
    def parser_txt(cls, txt, chunk_token_num=128, delimiter="\n!?;。；！？"):
        if not isinstance(txt, str):
            raise TypeError("txt type should be str!")
        cks = [""]
        tk_nums = [0]
        delimiter = delimiter.encode("utf-8").decode("unicode_escape").encode("latin1").decode("utf-8")

        def add_chunk(t):
            nonlocal cks, tk_nums, delimiter
            tnum = num_tokens_from_string(t)
            if tk_nums[-1] > chunk_token_num:
                cks.append(t)
                tk_nums.append(tnum)
            else:
                cks[-1] += t
                tk_nums[-1] += tnum

        dels = []
        s = 0
        for m in re.finditer(r"`([^`]+)`", delimiter, re.I):
            f, t = m.span()
            dels.append(m.group(1))
            dels.extend(list(delimiter[s:f]))
            s = t
        if s < len(delimiter):
            dels.extend(list(delimiter[s:]))
        dels = [re.escape(d) for d in dels if d]
        dels = [d for d in dels if d]
        dels = "|".join(dels)
        secs = re.split(rf"({dels})", txt)
        for sec in secs:
            if re.match(f"^{dels}$", sec):
                continue
            add_chunk(sec)

        return [[c, ""] for c in cks]


if __name__ == "__main__":
    import sys

    # 传入txt路径
    file_path = "/data/Langagent/deepdoc/data/identity.txt"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    parser = RAGFlowTxtParser()
    # 解析txt文件
    chunks = parser(file_path, chunk_token_num=128)
    print(f"📄 共切分出 {len(chunks)} 个段落：")
    for i, (text, _) in enumerate(chunks):
        print(f"\n=== Chunk {i + 1} ===")
        print(f"内容（前60字）: {text[:60]}...")
        print(f"Token 数量: {num_tokens_from_string(text)}")
