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


import io
import re

from PIL import Image


def vision_llm_figure_describe_ch_prompt() -> str:
    prompt = """
你是一位视觉数据分析专家。请分析图像内容，并提供对图像中数据可视化信息的详细描述。请重点识别图表类型（如柱状图、饼图、折线图、表格、流程图等）、结构及图中包含的文本信息（如标题、标签、坐标轴等）。

任务要求如下：
1. 描述图像的整体结构，说明其属于哪种可视化类型（图表、图、表格或图示等）。
2. 识别图中的坐标轴、图例、标题、标签等元素，并提供具体文字内容（如果存在）。
3. 提取图像中的数据点信息（如柱状图的高度、折线图坐标、饼图比例、表格中的行列数据等）。
4. 分析并说明图中展示的趋势、对比或模式。
5. 捕捉图中的任何注释、说明文字或脚注，并说明其与图像内容的关联。
6. 仅描述图像中真实存在的信息，不要假设或编造任何缺失的元素（如未出现的坐标轴或图例等）。

输出格式（仅包含与图像内容相关的部分）：
- 图像类型: [类型]
- 标题: [标题内容，如有]
- 坐标轴 / 图例 / 标签: [具体信息，如有]
- 数据点: [提取到的数据内容]
- 趋势 / 洞察: [分析与解读]
- 注释 / 脚注: [文字内容及其相关性，如有]

请确保描述具有高准确性、清晰度与完整性，且仅包含图像中真实存在的信息。避免冗余描述或不确定性判断。
"""
    return prompt


def vision_llm_figure_describe_prompt() -> str:
    prompt = """
You are an expert visual data analyst. Analyze the image and provide a comprehensive description of its content. Focus on identifying the type of visual data representation (e.g., bar chart, pie chart, line graph, table, flowchart), its structure, and any text captions or labels included in the image.

Tasks:
1. Describe the overall structure of the visual representation. Specify if it is a chart, graph, table, or diagram.
2. Identify and extract any axes, legends, titles, or labels present in the image. Provide the exact text where available.
3. Extract the data points from the visual elements (e.g., bar heights, line graph coordinates, pie chart segments, table rows and columns).
4. Analyze and explain any trends, comparisons, or patterns shown in the data.
5. Capture any annotations, captions, or footnotes, and explain their relevance to the image.
6. Only include details that are explicitly present in the image. If an element (e.g., axis, legend, or caption) does not exist or is not visible, do not mention it.

Output format (include only sections relevant to the image content):
- Visual Type: [Type]
- Title: [Title text, if available]
- Axes / Legends / Labels: [Details, if available]
- Data Points: [Extracted data]
- Trends / Insights: [Analysis and interpretation]
- Captions / Annotations: [Text and relevance, if available]

Ensure high accuracy, clarity, and completeness in your analysis, and includes only the information present in the image. Avoid unnecessary statements about missing elements.
"""
    return prompt


def clean_markdown_block(text):
    text = re.sub(r'^\s*```markdown\s*\n?', '', text)
    text = re.sub(r'\n?\s*```\s*$', '', text)
    return text.strip()


def picture_vision_llm_chunk(binary, vision_model, prompt=None, callback=None):
    """
    A simple wrapper to process image to markdown texts via VLM.

    Returns:
        Simple markdown texts generated by VLM.
    """
    callback = callback or (lambda prog, msg: None)

    img = binary
    txt = ""

    try:
        img_binary = io.BytesIO()
        img.save(img_binary, format='JPEG')
        img_binary.seek(0)

        ans = clean_markdown_block(vision_model.describe_with_prompt(img_binary.read(), prompt))

        txt += "\n" + ans

        return txt

    except Exception as e:
        callback(-1, str(e))

    return ""


def vision_figure_parser_figure_data_wraper(figures_data_without_positions):
    return [(
        (figure_data[1], [figure_data[0]]),
        [(0, 0, 0, 0, 0)]
    ) for figure_data in figures_data_without_positions if isinstance(figure_data[1], Image.Image)]


class VisionFigureParser:
    def __init__(self, vision_model, figures_data, *args, **kwargs):
        self.vision_model = vision_model
        self._extract_figures_info(figures_data)
        assert len(self.figures) == len(self.descriptions)
        assert not self.positions or (len(self.figures) == len(self.positions))

    def _extract_figures_info(self, figures_data):
        self.figures = []
        self.descriptions = []
        self.positions = []

        for item in figures_data:
            # position
            if len(item) == 2 and isinstance(item[1], list) and len(item[1]) == 1 and isinstance(item[1][0],
                                                                                                 tuple) and len(
                item[1][0]) == 5:
                img_desc = item[0]
                assert len(img_desc) == 2 and isinstance(img_desc[0], Image.Image) and isinstance(img_desc[1],
                                                                                                  list), "Should be (figure, [description])"
                self.figures.append(img_desc[0])
                self.descriptions.append(img_desc[1])
                self.positions.append(item[1])
            else:
                assert len(item) == 2 and isinstance(item, tuple) and isinstance(item[1],
                                                                                 list), f"get {len(item)=}, {item=}"
                self.figures.append(item[0])
                self.descriptions.append(item[1])

    def _assemble(self):
        self.assembled = []
        self.has_positions = len(self.positions) != 0
        for i in range(len(self.figures)):
            figure = self.figures[i]
            desc = self.descriptions[i]
            pos = self.positions[i] if self.has_positions else None

            figure_desc = (figure, desc)

            if pos is not None:
                self.assembled.append((figure_desc, pos))
            else:
                self.assembled.append((figure_desc,))

        return self.assembled

    def __call__(self, **kwargs):
        callback = kwargs.get("callback", lambda prog, msg: None)

        for idx, img_binary in enumerate(self.figures or []):
            figure_num = idx  # 0-based

            txt = picture_vision_llm_chunk(
                binary=img_binary,
                vision_model=self.vision_model,
                prompt=vision_llm_figure_describe_prompt(),
                callback=callback,
            )

            if txt:
                self.descriptions[figure_num] = txt + "\n".join(self.descriptions[figure_num])

        self._assemble()

        return self.assembled
