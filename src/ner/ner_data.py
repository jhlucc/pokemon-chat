import os
import random
import re

# 自动机 用于字符串匹配
import ahocorasick
from tqdm import tqdm
from configs.settings import *


# 制作训练集
class Build_Ner_data():
    def __init__(self):
        self.idx2type = idx2type = ["identity", "person", "Pokémon", "Region", "Town"]
        self.type2idx = type2idx = {"identity": 0, "person": 1, "Pokémon": 2, "Region": 3, "Town": 4}
        self.max_len = 30
        self.p = ['，', '。', '！', '；', '：', ',', '.', '?', '!', ';']
        self.ahos = [ahocorasick.Automaton() for i in range(len(idx2type))]

        for type in idx2type:
            with open(os.path.join(BASE_DIR, 'resources/data/entity_data/', f'{type}.txt'), encoding='utf-8') as f:
                all_en = f.read().split('\n')
            for en in all_en:
                if len(en) >= 2:
                    self.ahos[type2idx[type]].add_word(en, en)
        for i in range(len(self.ahos)):
            self.ahos[i].make_automaton()

    def split_text(self, text):
        """
        将长文本随机分割为短文本

        :param arg1: 长文本
        :return: 返回一个list,代表分割后的短文本
        :rtype: list
        """
        text = text.replace('\n', ',')
        pattern = r'([，。！；：,.?!;])(?=.)|[？,]'

        sentences = []

        for s in re.split(pattern, text):
            if s and len(s) > 0:
                sentences.append(s)

        sentences_text = [x for x in sentences if x not in self.p]
        sentences_Punctuation = [x for x in sentences[1::2] if x in self.p]
        split_text = []
        now_text = ''

        # 随机长度,有15%的概率生成短文本 10%的概率生成长文本
        for i in range(len(sentences_text)):
            if (len(now_text) > self.max_len and random.random() < 0.9 or random.random() < 0.15) and len(now_text) > 0:
                split_text.append(now_text)
                now_text = sentences_text[i]
                if i < len(sentences_Punctuation):
                    now_text += sentences_Punctuation[i]
            else:
                now_text += sentences_text[i]
                if i < len(sentences_Punctuation):
                    now_text += sentences_Punctuation[i]
        if len(now_text) > 0:
            split_text.append(now_text)

        # 随机选取30%的数据,把末尾标点改为。
        for i in range(len(split_text)):
            if random.random() < 0.3:
                if (split_text[i][-1] in self.p):
                    split_text[i] = split_text[i][:-1] + '。'
                else:
                    split_text[i] = split_text[i] + '。'
        return split_text

    def make_text_label(self, text):
        """
        通过ahocorasick类对文本进行识别，创造出文本的ner标签

        :param arg1: 文本
        :return: 返回一个list,代表标签
        :rtype: list
        """
        label = ['O'] * len(text)
        flag = 0
        mp = {}
        for type in self.idx2type:
            li = list(self.ahos[self.type2idx[type]].iter(text))
            if len(li) == 0:
                continue
            li = sorted(li, key=lambda x: len(x[1]), reverse=True)
            for en in li:
                ed, name = en
                st = ed - len(name) + 1
                if st in mp or ed in mp:
                    continue
                label[st:ed + 1] = ['B-' + type] + ['I-' + type] * (ed - st)
                flag = flag + 1
                for i in range(st, ed + 1):
                    mp[i] = 1
        return label, flag


# 将文本和对应的标签写入ner_data_aug.txt
def build_file(all_text, all_label):
    with open(os.path.join(BASE_DIR, 'resources/data/ner_data/', 'ner_data_aug.txt'), "w", encoding="utf-8") as f:
        for text, label in zip(all_text, all_label):
            for t, l in zip(text, label):
                f.write(f'{t} {l}\n')
            f.write('\n')


def load_book_text(file_path):
    """ 读取书籍文本并按段落分割 """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 以空行分割段落
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs


if __name__ == "__main__":
    book_path = os.path.join(BASE_DIR, 'resources/data/raw_data/', 'book.txt')

    paragraphs = load_book_text(book_path)

    build_ner_data = Build_Ner_data()

    all_text, all_label = [], []

    for paragraph in tqdm(paragraphs):
        # 长段落需要拆分为句子
        sentences = build_ner_data.split_text(paragraph)

        for sentence in sentences:
            if len(sentence) == 0:
                continue

            label, flag = build_ner_data.make_text_label(sentence)
            if flag >= 1:
                assert len(sentence) == len(label)
                all_text.append(sentence)
                all_label.append(label)

    build_file(all_text, all_label)
