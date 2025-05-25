import os
import pickle

import ahocorasick
import torch
from seqeval.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from configs.settings import *

# 模型训练
cache_model = CACHE_BERTA_MODEL


def get_data(path, max_len=None):
    all_text, all_tag = [], []
    with open(path, 'r', encoding='utf8') as f:
        all_data = f.read().split('\n')

    sen, tag = [], []
    for data in all_data:
        data = data.split(' ')
        if (len(data) != 2):
            if len(sen) > 2:
                all_text.append(sen)
                all_tag.append(tag)
            sen, tag = [], []
            continue
        te, ta = data
        sen.append(te)
        tag.append(ta)
    if max_len is not None:
        return all_text[:max_len], all_tag[:max_len]
    return all_text, all_tag


class rule_find:
    def __init__(self):
        self.idx2type = idx2type = ["identity", "person", "Pokémon", "Region", "Town"]
        self.type2idx = type2idx = {"identity": 0, "person": 1, "Pokémon": 2, "Region": 3, "Town": 4}
        self.ahos = [ahocorasick.Automaton() for i in range(len(self.type2idx))]

        for type in idx2type:
            with open(os.path.join(BASE_DIR, 'resources/data/entity_data/', f'{type}.txt'), encoding='utf-8') as f:
                all_en = f.read().split('\n')
            for en in all_en:
                en = en.split(' ')[0]
                if len(en) >= 1:
                    self.ahos[type2idx[type]].add_word(en, en)
        for i in range(len(self.ahos)):
            self.ahos[i].make_automaton()

    # sen -> (start,end,cls,word)
    def find(self, sen):
        rule_result = []
        mp = {}
        all_res = []
        all_ty = []
        for i in range(len(self.ahos)):
            now = list(self.ahos[i].iter(sen))
            all_res.extend(now)
            for j in range(len(now)):
                all_ty.append(self.idx2type[i])
        if len(all_res) != 0:
            combined = list(zip(all_res, all_ty))
            # 按实体长度从长到短排序
            combined.sort(key=lambda x: len(x[0][1]), reverse=True)

            # 清空 all_res 和 all_ty，重新填充排序后的结果
            all_res = [res for res, ty in combined]
            all_ty = [ty for res, ty in combined]

            # 遍历排序后的匹配结果
            for i, res in enumerate(all_res):
                # 计算实体的起始位置和结束位置
                be = res[0] - len(res[1]) + 1
                ed = res[0]
                # 如果起始位置或结束位置已被占用，跳过当前实体
                if be in mp or ed in mp:
                    continue
                # 将匹配到的实体信息添加到 rule_result
                rule_result.append((be, ed, all_ty[i], res[1]))
                # 记录当前实体占用的字符位置
                for t in range(be, ed + 1):
                    mp[t] = 1
        return rule_result


# 由模型输出的tag(B-identity)转换为(start,end,cls)
def find_entities(tag):
    result = []  # [(2,3,'Person'),(7,10,'Indentity')]
    label_len = len(tag)
    i = 0
    while (i < label_len):
        if (tag[i][0] == 'B'):
            type = tag[i].strip('B-')
            j = i + 1
            while (j < label_len and tag[j][0] == 'I'):
                j += 1
            result.append((i, j - 1, type))
            i = j
        else:
            i = i + 1
    return result


class tfidf_alignment:
    """
    以Pokemon这个实体为例，该实体列表中一共出现6个字符，每个实体对应一个6维向量
    ['皮卡丘','耿鬼','雷丘']
    --->
    [[x,x,x,x,x,x],[x,x,x,x,x,x],[x,x,x,x,x,x]]
    """

    def __init__(self):
        eneities_path = os.path.join(ENTITY_DATA, '')
        files = os.listdir(eneities_path)
        # 排除 .py 文件
        files = [docu for docu in files if '.py' not in docu]
        self.tag_2_embs = {}
        self.tag_2_tfidf_model = {}
        self.tag_2_entity = {}
        for ty in files:
            with open(os.path.join(eneities_path, ty), 'r', encoding='utf-8') as f:
                entities = f.read().split('\n')
                # 过滤长度过长或过短的实体
                entities = [
                    ent for ent in entities
                    if 1 <= len(ent.split(' ')[0]) <= 15
                ]
                # 只取每行的第一个词
                en_name = [ent.split(' ')[0] for ent in entities]
                # 去掉文件名后缀 .txt
                ty = ty.strip('.txt')
                # 记录实体列表
                self.tag_2_entity[ty] = en_name
                # 初始化 TF-IDF，
                tfidf_model = TfidfVectorizer(analyzer="char")
                embs = tfidf_model.fit_transform(en_name)  # 稀疏矩阵
                self.tag_2_tfidf_model[ty] = tfidf_model
                self.tag_2_embs[ty] = embs  # 保持稀疏格式

    def align(self, ent_list):
        """
        ent_list 为 [(start_idx, end_idx, cls, ent), ...]
        返回一个 dict：{cls: best_matched_entity_name}
        """
        new_result = {}
        for s, e, cls, ent in ent_list:
            # 若该类型不在词典中，则跳过
            if cls not in self.tag_2_tfidf_model:
                continue

            # 对当前实体做 TF-IDF 编码
            ent_emb = self.tag_2_tfidf_model[cls].transform([ent])  # 稀疏矩阵
            # 和已知实体向量 self.tag_2_embs[cls] 做相似度
            sim_score = cosine_similarity(ent_emb, self.tag_2_embs[cls])
            max_idx = sim_score[0].argmax()
            max_score = sim_score[0, max_idx]

            # 如果相似度大于阈值 0.5，就认为匹配
            if max_score >= 0.7:
                if cls not in new_result:
                    new_result[cls] = []
                new_result[cls].append(self.tag_2_entity[cls][max_idx])

        # 去重
        for cls in new_result:
            new_result[cls] = list(set(new_result[cls]))

        return new_result


class Nerdataset(Dataset):
    def __init__(self, all_text, all_label, tokenizer, max_len, tag2idx, is_dev=False):
        self.all_text = all_text
        self.all_label = all_label
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag2idx = tag2idx
        self.is_dev = is_dev

    def __getitem__(self, x):
        text, label = self.all_text[x], self.all_label[x]
        if self.is_dev:
            max_len = min(len(self.all_text[x]) + 2, 500)
        else:
            max_len = self.max_len
        text, label = text[:max_len - 2], label[:max_len - 2]

        x_len = len(text)
        assert len(text) == len(label)
        text_idx = self.tokenizer.encode(text, add_special_token=True)
        label_idx = [self.tag2idx['<PAD>']] + [self.tag2idx[i] for i in label] + [self.tag2idx['<PAD>']]

        text_idx += [0] * (max_len - len(text_idx))
        label_idx += [self.tag2idx['<PAD>']] * (max_len - len(label_idx))
        return torch.tensor(text_idx), torch.tensor(label_idx), x_len

    def __len__(self):
        return len(self.all_text)


def build_tag2idx(all_tag):
    tag2idx = {'<PAD>': 0}
    for sen in all_tag:
        for tag in sen:
            tag2idx[tag] = tag2idx.get(tag, len(tag2idx))
    return tag2idx


class Bert_Model(nn.Module):
    def __init__(self, model_name, hidden_size, tag_num, bi):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.gru = nn.RNN(input_size=768, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=bi)
        if bi:
            self.classifier = nn.Linear(hidden_size * 2, tag_num)
        else:
            self.classifier = nn.Linear(hidden_size, tag_num)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, x, label=None):
        bert_0, _ = self.bert(x, attention_mask=(x > 0), return_dict=False)
        gru_0, _ = self.gru(bert_0)
        pre = self.classifier(gru_0)
        if label is not None:
            loss = self.loss_fn(pre.reshape(-1, pre.shape[-1]), label.reshape(-1))
            return loss
        else:
            return torch.argmax(pre, dim=-1).squeeze(0)


def merge(model_result_word, rule_result):
    result = model_result_word + rule_result
    result = sorted(result, key=lambda x: len(x[-1]), reverse=True)
    check_result = []
    # 去除重叠的实体
    mp = {}
    for res in result:
        if res[0] in mp or res[1] in mp:
            continue
        check_result.append(res)
        for i in range(res[0], res[1] + 1):
            mp[i] = 1
    return check_result


def get_ner_result(model, tokenizer, sen, rule, tfidf_r, device, idx2tag):
    sen_to = tokenizer.encode(sen, add_special_tokens=True, return_tensors='pt').to(device)

    pre = model(sen_to).tolist()

    pre_tag = [idx2tag[i] for i in pre[1:-1]]
    model_result = find_entities(pre_tag)  # (start,end,cls)
    model_result_word = []  # [(start,end,cls,word), ...]
    for res in model_result:
        word = sen[res[0]:res[1] + 1]
        model_result_word.append((res[0], res[1], res[2], word))
    rule_result = rule.find(sen)  # [(start,end,cls,word), ...]

    merge_result = merge(model_result_word, rule_result)
    tfidf_result = tfidf_r.align(merge_result)

    # print('模型结果',model_result_word)
    # print('规则结果',rule_result)
    # print('整合结果', merge_result)
    # print('tfidf对齐结果', tfidf_result)
    return tfidf_result


if __name__ == "__main__":
    all_text, all_label = get_data(os.path.join(NER_DATA, 'ner_data_aug.txt'))
    train_text, dev_text, train_label, dev_label = train_test_split(all_text, all_label, test_size=0.02,
                                                                    random_state=42)

    # 加载太慢了，预处理一下
    if os.path.exists('../../resources/data/ner_data/tag2idx.npy'):
        with open('../../resources/data/ner_data/tag2idx.npy', 'rb') as f:
            tag2idx = pickle.load(f)
    else:
        tag2idx = build_tag2idx(all_label)
        with open('../../resources/data/ner_data/tag2idx.npy', 'wb') as f:
            pickle.dump(tag2idx, f)

    idx2tag = list(tag2idx)

    max_len = 50
    epoch = 1
    batch_size = 60
    hidden_size = 128
    bi = True
    model_name = MODEL_ROBERTA_PATH
    # pip install -U huggingface_hub   huggingface-cli download --resume-download hfl/chinese-roberta-wwm-ext --local-dir ./
    tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=MODEL_ROBERTA_PATH)
    lr = 1e-5
    is_train = False

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    train_dataset = Nerdataset(train_text, train_label, tokenizer, max_len, tag2idx)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dev_dataset = Nerdataset(dev_text, dev_label, tokenizer, max_len, tag2idx, is_dev=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False)

    model = Bert_Model(model_name, hidden_size, len(tag2idx), bi)

    # 尝试加载已存在的模型权重
    pt_path = os.path.join(cache_model, "best_roberta.pt")
    if os.path.exists(pt_path):
        print("加载已有模型")
        model.load_state_dict(torch.load(pt_path, map_location=device))
    else:
        is_train = True  # 如果没有找到模型，则需要进行训练

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bestf1 = -1
    if is_train:
        for e in range(epoch):
            loss_sum = 0
            ba = 0
            for x, y, batch_len in tqdm(train_dataloader):
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad()
                loss = model(x, y)
                loss.backward()

                opt.step()
                loss_sum += loss
                ba += 1
            all_pre = []
            all_label = []
            for x, y, batch_len in tqdm(dev_dataloader):
                assert len(x) == len(y)
                x = x.to(device)
                pre = model(x)
                pre = [idx2tag[i] for i in pre[1:batch_len + 1]]
                all_pre.append(pre)

                label = [idx2tag[i] for i in y[0][1:batch_len + 1]]
                all_label.append(label)
            f1 = f1_score(all_pre, all_label)
            if f1 > bestf1:
                bestf1 = f1
                print(f'e={e},loss={loss_sum / ba:.5f} f1={f1:.5f} ---------------------->best')
                torch.save(model.state_dict(), f'{cache_model}.pt')
            else:
                print(f'e={e},loss={loss_sum / ba:.5f} f1={f1:.5f}')

    rule = rule_find()
    tfidf_r = tfidf_alignment()

    while (True):
        sen = input('请输入:')
        print(get_ner_result(model, tokenizer, sen, rule, tfidf_r, device, idx2tag))
