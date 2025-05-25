import sys

# 设置标准输出为UTF-8编码
sys.stdout.reconfigure(encoding='utf-8')

import os
import json
from py2neo import Graph, Node


class MedicalGraphFromJson:
    def __init__(self):
        cur_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.data_path = os.path.join(cur_dir, 'resources/data/kg_data')
        self.g = Graph("bolt://localhost:7687", auth=("neo4j", "tczslw278"))
        self.rel_file = 'relations.json'
        self.node_file = 'entities.json'

    def build_graph(self):
        res = self.build_nodes()
        if res == -1:
            print('no nodes file, can not create relations')
            return
        self.build_rels()

    def build_nodes(self):
        node_file = os.path.join(self.data_path, self.node_file)
        if not os.path.exists(node_file):
            return -1
        with open(node_file, "r", encoding="utf-8") as f:
            nodes = json.load(f)
        for node in nodes:
            self.create_node(node)
        return 0

    def create_node(self, node):
        label = node['label']
        node1 = node['name']
        # 如果是 Person 类型
        if label == 'Person':
            n = Node(
                label,
                name=node1.get('chinese_name', ''),
                japanese_name=node1.get('japanese_name', ''),
                english_name=node1.get('english_name', ''),
                gender=node1.get('gender', '')
            )
        # 如果是 Pokémon 类型，处理宝可梦的属性
        elif label == 'Pokémon':
            n = Node(
                label,
                name=node1.get('chinese_name', ''),
                japanese_name=node1.get('japanese_name', ''),
                english_name=node1.get('english_name', ''),
                ability=node1.get('ability', ''),
                hidden_ability=node1.get('hidden_ability', ''),
                height=node1.get('height', ''),
                weight=node1.get('weight', ''),
                evolution_level=node1.get('evolution_level', ''),
                attr_ability=json.dumps(node1.get('attr_ability', {}), ensure_ascii=False)  # 将字典转换为字符串存储
            )
        # 默认处理方式
        else:
            n = Node(label, name=node1)

        # 创建节点
        self.g.create(n)

    def build_rels(self):
        rel_file = os.path.join(self.data_path, self.rel_file)

        if not os.path.exists(rel_file):
            print(self.rel_file, 'not exist, skip')
            return
        rel_file = os.path.join(self.data_path, self.rel_file)
        with open(rel_file, "r", encoding="utf-8") as f:
            relations = json.load(f)

        for rel in relations:
            self.create_rel(rel)

    def create_rel(self, rels_set):
        cnt = 0
        start_entity_type = rels_set['start_entity_type']
        end_entity_type = rels_set['end_entity_type']
        rel_type = rels_set['rel_type']
        rel_name = rels_set['rel_name']
        rels = rels_set['rels']

        for rel in rels:
            p = rel['start_entity_name']
            q = rel['end_entity_name']
            query = "MATCH (p:%s), (q:%s) WHERE p.name='%s' AND q.name='%s' CREATE (p)-[rel:%s{name:'%s'}]->(q)" % (
                start_entity_type, end_entity_type, p, q, rel_type, rel_name)

            try:
                self.g.run(query)
                cnt += 1
                print(f"{rel_type} {cnt}/{len(rels)}")
            except Exception as e:
                print(e)
        return


if __name__ == '__main__':
    handler = MedicalGraphFromJson()
    handler.build_graph()
