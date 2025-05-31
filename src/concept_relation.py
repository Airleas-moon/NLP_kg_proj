import spacy
from spacy.matcher import Matcher
import json
from pathlib import Path
import requests
from collections import defaultdict
from tqdm import tqdm
import os

# 确保输出目录存在
os.makedirs("output", exist_ok=True)

# 加载模型
nlp = spacy.load("en_core_web_sm")

# 知识库API配置
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

class WikidataAPI:
    @staticmethod
    def get_entity_id(text, entity_type):
        """通过文本获取Wikidata实体ID"""
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "type": "item" if entity_type != "PER" else "item",
            "search": text
        }
        try:
            response = requests.get(WIKIDATA_API, params=params, timeout=10)
            results = response.json().get("search", [])
            return results[0]["id"] if results else None
        except Exception as e:
            print(f"Error fetching entity ID for {text}: {str(e)}")
            return None

    @staticmethod
    def get_first_relation_between_entities(entity1_id, entity2_id):
        """只获取两个实体间的第一个关系（加速版）"""
        query = """
        SELECT ?relation ?relationLabel WHERE {
          wd:%s ?relation wd:%s.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 1
        """ % (entity1_id, entity2_id)
        url = "https://query.wikidata.org/sparql"
        try:
            response = requests.get(url, params={"format": "json", "query": query}, timeout=10)
            bindings = response.json().get("results", {}).get("bindings", [])
            if bindings:
                item = bindings[0]
                relation_uri = item.get("relation", {}).get("value", "")
                relation_id = relation_uri.split("/")[-1]
                relation_label = item.get("relationLabel", {}).get("value", relation_id)
                return (relation_id, relation_label)
            return None
        except Exception as e:
            print(f"Error fetching first relation between {entity1_id} and {entity2_id}: {str(e)}")
            return None

class DistantSupervisionRelationExtractor:
    def __init__(self, nlp):
        self.nlp = nlp
        self.matcher = Matcher(nlp.vocab)
        self.wikidata = WikidataAPI()
        self.cache = defaultdict(dict)  # 缓存实体和关系查询结果
      
    def get_entity_mapping(self, text, entity_type):
        """获取或缓存实体映射"""
        if text not in self.cache["entities"]:
            self.cache["entities"][text] = self.wikidata.get_entity_id(text, entity_type)
        return self.cache["entities"][text]
  
    def get_cached_relations(self, entity1_id, entity2_id):
        """获取或缓存实体间关系"""
        cache_key = f"{entity1_id}_{entity2_id}"
        if cache_key not in self.cache["relations"]:
            self.cache["relations"][cache_key] = self.wikidata.get_relations_between_entities(entity1_id, entity2_id)
        return self.cache["relations"][cache_key]
  
    def extract_relations(self, doc):
        """基于远程监督的关系抽取（只获取第一个关系）"""
        relations = []
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                subj, subj_type = entities[i]
                obj, obj_type = entities[j]
                
                subj_id = self.get_entity_mapping(subj, subj_type)
                obj_id = self.get_entity_mapping(obj, obj_type)
                
                if subj_id and obj_id:
                    # 只查询第一个关系
                    result = self.wikidata.get_first_relation_between_entities(subj_id, obj_id)
                    if result:
                        rel_id, rel_label = result
                        relations.append({
                            "subject": subj,
                            "subject_type": subj_type,
                            "relation": rel_label,
                            "object": obj,
                            "object_type": obj_type,
                        })
        return relations

def process_conll_file(file_path):
    """处理CoNLL格式文件"""
    sentences = []
    current_sentence = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                continue
            if not line:
                if current_sentence:
                    sentences.append(" ".join(current_sentence))
                    current_sentence = []
                continue
            word = line.split()[0]
            current_sentence.append(word)
    if current_sentence:
        sentences.append(" ".join(current_sentence))
    return sentences

def main():
    # 初始化远程监督抽取器
    ds_extractor = DistantSupervisionRelationExtractor(nlp)
  
    # 处理训练数据
    sentences = process_conll_file("../data/raw/train.txt")[:5]  # 测试少量数据
    results = []
    
    # 添加进度条
    for sentence in tqdm(sentences, desc="Processing sentences"):
        doc = nlp(sentence)
        relations = ds_extractor.extract_relations(doc)
        
        if relations:
            results.append({
                "text": sentence,
                "entities": [{"text": ent.text, "type": ent.label_} for ent in doc.ents],
                "relations": relations
            })
    
    # 保存结果到JSON文件
    output_path = Path("output/relations_with_entities.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print(f"Processed {len(sentences)} sentences, found {sum(len(item['relations']) for item in results)} relations")

if __name__ == "__main__":
    main()
