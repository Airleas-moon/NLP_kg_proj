import spacy
from spacy.matcher import Matcher
import json
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import defaultdict
from tqdm import tqdm
import os
from functools import lru_cache

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 确保不使用GPU
print("Running on CPU")

# 加载模型（使用更轻量级的sm模型而不是trf）
nlp = spacy.load("en_core_web_sm", 
                 exclude=["tagger", "parser", "attribute_ruler", "lemmatizer"])

# 知识库API配置
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

class WikidataSession:
    """自定义Wikidata会话管理，包含自动重试和持久化连接"""
    def __init__(self):
        self.session = requests.Session()
        # 配置自动重试策略
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[408, 429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
    
    def get(self, url, params=None, timeout=10):
        return self.session.get(url, params=params, timeout=timeout)

class WikidataAPI:
    """优化后的Wikidata API客户端"""
    _session = None
    
    @classmethod
    def get_session(cls):
        if cls._session is None:
            cls._session = WikidataSession()
        return cls._session
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def get_entity_id(text, entity_type):
        """通过文本获取Wikidata实体ID（带缓存）"""
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "type": "item",
            "search": text
        }
        print(f"Querying Wikidata for: {text}")
        try:
            response = WikidataAPI.get_session().get(WIKIDATA_API, params=params)
            response.raise_for_status()
            results = response.json().get("search", [])
            return results[0]["id"] if results else None
        except Exception as e:
            print(f"Error getting entity ID for {text}: {str(e)}")
            return None
    
    @staticmethod
    @lru_cache(maxsize=500)
    def get_property_label(property_id):
        """获取属性的标签（带缓存）"""
        params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": property_id,
            "props": "labels",
            "languages": "en"
        }
        try:
            response = WikidataAPI.get_session().get(WIKIDATA_API, params=params)
            response.raise_for_status()
            label = response.json()["entities"][property_id]["labels"]["en"]["value"]
            return label
        except Exception as e:
            print(f"Error getting label for property {property_id}: {str(e)}")
            return property_id
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def get_first_relation_between_entities(entity1_id, entity2_id):
        """获取两个实体间的第一个关系（带缓存）"""
        query = """
        SELECT ?relation ?relationLabel WHERE {
          wd:%s ?relation wd:%s.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 1
        """ % (entity1_id, entity2_id)
        
        try:
            response = WikidataAPI.get_session().get(
                SPARQL_ENDPOINT,
                params={"format": "json", "query": query},
                timeout=15
            )
            response.raise_for_status()
            bindings = response.json().get("results", {}).get("bindings", [])
            if bindings:
                item = bindings[0]
                relation_uri = item.get("relation", {}).get("value", "")
                relation_id = relation_uri.split("/")[-1]
                relation_label = item.get("relationLabel", {}).get("value", "")
                
                if not relation_label or relation_label.startswith("http"):
                    relation_label = WikidataAPI.get_property_label(relation_id)
                
                return (relation_id, relation_label)
            return None
        except Exception as e:
            print(f"Error getting relations between {entity1_id} and {entity2_id}: {str(e)}")
            return None

class DistantSupervisionRelationExtractor:
    """优化后的关系抽取器"""
    def __init__(self, nlp):
        self.nlp = nlp
        self.matcher = Matcher(nlp.vocab)
        self.wikidata = WikidataAPI()
        self.cache = defaultdict(dict)
    
    def get_entity_mapping(self, text, entity_type):
        """获取或缓存实体映射"""
        cache_key = (text, entity_type)  # 使用元组作为键
        if cache_key not in self.cache["entities"]:
            self.cache["entities"][cache_key] = self.wikidata.get_entity_id(text, entity_type)
        return self.cache["entities"][cache_key]
    
    def get_cached_relations(self, entity1_id, entity2_id):
        """获取或缓存实体间关系"""
        cache_key = (entity1_id, entity2_id)
        if cache_key not in self.cache["relations"]:
            self.cache["relations"][cache_key] = self.wikidata.get_first_relation_between_entities(entity1_id, entity2_id)
        return [self.cache["relations"][cache_key]] if self.cache["relations"][cache_key] else []
    
    def extract_relations(self, entities):
        """基于远程监督的关系抽取"""
        relations = []
        
        # 预加载所有实体ID
        entity_ids = {}
        for text, entity_type in entities:
            entity_ids[(text, entity_type)] = self.get_entity_mapping(text, entity_type)
        
        # 检查实体间关系
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                subj, subj_type = entities[i]
                obj, obj_type = entities[j]
                
                subj_id = entity_ids.get((subj, subj_type))
                obj_id = entity_ids.get((obj, obj_type))
                
                if subj_id and obj_id:
                    for rel_id, rel_label in self.get_cached_relations(subj_id, obj_id):
                        relations.append({
                            "subject": subj,
                            "relation": rel_label,
                            "object": obj,
                        })
        
        return relations

def load_entities_from_json(file_path):
    """加载entities.json文件中的实体"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def main():
    # 初始化远程监督抽取器
    print("Initializing extractor...")
    ds_extractor = DistantSupervisionRelationExtractor(nlp)
    
    print("Loading entities from JSON...")
    entities_data = load_entities_from_json("output/entities.json")[:20]
    
    results = []
    
    for entry in tqdm(entities_data, desc="Processing entries"):
        sentence = entry["sentence"]
        entities = entry["entities"]
        
        relations = ds_extractor.extract_relations(entities)
        
        results.append({
            "sentence": sentence,
            "entities": entities,
            "relations": relations or []
        })
    
    # 保存结果
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "relations_with_entities.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Processing completed. Results saved to {output_dir / 'relations_with_entities.json'}")

if __name__ == "__main__":
    main()
