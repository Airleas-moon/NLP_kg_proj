import spacy
from spacy.matcher import Matcher
import json
from pathlib import Path
import requests
from collections import defaultdict
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 确保不使用GPU
print("Running on CPU")

# 加载模型（使用更轻量级的sm模型而不是trf）
nlp = spacy.load("en_core_web_sm", 
                 exclude=["tagger", "parser", "attribute_ruler", "lemmatizer"])

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
        print(f"Querying Wikidata for: {text}")
        try:
            response = requests.get(WIKIDATA_API, params=params, timeout=10)
            results = response.json().get("search", [])
            return results[0]["id"] if results else None
        except Exception as e:
            print(f"Error getting entity ID for {text}: {str(e)}")
            return None
    
    @staticmethod
    def get_property_label(property_id):
        """获取属性的标签（人类可读的描述）"""
        params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": property_id,
            "props": "labels",
            "languages": "en"
        }
        try:
            response = requests.get(WIKIDATA_API, params=params)
            label = response.json()["entities"][property_id]["labels"]["en"]["value"]
            return label
        except Exception as e:
            print(f"Error getting label for property {property_id}: {str(e)}")
            return property_id  # 如果获取失败，返回原始ID
    
    @staticmethod
    def get_first_relation_between_entities(entity1_id, entity2_id):
        """只获取两个实体间的第一个关系（优化性能）"""
        query = """
        SELECT ?relation ?relationLabel WHERE {
          wd:%s ?relation wd:%s.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 1
        """ % (entity1_id, entity2_id)
      
        url = "https://query.wikidata.org/sparql"
        try:
            response = requests.get(url, params={"format": "json", "query": query}, timeout=15)
            bindings = response.json().get("results", {}).get("bindings", [])
            if bindings:
                item = bindings[0]
                relation_uri = item.get("relation", {}).get("value", "")
                relation_id = relation_uri.split("/")[-1]
                relation_label = item.get("relationLabel", {}).get("value", "")
              
                # 如果SPARQL查询没有返回标签，再单独获取
                if not relation_label or relation_label.startswith("http"):
                    relation_label = WikidataAPI.get_property_label(relation_id)
              
                return (relation_id, relation_label)
            return None
        except Exception as e:
            print(f"Error getting relations between {entity1_id} and {entity2_id}: {str(e)}")
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
        """获取或缓存实体间关系（只获取第一个关系）"""
        cache_key = f"{entity1_id}_{entity2_id}"
        if cache_key not in self.cache["relations"]:
            self.cache["relations"][cache_key] = self.wikidata.get_first_relation_between_entities(entity1_id, entity2_id)
        return [self.cache["relations"][cache_key]] if self.cache["relations"][cache_key] else []
  
    def extract_relations(self, entities):
        """基于远程监督的关系抽取（带友好标签）"""
        relations = []
      
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                subj, subj_type = entities[i]
                obj, obj_type = entities[j]
              
                subj_id = self.get_entity_mapping(subj, subj_type)
                obj_id = self.get_entity_mapping(obj, obj_type)
              
                if subj_id and obj_id:
                    wikidata_relations = self.get_cached_relations(subj_id, obj_id)
                  
                    for rel_id, rel_label in wikidata_relations:
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
    entities_data = load_entities_from_json("output/entities.json")[:1]
  
    results = []
  
    for entry in tqdm(entities_data, desc="Processing entries"):
        sentence = entry["sentence"]
        entities = entry["entities"]
      
        relations = ds_extractor.extract_relations(entities)
      
        if relations:
            results.append({
                "sentence": sentence,
                "entities": entities,
                "relations": relations
            })
        else:
            results.append({
                "sentence": sentence,
                "entities": entities,
                "relations": []
            })
  
    # 保存结果
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "relations_with_entities.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Processing completed. Results saved to {output_dir / 'relations_with_entities.json'}")

if __name__ == "__main__":
    main()
