import spacy
from spacy.matcher import Matcher
import json
from pathlib import Path
import requests
from collections import defaultdict
from tqdm import tqdm
import torch
spacy.require_gpu() 
print("CUDA Available:", torch.cuda.is_available())
print(torch.__version__)  # PyTorch版本
print(torch.version.cuda) 
nlp = spacy.load("en_core_web_trf", 
                 exclude=["tagger", "parser", "attribute_ruler", "lemmatizer"])
nlp.enable_pipe("transformer") 
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
        except:
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
        except:
            return property_id  # 如果获取失败，返回原始ID
    
    @staticmethod
    def get_relations_between_entities(entity1_id, entity2_id):
        """获取两个实体间的所有关系（带标签）"""
        query = """
        SELECT ?relation ?relationLabel WHERE {
          wd:%s ?relation wd:%s.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        """ % (entity1_id, entity2_id)
      
        url = "https://query.wikidata.org/sparql"
        try:
            response = requests.get(url, params={"format": "json", "query": query})
            relations = []
            for item in response.json().get("results", {}).get("bindings", []):
                relation_id = item.get("relation", {}).get("value", "").split("/")[-1]
                relation_label = item.get("relationLabel", {}).get("value", "")
              
                # 如果SPARQL查询没有返回标签，再单独获取
                if not relation_label or relation_label.startswith("http"):
                    relation_label = WikidataAPI.get_property_label(relation_id)
              
                relations.append((relation_id, relation_label))
            return relations
        except:
            return []
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
                        # 确保标签不是URL格式
                        if rel_label.startswith("http"):
                            rel_label = rel_id  # 回退到显示属性ID
                      
                        relations.append({
                            "subject": subj,
                            "subject_type": subj_type,
                            "relation": rel_label,  # 现在这里是友好标签
                            "relation_id": rel_id,
                            "object": obj,
                            "object_type": obj_type
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
    entities_data = load_entities_from_json("output/entities.json")
  
    results = []
  
    for entry in tqdm(entities_data, desc="Processing entries"):
        sentence = entry["sentence"]
        entities = entry["entities"]
      
        print(f"\nProcessing sentence: {sentence[:50]}...")
        relations = ds_extractor.extract_relations(entities)
        print(f"Found {len(relations)} relations")
      
        if relations:
            results.append({
                "sentence": sentence,
                "entities": entities,
                "relations": relations
            })
        else:
            results.append({
                "sentence": sentence,
                "entities": entities
            })
  
    # 保存结果
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "relations_with_entities.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
if __name__ == "__main__":
    main()