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
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from threading import Lock
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wikidata_processing_valid.log'),
        logging.StreamHandler()
    ]
)

os.environ["CUDA_VISIBLE_DEVICES"] = "" 
logger = logging.getLogger(__name__)
logger.info("Running on CPU")

# 加载模型（使用更轻量级的sm模型而不是trf）
nlp = spacy.load("en_core_web_sm", 
                 exclude=["tagger", "parser", "attribute_ruler", "lemmatizer"])

# 知识库API配置
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

class RateLimiter:
    """请求速率限制器"""
    _instance = None
    _lock = Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_limiter()
        return cls._instance
    
    def _init_limiter(self):
        self.last_request_time = 0
        self.min_interval = 1  
        self.lock = Lock()
    
    def wait_for_next_request(self):
        with self.lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
            self.last_request_time = time.time()

class WikidataSession:
    def __init__(self):
        self.session = requests.Session()
        self.rate_limiter = RateLimiter()
        
        # 配置自动重试策略
        retry_strategy = Retry(
            total=3,
            backoff_factor=1, 
            status_forcelist=[408, 429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_maxsize=4,  
            pool_block=True  
        )
        self.session.mount("https://", adapter)
        
        
        self.session.headers.update({
            'User-Agent': 'AcademicResearchBot/1.0 ',
            'Accept': 'application/json'
        })
    
    def get(self, url, params=None, timeout=10):
        """带速率限制的GET请求"""
        self.rate_limiter.wait_for_next_request()
        
        try:
            response = self.session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            
            
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 30))
                logger.warning(f"Rate limited. Waiting {retry_after} seconds")
                time.sleep(retry_after)
                return self.get(url, params, timeout)
                
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            raise

class WikidataAPI:
    """优化后的Wikidata API客户端"""
    _session = None
    BATCH_SIZE = 20  # 批处理大小
    MAX_WORKERS = 4  # 最大并发线程数
    
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
        try:
            response = WikidataAPI.get_session().get(WIKIDATA_API, params=params)
            results = response.json().get("search", [])
            return results[0]["id"] if results else None
        except Exception as e:
            logger.error(f"Error getting entity ID for {text}: {str(e)}")
            return None
    
    @classmethod
    def get_entity_ids_batch(cls, entities):
        """批量获取实体ID（带速率限制）"""
        results = {}
        with ThreadPoolExecutor(max_workers=cls.MAX_WORKERS) as executor:
            futures = {
                executor.submit(cls.get_entity_id, text, etype): (text, etype)
                for text, etype in entities
            }
            for future in as_completed(futures):
                text, etype = futures[future]
                try:
                    results[(text, etype)] = future.result()
                except Exception as e:
                    logger.error(f"Error processing entity {text}: {str(e)}")
                    results[(text, etype)] = None
        return results
    
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
            label = response.json()["entities"][property_id]["labels"]["en"]["value"]
            return label
        except Exception as e:
            logger.error(f"Error getting label for property {property_id}: {str(e)}")
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
                params={"format": "json", "query": query}
            )
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
            logger.error(f"Error getting relations between {entity1_id} and {entity2_id}: {str(e)}")
            return None
    
    @classmethod
    def get_relations_batch(cls, entity_pairs):
        """批量获取实体间关系（带速率限制）"""
        results = {}
        with ThreadPoolExecutor(max_workers=cls.MAX_WORKERS) as executor:
            futures = {
                executor.submit(cls.get_first_relation_between_entities, e1, e2): (e1, e2)
                for e1, e2 in entity_pairs
            }
            for future in as_completed(futures):
                e1, e2 = futures[future]
                try:
                    results[(e1, e2)] = future.result()
                except Exception as e:
                    logger.error(f"Error processing relation between {e1} and {e2}: {str(e)}")
                    results[(e1, e2)] = None
        return results

class DistantSupervisionRelationExtractor:
    """优化后的关系抽取器"""
    def __init__(self, nlp):
        self.nlp = nlp
        self.matcher = Matcher(nlp.vocab)
        self.wikidata = WikidataAPI()
        self.cache = defaultdict(dict)
    
    def get_entity_mapping(self, text, entity_type):
        """获取或缓存实体映射"""
        cache_key = (text, entity_type)
        if cache_key not in self.cache["entities"]:
            self.cache["entities"][cache_key] = self.wikidata.get_entity_id(text, entity_type)
        return self.cache["entities"][cache_key]
    
    def get_entity_mappings_batch(self, entities):
        """批量获取实体映射"""
        # 先检查缓存
        to_fetch = [(text, etype) for text, etype in entities 
                   if (text, etype) not in self.cache["entities"]]
        
        if to_fetch:
            # 批量查询缺失的实体
            batch_results = self.wikidata.get_entity_ids_batch(to_fetch)
            self.cache["entities"].update(batch_results)
        
        return {e: self.cache["entities"].get(e) for e in entities}
    
    def get_cached_relations(self, entity1_id, entity2_id):
        """获取或缓存实体间关系"""
        cache_key = (entity1_id, entity2_id)
        if cache_key not in self.cache["relations"]:
            self.cache["relations"][cache_key] = self.wikidata.get_first_relation_between_entities(entity1_id, entity2_id)
        return [self.cache["relations"][cache_key]] if self.cache["relations"][cache_key] else []
    
    def get_relations_batch(self, entity_pairs):
        """批量获取实体间关系"""
        # 先检查缓存
        to_fetch = [(e1, e2) for e1, e2 in entity_pairs 
                   if (e1, e2) not in self.cache["relations"]]
        
        if to_fetch:
            # 批量查询缺失的关系
            batch_results = self.wikidata.get_relations_batch(to_fetch)
            self.cache["relations"].update(batch_results)
        
        return {pair: self.cache["relations"].get(pair) for pair in entity_pairs}
    
    def extract_relations(self, entities):
        """基于远程监督的关系抽取（批处理优化版）"""
        relations = []
        
        # 1. 批量获取所有实体ID
        entity_ids = self.get_entity_mappings_batch([(text, etype) for text, etype in entities])
        
        # 2. 收集所有可能的实体对
        entity_pairs = []
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                subj, subj_type = entities[i]
                obj, obj_type = entities[j]
                subj_id = entity_ids.get((subj, subj_type))
                obj_id = entity_ids.get((obj, obj_type))
                if subj_id and obj_id:
                    entity_pairs.append((subj_id, obj_id))
        
        # 3. 批量查询关系
        relation_results = self.get_relations_batch(entity_pairs)
        
        # 4. 构建结果
        for (subj_id, obj_id), rel in relation_results.items():
            if rel:
                rel_id, rel_label = rel
                # 查找原始文本
                subj_text = next(text for (text, etype), eid in entity_ids.items() 
                                if eid == subj_id)
                obj_text = next(text for (text, etype), eid in entity_ids.items() 
                               if eid == obj_id)
                relations.append({
                    "subject": subj_text,
                    "relation": rel_label,
                    "object": obj_text
                })
                logger.info(f"Extracted relation: {subj_text} - {rel_label} - {obj_text}")
        
        return relations

def load_entities_from_json(file_path):
    """加载entities.json文件中的实体"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def main():
    # 初始化远程监督抽取器
    logger.info("Initializing extractor...")
    ds_extractor = DistantSupervisionRelationExtractor(nlp)
    
    logger.info("Loading entities from JSON...")
    entities_data = load_entities_from_json("output/entities_valid.json")
    
    results = []
    
    # 分块处理防止内存溢出
    CHUNK_SIZE = 100
    for i in tqdm(range(0, len(entities_data), CHUNK_SIZE), desc="Processing chunks"):
        chunk = entities_data[i:i+CHUNK_SIZE]
        chunk_results = []
        
        for entry in chunk:
            sentence = entry["sentence"]
            entities = entry["entities"]
            
            try:
                relations = ds_extractor.extract_relations(entities)
                chunk_results.append({
                    "sentence": sentence,
                    "entities": entities,
                    "relations": relations or []
                })
            except Exception as e:
                logger.error(f"Error processing sentence: {sentence[:50]}... Error: {str(e)}")
                chunk_results.append({
                    "sentence": sentence,
                    "entities": entities,
                    "relations": [],
                    "error": str(e)
                })
        
        # 保存当前块结果
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "relations_valid.json"
        
        # 处理JSON文件写入方式
        if not output_file.exists():
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunk_results, f, indent=2, ensure_ascii=False)
        else:
            with open(output_file, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                data.extend(chunk_results)
                f.seek(0)
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processed chunk {i//CHUNK_SIZE + 1}/{(len(entities_data)-1)//CHUNK_SIZE + 1}")
    
    logger.info(f"Processing completed. Results saved to output/relations_valid.json")

if __name__ == "__main__":
    main()
