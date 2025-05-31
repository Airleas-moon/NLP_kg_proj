import spacy
import json
from itertools import groupby
from pathlib import Path
nlp = spacy.load("en_core_web_trf")
print(nlp("Hello, world!").text) 
LABEL_MAPPING = {
    "ORG": "ORG",
    "PERSON": "PER",
    "GPE": "LOC",
    "NORP": "MISC",
    "FAC": "MISC",
    "PRODUCT": "MISC"
}

def process_conll_file(file_path):
    """处理CoNLL格式文件，返回句子列表"""
    sentences = []
    current_sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith("-DOCSTART-"):
                if current_sentence:
                    sentences.append(" ".join(current_sentence))
                    current_sentence = []
                continue
                
            if not line:  # 空行表示句子结束
                if current_sentence:
                    sentences.append(" ".join(current_sentence))
                    current_sentence = []
                continue
                
            # 提取第一个字段（单词）
            parts = line.split()
            if parts:
                current_sentence.append(parts[0])
                
        # 处理文件末尾的最后一个句子
        if current_sentence:
            sentences.append(" ".join(current_sentence))
    
    return sentences

def extract_entities(sentences):
    """执行NER并提取格式化结果"""
    results = []
    
    for sent in sentences:
        doc = nlp(sent)
        entities = []
        
        for ent in doc.ents:
            mapped_label = LABEL_MAPPING.get(ent.label_, None)
            if mapped_label:
                entities.append((ent.text, mapped_label))
        
        results.append(entities)
    
    return results

def save_to_json(output_path, sentences, entity_results):
    """保存为JSON格式"""
    data = [{
        "sentence": sent,
        "entities": ents
    } for sent, ents in zip(sentences, entity_results)]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
# 使用示例
if __name__ == "__main__":
    #
    input_file = "../data/raw/train.txt"  
    sentences = process_conll_file(input_file)
    entity_results = extract_entities(sentences)
    
    # 创建输出目录
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # 保存为JSON格式
    json_path = output_dir / "entities.json"
    save_to_json(json_path, sentences, entity_results)

    print("Sample Results:")
    for i, (sent, ents) in enumerate(zip(sentences[:3], entity_results[:3])):
        print(f"\nSentence {i+1}: {sent}")
        print(f"Entities: {json.dumps(ents, indent=2, ensure_ascii=False)}")
    
    
    
    