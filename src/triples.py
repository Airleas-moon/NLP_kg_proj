import json

def extract_all_triples(json_data):
    """提取所有关系三元组"""
    triples = []
    for item in json_data:
        for rel in item["relations"]:
            triple = (rel["subject"], rel["relation"], rel["object"])
            triples.append(triple)
    return triples

if __name__ == "__main__":
    with open("output/relations_valid.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    triples = extract_all_triples(json_data)
    
    print(f"总三元组数量: {len(triples)}")
    print("示例三元组:")
    for triple in triples[:10]:  
        print(triple)
    
    with open("output/triples_valid.json", "w", encoding="utf-8") as f:
        json.dump(triples, f, indent=2, ensure_ascii=False)
    print("\n已保存到 triples_test.json")
