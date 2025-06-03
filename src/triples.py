import json

def extract_unique_relations(json_data):
    """每种关系只保留一个任意三元组"""
    unique_relations = {}
    for item in json_data:
        for rel in item["relations"]:
            relation_type = rel["relation"]
            if relation_type not in unique_relations:  # 只保留首次出现的关系
                unique_relations[relation_type] = (rel["subject"], rel["object"], relation_type)
    return list(unique_relations.values())  # 返回所有唯一关系

if __name__ == "__main__":
    # 读取数据
    with open("output/relations_with_entities.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    # 处理并去重
    triples = extract_unique_relations(json_data)
    
    # 打印统计信息
    print(f"唯一关系类型数量: {len(triples)}")
    print("示例（每种关系保留一个三元组）:")
    for triple in triples[:10]:  # 打印前10个示例
        print(triple)
    
    # 保存结果
    with open("output/unique_relations.json", "w", encoding="utf-8") as f:
        json.dump(triples, f, indent=2, ensure_ascii=False)
    print("\n唯一关系三元组已保存到 unique_relations.json")
