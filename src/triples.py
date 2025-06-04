import json

def extract_all_triples(json_data):
    """提取所有关系三元组（不去重）"""
    triples = []
    for item in json_data:
        for rel in item["relations"]:
            # 转换为(subject, relation, object)三元组
            triple = (rel["subject"], rel["relation"], rel["object"])
            triples.append(triple)
    return triples

if __name__ == "__main__":
    # 读取数据
    with open("output/relations_valid.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    # 提取所有三元组
    triples = extract_all_triples(json_data)
    
    # 打印统计信息
    print(f"总三元组数量: {len(triples)}")
    print("示例三元组:")
    for triple in triples[:10]:  # 打印前10个示例
        print(triple)
    
    # 保存结果
    with open("output/triples_valid.json", "w", encoding="utf-8") as f:
        json.dump(triples, f, indent=2, ensure_ascii=False)
    print("\n所有关系三元组已保存到 triples_test.json")
