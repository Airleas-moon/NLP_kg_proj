from collections import defaultdict
import json

def filter_high_frequency_relations(triples, min_freq=10):
    # 第一步：统计关系频次
    relation_counter = defaultdict(int)
    for triple in triples:
        relation = triple[1]  # 关系是三元组的第二个元素
        relation_counter[relation] += 1
    
    # 第二步：筛选高频关系
    high_freq_relations = {rel for rel, count in relation_counter.items() if count >= min_freq}
    
    # 第三步：筛选保留高频关系的三元组
    filtered_triples = [triple for triple in triples if triple[1] in high_freq_relations]
    
    # 返回结果
    return {
        "relation_stats": dict(relation_counter),
        "filtered_triples": filtered_triples,
        "high_freq_relations": list(high_freq_relations)
    }

# 示例数据
sample_triples = [
    ["JAPAN", "shares border with", "CHINA"],
    ["Japan", "diplomatic relation", "Syria"],
    ["China", "diplomatic relation", "Uzbekistan"],
    ["China", "language used", "Chinese"],
    ["Japan", "diplomatic relation", "Syria"],
    ["Nader Jokhadar", "place of birth", "Syria"],
    ["Japan", "diplomatic relation", "Syrian"]
]

# 执行处理
result = filter_high_frequency_relations(sample_triples, min_freq=2)

# 打印结果
print("关系类型统计:")
for rel, count in sorted(result["relation_stats"].items(), key=lambda x: -x[1]):
    print(f"{rel}: {count}次")

print("\n高频关系(出现≥2次):", result["high_freq_relations"])
print("\n筛选后的三元组:")
for triple in result["filtered_triples"]:
    print(triple)
