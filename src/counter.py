import json
from collections import defaultdict

def count_relation_types(file_path):
    relation_stats = defaultdict(int)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        for item in data:
            for relation in item.get('relations', []):
                rel_type = relation['relation']
                relation_stats[rel_type] += 1
    
    print("关系类型统计结果:")
    for rel_type, count in relation_stats.items():
        print(f"「{rel_type}」: {count}次")
    
    print(f"\n总共有 {len(relation_stats)} 种不同的关系类型")
    return dict(relation_stats)

# 使用示例
if __name__ == "__main__":
    stats = count_relation_types("output/relations_test.json")
    print("\n详细统计:")
    print(json.dumps(stats, indent=2))
