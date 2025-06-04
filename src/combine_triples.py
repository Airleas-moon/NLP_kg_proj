import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import statistics

class AutoRelationAnalyzer:
    def __init__(self):
        self.relation_stats = defaultdict(lambda: {
            'symmetric_pairs': set(),
            'unique_heads': set(),
            'unique_tails': set()
        })

    def analyze_triples(self, triples: List[Tuple[str, str, str]]) -> Dict[str, Dict]:
        """自动分析所有关系模式（无实体类型推断）"""
        for h, r, t in triples:
            stats = self.relation_stats[r]
            stats['symmetric_pairs'].add((h, t))
            stats['unique_heads'].add(h)
            stats['unique_tails'].add(t)
        return self._derive_constraints()

    def _derive_constraints(self) -> Dict[str, Dict]:
        """推导动态约束规则（仅基于关系模式）"""
        constraints = {}
        for rel, stats in self.relation_stats.items():
            # 对称性检测
            symmetric_evidence = sum(
                1 for (h, t) in stats['symmetric_pairs'] 
                if (t, h) in stats['symmetric_pairs']
            )
            total_pairs = len(stats['symmetric_pairs'])
            symmetric_score = symmetric_evidence / total_pairs if total_pairs > 0 else 0
            
            # 唯一性检测
            head_counts = Counter(h for h, _ in stats['symmetric_pairs'])
            avg_tails_per_head = statistics.mean(head_counts.values()) if head_counts else 0
            
            constraints[rel] = {
                'symmetric': symmetric_score > 0.8,
                'unique_head': avg_tails_per_head < 1.1
            }
        return constraints

class TripleProcessor:
    def __init__(self):
        self.analyzer = AutoRelationAnalyzer()
        self.dynamic_constraints = {}

    def load_and_analyze(self, file_path: str) -> List[Tuple[str, str, str]]:
        """加载并分析三元组文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            triples = json.load(f)
            validated = [(h.strip(), r.strip(), t.strip()) for h, r, t in triples]
            self.dynamic_constraints = self.analyzer.analyze_triples(validated)
            return validated

    def detect_all_conflicts(self, original: List[Tuple], new: List[Tuple]) -> Dict:
        """保守冲突检测（无类型检查）"""
        conflict_types = defaultdict(set)
        original_set = set(original)
        new_set = set(new)
        
        # 构建关系索引 - 修复初始化方式
        rel_index = defaultdict(lambda: {
            'head_to_tails': defaultdict(set),
            'tail_to_heads': defaultdict(set)
        })
        
        for h, r, t in original + new:
            rel_index[r]['head_to_tails'][h].add(t)
            rel_index[r]['tail_to_heads'][t].add(h)
        for h, r, t in new_set:
            # 直接冲突检测
            if (h, r, t) in original_set:
                conflict_types['direct_collision'].add((h, r, t))
                continue
                
            # 动态规则检测（仅对称性和唯一性）
            if r in self.dynamic_constraints:
                constraints = self.dynamic_constraints[r]
                
                # 对称性冲突
                if constraints['symmetric'] and (t, r, h) not in original_set:
                    conflict_types['asymmetric_violation'].add((h, r, t))
                
                # 唯一性冲突
                if constraints['unique_head'] and len(rel_index[r]['head_to_tails'][h]) > 1:
                    conflict_types['uniqueness_violation'].add((h, r, t))
        return dict(conflict_types)


    def merge_with_auto_rules(self, original_file: str, new_file: str, output_path: str):
        """保守合并流程"""
        original = self.load_and_analyze(original_file)
        new = self.load_and_analyze(new_file)
        
        conflicts = self.detect_all_conflicts(original, new)
        all_conflicts = set().union(*conflicts.values())
        
        # 执行合并（保留原始冲突数据）
        merged = set(original)
        merged.update(t for t in new if t not in all_conflicts)
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(list(merged), f, indent=2, ensure_ascii=False)
        
        # 生成简化报告
        self.generate_basic_report(conflicts, output_path.replace('.json', '_report.json'))

    def generate_basic_report(self, conflicts: Dict, report_path: str):
        """生成基本冲突报告"""
        report = {
            "conflict_statistics": {
                "total_conflicts": sum(len(v) for v in conflicts.values()),
                "by_type": {k: len(v) for k, v in conflicts.items()},
                "top_conflict_relations": Counter(
                    r for conflicts in conflicts.values() for h, r, t in conflicts
                ).most_common(5)
            },
            "conflict_examples": {
                k: list(v)[:3] for k, v in conflicts.items() if v
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    processor = TripleProcessor()
    processor.merge_with_auto_rules(
        original_file="output/triples_train.json",
        new_file="output/triples_test.json",
        output_path="output/merged_triples.json"
    )
