import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from matplotlib.lines import Line2D

def load_triples(file_path):
    """加载三元组数据"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)  # 格式: [[subject, object, relation], ...]

def build_graph(triples, top_n=20):
    G = nx.DiGraph()
    node_counter = Counter()
    
    # 先统计所有节点频次
    for subj, obj, _ in triples:
        node_counter.update([subj, obj])
    
    # 取TopN节点
    top_nodes = [node for node, _ in node_counter.most_common(top_n)]
    
    # 只添加这些节点之间的边
    for subj, obj, rel in triples:
        if subj in top_nodes and obj in top_nodes:
            G.add_edge(subj, obj, label=rel)
    
    # 确保所有节点都被添加（即使没有边）
    G.add_nodes_from(top_nodes)
    return G

def visualize_subgraph(subgraph, save_path):
    """可视化知识子图（优化版）"""
    plt.figure(figsize=(20, 15))
    
    # ========== 1. 智能布局 ==========
    if len(subgraph) <= 30:
        pos = nx.kamada_kawai_layout(subgraph)
    else:
        pos = nx.spring_layout(subgraph, k=1.5, iterations=100, seed=42)
    
    # ========== 2. 节点样式 ==========
    degrees = dict(subgraph.degree())
    node_sizes = [800 + 150 * degrees[n] for n in subgraph.nodes()]
    node_colors = [degrees[n] for n in subgraph.nodes()]
    
    # 绘制节点（带边框）
    nodes = nx.draw_networkx_nodes(
        subgraph, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.plasma,
        alpha=0.85,
        linewidths=1.5,
        edgecolors="white"
    )
    
    # ========== 3. 边样式 ==========
    # 获取所有唯一关系类型
    unique_rels = list(set([d["label"] for _, _, d in subgraph.edges(data=True)]))
    rel_colormap = plt.cm.get_cmap("tab20", len(unique_rels))
    
    # 按关系类型分组绘制边
    for i, rel in enumerate(unique_rels):
        edges = [(u, v) for u, v, d in subgraph.edges(data=True) if d["label"] == rel]
        
        # 曲线边减少交叉
        nx.draw_networkx_edges(
            subgraph, pos,
            edgelist=edges,
            width=1.5,
            edge_color=[rel_colormap(i)] * len(edges),
            style="solid",
            arrows=True,
            arrowsize=15,
            arrowstyle="->",
            connectionstyle="arc3,rad=0.15"  # 曲线边
        )
    
    # ========== 4. 标签优化 ==========
    # 节点标签（垂直偏移避免遮挡）
    label_pos = {k: [v[0], v[1]+0.025] for k, v in pos.items()}
    nx.draw_networkx_labels(
        subgraph, label_pos,
        font_size=10,
        font_family="sans-serif",
        font_weight="bold",
        bbox=dict(
            facecolor="white",
            alpha=0.85,
            edgecolor="none",
            pad=0.3,
            boxstyle="round,pad=0.3"
        )
    )
    
    # 边标签（只显示重要关系）
    edge_labels = {}
    for u, v, d in subgraph.edges(data=True):
        if degrees[u] + degrees[v] > np.percentile(list(degrees.values()), 75):
            edge_labels[(u, v)] = d["label"]
    
    nx.draw_networkx_edge_labels(
        subgraph, pos,
        edge_labels=edge_labels,
        font_size=8,
        font_weight="bold",
        label_pos=0.5,
        bbox=dict(
            facecolor="white",
            alpha=0.9,
            edgecolor="none",
            boxstyle="round,pad=0.2"
        )
    )
    
    # ========== 5. 图例系统 ==========
    legend_elements = [
        Line2D([0], [0], 
               color=rel_colormap(i), 
               lw=3, 
               label=f"{rel} (n={len([e for e in subgraph.edges(data=True) if e[2]['label']==rel])})"
        ) for i, rel in enumerate(unique_rels)
    ]
    
    plt.legend(
        handles=legend_elements,
        title="Relation Types",
        loc="upper right",
        fontsize=9,
        title_fontsize=10,
        framealpha=0.9
    )
    
    # ========== 6. 标题与保存 ==========
    plt.title(
        f"Knowledge Graph Visualization\n(Top {len(subgraph)} Connected Nodes, {len(subgraph.edges())} Relations)",
        fontsize=14,
        pad=20
    )
    plt.axis("off")
    plt.tight_layout()
    
    # 保存高清图像
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close()
    print(f"可视化结果已保存至: {save_path}")

if __name__ == "__main__":
    # 参数配置
    INPUT_FILE = "output/unique_relations.json"  # 输入三元组文件
    OUTPUT_IMAGE = "output/kg_visualization.png" # 输出图片路径
    TOP_N = 45  # 建议20-30个节点
    
    # 执行流程
    print("正在加载数据...")
    triples = load_triples(INPUT_FILE)
    
    print("构建知识图谱...")
    subgraph = build_graph(triples, top_n=TOP_N)
    
    print(f"可视化子图（{len(subgraph)}节点, {len(subgraph.edges())}边）...")
    visualize_subgraph(subgraph, OUTPUT_IMAGE)