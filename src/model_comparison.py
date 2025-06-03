import json
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.models import TransE, ComplEx, DistMult
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_and_prepare_data(file_path):
    """加载并预处理数据"""
    with open(file_path, 'r') as f:
        triples = json.load(f)
    
    # 转换为PyKEEN需要的三元组格式
    triples_array = np.array(triples)
    return TriplesFactory.from_labeled_triples(
        triples=triples_array,
        create_inverse_triples=False  # 根据需求决定是否创建逆向关系
    )

def train_evaluate_model(model_class, training, testing, model_name):
    """训练和评估单个模型"""
    result = pipeline(
        training=training,
        testing=testing,
        model=model_class,
        training_kwargs=dict(num_epochs=500, use_tqdm_batch=False),
        evaluation_kwargs=dict(use_tqdm=False),
        random_seed=42,
        device='cuda'  # 使用GPU加速

    )
    
    # 保存模型
    result.save_to_directory(f"output/{model_name}")
    return result

def compare_models(triples_file):
    """比较三种模型性能"""
    # 加载数据
    tf = load_and_prepare_data(triples_file)
    training, testing = tf.split([0.8, 0.2])
    
    # 定义模型配置
    models = {
        "TransE": TransE,
        "ComplEx": ComplEx, 
        "DistMult": DistMult
    }
    
    # 训练和评估
    results = {}
    for name, model_class in models.items():
        print(f"\n=== 训练 {name} 模型 ===")
        results[name] = train_evaluate_model(model_class, training, testing, name)
    
    # 性能比较
    compare_results(results)

def compare_results(results):
    """可视化比较结果"""
    metrics = []
    for name, result in results.items():
        metrics.append({
            "Model": name,
            "MRR": result.metric_results.get_metric("both.realistic.inverse_harmonic_mean_rank"),
            "Hits@10": result.metric_results.get_metric("both.realistic.hits_at_10"),
            "Training Time": result.train_seconds  # 直接使用总训练时间
        })
    
    df = pd.DataFrame(metrics)
    
    # 绘制性能对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    df.plot.bar(x="Model", y="MRR", ax=axes[0], title="MRR Comparison", legend=False)
    df.plot.bar(x="Model", y="Hits@10", ax=axes[1], title="Hits@10 Comparison", legend=False)
    df.plot.bar(x="Model", y="Training Time", ax=axes[2], title="Training Time (s)", legend=False)
    
    plt.tight_layout()
    plt.savefig("output/model_comparison_7_3.png", dpi=300)
    
    # 打印详细结果
    print("\n=== 详细评估结果 ===")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    # 配置参数
    TRIPLES_FILE = "output/triples.json"
    tf = load_and_prepare_data(TRIPLES_FILE)
    print(f"实体数量: {len(tf.entity_to_id)}")
    print(f"关系数量: {len(tf.relation_to_id)}")
    print(f"三元组数量: {len(tf.triples)}")
    # 执行比较
    compare_models(TRIPLES_FILE)