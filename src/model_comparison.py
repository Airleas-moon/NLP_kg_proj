import json
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.models import TransE, ComplEx, DistMult
from pykeen.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Set, Tuple, Optional, Dict

def load_triples(file_path) -> np.ndarray:
    """加载三元组JSON文件并返回numpy数组"""
    with open(file_path, 'r', encoding='utf-8') as f:
        triples = json.load(f)
    return np.array([tuple(t) for t in triples])

def save_triples(triples, file_path):
    """保存三元组到JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump([list(t) for t in triples], f, indent=2)

def get_incremental_triples(current_train: np.ndarray, previous_train: np.ndarray) -> np.ndarray:
    """获取增量三元组（当前训练集 - 之前训练过的三元组）"""
    current_set = {tuple(t) for t in current_train}
    previous_set = {tuple(t) for t in previous_train}
    incremental_set = current_set - previous_set
    return np.array(list(incremental_set))

def prepare_datasets(train_file: str, valid_file: str, test_file: str, 
                    previous_train_file: str = None) -> Tuple[TriplesFactory, TriplesFactory, TriplesFactory]:
    """准备数据集，支持增量训练模式"""
    current_train = load_triples(train_file)
    valid_triples = load_triples(valid_file)
    test_triples = load_triples(test_file)
    
    if previous_train_file and Path(previous_train_file).exists():
        previous_train = load_triples(previous_train_file)
        incremental_triples = get_incremental_triples(current_train, previous_train)
        print(f"\n增量训练模式: 原始三元组 {len(previous_train)} | 新增三元组 {len(incremental_triples)}")
        combined_train = np.concatenate([previous_train, incremental_triples])
    else:
        print("\n全量训练模式")
        combined_train = current_train
    
    all_triples = np.concatenate([combined_train, valid_triples, test_triples])
    tf_all = TriplesFactory.from_labeled_triples(all_triples)
    
    training = TriplesFactory.from_labeled_triples(
        triples=combined_train,
        entity_to_id=tf_all.entity_to_id,
        relation_to_id=tf_all.relation_to_id
    )
    
    validation = TriplesFactory.from_labeled_triples(
        triples=valid_triples,
        entity_to_id=tf_all.entity_to_id,
        relation_to_id=tf_all.relation_to_id
    )
    
    testing = TriplesFactory.from_labeled_triples(
        triples=test_triples,
        entity_to_id=tf_all.entity_to_id,
        relation_to_id=tf_all.relation_to_id
    )
    
    return training, validation, testing

def load_pretrained_model(model_dir: Path, model_class) -> Optional[object]:
    """尝试加载预训练模型"""
    try:
        model_path = model_dir / "trained_model.pkl"
        if model_path.exists():
            print(f"从 {model_path} 加载预训练模型")
            return load_model(model_path)
        return None
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None

def evaluate_model(model, testing, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict:
    """评估已加载的模型"""
    from pykeen.evaluation import Evaluator
    evaluator = Evaluator()
    return evaluator.evaluate(
        model=model,
        mapped_triples=testing.mapped_triples,
        device=device,
        use_tqdm=False
    )

def train_or_load_model(model_class, training, validation, testing, model_name: str, 
                       save_triples_copy: bool = True, force_retrain: bool = False):
    """训练或加载已有模型"""
    model_dir = Path(f"output/models/{model_name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 尝试加载已有模型
    if not force_retrain:
        model = load_pretrained_model(model_dir, model_class)
        if model is not None:
            # 评估已加载的模型
            metric_results = evaluate_model(model, testing)
            return {
                'model': model,
                'metric_results': metric_results,
                'train_seconds': 0,  # 加载模型时间为0
                'losses': [0]  # 无训练损失
            }
    
    # 如果没有找到模型或强制重新训练，则训练新模型
    print(f"训练新模型 {model_name}...")
    result = pipeline(
        training=training,
        validation=validation,
        testing=testing,
        model=model_class,
        training_kwargs=dict(
            num_epochs=500,
            use_tqdm_batch=False,
            checkpoint_name=f'{model_name}_checkpoint.pt',
            checkpoint_frequency=10
        ),
        evaluation_kwargs=dict(use_tqdm=False),
        random_seed=42,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 保存模型和训练数据
    result.save_to_directory(model_dir)
    
    if save_triples_copy:
        triples_copy_path = model_dir / "triples_copy.json"
        save_triples(training.triples, triples_copy_path)
        print(f"已保存训练三元组副本到: {triples_copy_path}")
    
    return result

def compare_results(results):
    """可视化比较结果，仅保留Hits@10指标"""
    metrics = []
    for name, result in results.items():
        metrics.append({
            "Model": name,
            "MRR": result['metric_results'].get_metric("both.realistic.inverse_harmonic_mean_rank"),
            "Hits@10": result['metric_results'].get_metric("both.realistic.hits_at_10"),
            "Training Time (s)": result['train_seconds'],
            "Validation Loss": min(result['losses']) if result['losses'] else 0
        })
    
    df = pd.DataFrame(metrics)
    
    # 绘制性能对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    df.plot.bar(x="Model", y="MRR", ax=axes[0, 0], title="MRR Comparison", legend=False)
    df.plot.bar(x="Model", y="Hits@10", ax=axes[0, 1], title="Hits@10 Comparison", legend=False)
    df.plot.bar(x="Model", y="Training Time (s)", ax=axes[1, 0], title="Training Time", legend=False)
    df.plot.bar(x="Model", y="Validation Loss", ax=axes[1, 1], title="Best Validation Loss", legend=False)
    
    plt.tight_layout()
    plt.savefig("output/model_comparison.png", dpi=300)
    
    print("\n=== 评估结果 ===")
    print(df[["Model", "MRR", "Hits@10"]].to_markdown(index=False))

def compare_models(train_file: str, valid_file: str, test_file: str, 
                  incremental_mode: bool = False, force_retrain: bool = False):
    """比较模型性能，支持加载已有模型"""
    print("准备数据集...")
    
    previous_train_file = None
    if incremental_mode:
        model_dirs = sorted(Path("output/models").glob("*"))
        if model_dirs:
            latest_dir = max(model_dirs, key=lambda d: d.stat().st_mtime)
            previous_train_file = latest_dir / "triples_copy.json"
            print(f"检测到增量训练模式，将使用上次训练数据: {previous_train_file}")
    
    training, validation, testing = prepare_datasets(
        train_file, valid_file, test_file, previous_train_file
    )
    
    print("\n=== 数据集统计 ===")
    print(f"训练集三元组数量: {len(training.triples)}")
    print(f"验证集三元组数量: {len(validation.triples)}")
    print(f"测试集三元组数量: {len(testing.triples)}")
    print(f"实体数量: {len(training.entity_to_id)}")
    print(f"关系数量: {len(training.relation_to_id)}")
    
    models = {
        "TransE": TransE,
        "ComplEx": ComplEx, 
        "DistMult": DistMult
    }
    
    results = {}
    for name, model_class in models.items():
        print(f"\n=== 处理 {name} 模型 ===")
        results[name] = train_or_load_model(
            model_class, training, validation, testing, name,
            save_triples_copy=incremental_mode,
            force_retrain=force_retrain
        )
    
    compare_results(results)

if __name__ == "__main__":
    TRAIN_FILE = "output/triples_train.json"
    VALID_FILE = "output/triples_valid.json"
    TEST_FILE = "output/triples_test.json"
    
    for file_path in [TRAIN_FILE, VALID_FILE, TEST_FILE]:
        if not Path(file_path).exists():
            print(f"错误: 找不到输入文件 {file_path}!")
            exit(1)
    
    compare_models(TRAIN_FILE, VALID_FILE, TEST_FILE, 
                  incremental_mode=True, 
                  force_retrain=False)  # 设置为True强制重新训练
