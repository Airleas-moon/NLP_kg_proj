import json
import bz2
from pathlib import Path
import sqlite3
from tqdm import tqdm
import os
import time

def create_sqlite_db(dump_path: str, output_db: str):
    """将Wikidata JSON dump转换为SQLite数据库（带增强型进度条）"""
    
    # 获取文件大小用于进度计算
    file_size = os.path.getsize(dump_path)
    
    # 初始化数据库
    conn = sqlite3.connect(output_db)
    cursor = conn.cursor()
    
    # 创建表结构
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS entities (
        id TEXT PRIMARY KEY,
        labels TEXT,
        descriptions TEXT
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS relations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id TEXT,
        predicate_id TEXT,
        object_id TEXT,
        FOREIGN KEY(subject_id) REFERENCES entities(id),
        FOREIGN KEY(object_id) REFERENCES entities(id)
    )
    """)
    
    # 添加索引（处理完成后添加更快）
    cursor.execute("PRAGMA journal_mode = OFF")  # 导入期间禁用日志以获得更快速度
    cursor.execute("PRAGMA synchronous = OFF")
    cursor.execute("PRAGMA cache_size = 1000000")  # 1GB缓存
    
    # 开始事务（大幅提升写入速度）
    cursor.execute("BEGIN TRANSACTION")
    
    # 初始化计数器
    entities_count = 0
    relations_count = 0
    last_commit_time = time.time()
    
    # 解析JSON dump
    with bz2.open(dump_path, 'rt', encoding='utf-8') as f:
        # 增强的进度条配置
        with tqdm(
            total=file_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc="Processing Wikidata",
            mininterval=1,  # 最小更新间隔
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        ) as pbar:
            for line in f:
                if line.startswith('[') or line.startswith(']'):
                    pbar.update(len(line.encode('utf-8')))
                    continue
                    
                # 更新进度条（按字节计算）
                pbar.update(len(line.encode('utf-8')))
                
                try:
                    entity = json.loads(line.rstrip(',\n'))
                except json.JSONDecodeError:
                    continue
                    
                entity_id = entity.get('id')
                if not entity_id:
                    continue
                
                # 存储实体基本信息
                labels = json.dumps(entity.get('labels', {}))
                descriptions = json.dumps(entity.get('descriptions', {}))
                cursor.execute(
                    "INSERT OR IGNORE INTO entities VALUES (?, ?, ?)",
                    (entity_id, labels, descriptions)
                )
                entities_count += 1
                
                # 存储关系数据
                claims = entity.get('claims', {})
                for prop_id, statements in claims.items():
                    for stmt in statements:
                        if 'mainsnak' not in stmt:
                            continue
                            
                        mainsnak = stmt['mainsnak']
                        if mainsnak.get('datatype') == 'wikibase-item':
                            object_id = mainsnak.get('datavalue', {}).get('value', {}).get('id')
                            if object_id:
                                cursor.execute(
                                    "INSERT INTO relations (subject_id, predicate_id, object_id) VALUES (?, ?, ?)",
                                    (entity_id, prop_id, object_id)
                                )
                                relations_count += 1
                
                # 定期提交和更新进度条附加信息
                if time.time() - last_commit_time > 30:  # 每30秒提交一次
                    conn.commit()
                    cursor.execute("BEGIN TRANSACTION")
                    last_commit_time = time.time()
                    pbar.set_postfix({
                        'entities': f'{entities_count:,}',
                        'relations': f'{relations_count:,}'
                    })
    
    # 最终提交
    conn.commit()
    
    # 添加索引（所有数据导入后创建更快）
    tqdm.write("\nCreating indexes...")
    with tqdm(total=3, desc="Indexing") as pbar:
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject_id)")
        pbar.update(1)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(object_id)")
        pbar.update(1)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relations_predicate ON relations(predicate_id)")
        pbar.update(1)
    
    conn.close()
    
    print(f"\nProcessing completed! Total: {entities_count:,} entities, {relations_count:,} relations")

if __name__ == "__main__":
    dump_path = "latest-all.json.bz2"  # 下载的Wikidata dump路径
    output_db = "wikidata.db"
    
    print("Starting Wikidata dump processing...")
    print(f"Input file: {dump_path}")
    print(f"Output database: {output_db}")
    
    start_time = time.time()
    create_sqlite_db(dump_path, output_db)
    
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time/3600:.2f} hours")