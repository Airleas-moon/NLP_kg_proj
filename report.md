### NLP大作业 



#### 项目结构

/KG_project
│
├── data/
│   ├── raw/          # 原始数据集
│   └── processed/    # 处理后的三元组
│
├── src/
│   ├── entity_extraction.py
│   ├── relation_extraction.py
│   └── visualization.py
│
├── requirements.txt
└── report.md     #报告



使用spacy提供的预训练模型en_core_web_sm，但出现误差

{    "sentence": "U.N. official Ekeus heads for Baghdad .",    "entities": [      [        "U.N.",        "ORG"      ],      [        "Ekeus",        "ORG"      ],      [        "Baghdad",        "LOC"      ]    ]  },将Ekeus识别成了ORG

尝试更大更精确的模型en_core_web_trf后，结果变精确了。

{ "sentence": "U.N. official Ekeus heads for Baghdad .","entities": [["U.N.","ORG"],["Ekeus", "PER" ],   [   "Baghdad",    "LOC"   ]  ] },