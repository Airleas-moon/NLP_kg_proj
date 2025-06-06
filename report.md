### NLP大作业 



#### 项目结构



```python
/KG_project
│
├── data/
|   └── raw/          # 原始数据集
│   
│
├── src/
│   ├── entity_extraction.py
│   ├── relation_extraction.py
│   └── visualization.py
│
├── requirements.txt
└── report.md     #报告
```



使用spacy提供的预训练模型en_core_web_sm，但出现误差

{    "sentence": "U.N. official Ekeus heads for Baghdad .",    "entities": [      [        "U.N.",        "ORG"      ],      [        "Ekeus",        "ORG"      ],      [        "Baghdad",        "LOC"      ]    ]  },将Ekeus识别成了ORG

尝试更大更精确的模型en_core_web_trf后，结果变精确了。

{ "sentence": "U.N. official Ekeus heads for Baghdad .","entities": [["U.N.","ORG"],["Ekeus", "PER" ],   [   "Baghdad",    "LOC"   ]  ] },



![image-20250604211247788](C:\Users\studying\AppData\Roaming\Typora\typora-user-images\image-20250604211247788.png)

![image-20250603191115050](C:\Users\studying\AppData\Roaming\Typora\typora-user-images\image-20250603191115050.png)

![image-20250604211347350](C:\Users\studying\AppData\Roaming\Typora\typora-user-images\image-20250604211347350.png)
