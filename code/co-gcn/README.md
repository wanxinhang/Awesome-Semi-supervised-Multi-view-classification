# Co-GCN

#### 介绍

Co-GCN for Multi-view Semi-supervised Learning的代码实现, 相关论文发表在 AAAI 2020 上.

#### 论文摘要

> In many real-world applications, the data have several disjoint sets of features and each set is called as a view. Researchers have developed many multi-view learning methods in the past decade. In this paper, we bring Graph Convolutional Network (GCN) into multi-view learning and propose a novel multi-view semi-supervised learning method Co-GCN by adaptively exploiting the graph information from the multiple views with combined Laplacians. Experimental results on real-world data sets verify that Co-GCN can achieve better performance compared with state-of-the-art multi-view semisupervised methods.


#### Python依赖

依赖 pytorch, numpy, spicy, scikit-learn

#### 数据集格式

单个数据集放在目录 ./data/\${数据集名称} 下, 包含features.csv和targets.csv 两个文件. 假设数据集包含$n$个样本, 每个样本在$v$个视图上有$m$个特征则:

- features.csv大小为 $(n+1)\times{m}$ , 其中第一行数字属于$0,…,v-1$, 代表该特征属于的视图.
- targets.csv大小为 $n\times{1}$ , 每一行一个数字代表该样本的标记.

数据集地址: 
- 链接: https://pan.baidu.com/s/1Iy6hRlgnON4Q3_tQCs0TNg 
- 提取码: uyvx

#### 使用教程

```shell
python -u run.py --dataset course --epochs 400 --metric cosine --labeled_ratio 0.05 --neighbors 7
```

#### 引用

```txt
@inproceedings{li2020cogcn,
  author    = {Shu Li and
               Wen{-}Tao Li and
               Wei Wang},
  title     = {Co-GCN for Multi-View Semi-Supervised Learning},
  booktitle = {The Thirty-Fourth {AAAI} Conference on Artificial Intelligence, {AAAI}
               2020, The Thirty-Second Innovative Applications of Artificial Intelligence
               Conference, {IAAI} 2020, The Tenth {AAAI} Symposium on Educational
               Advances in Artificial Intelligence, {EAAI} 2020, New York, NY, USA,
               February 7-12, 2020},
  pages     = {4691--4698},
  publisher = {{AAAI} Press},
  year      = {2020}
}
```