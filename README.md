# Rotary Transformer

Rotary Transformer，简称RoFormer，是我们自研的语言模型之一，主要是为Transformer结构设计了新的旋转式位置编码（Rotary Position Embedding，RoPE）。RoPE具有良好的理论性质，且是目前唯一一种可以应用到线性Attention的绝对位置编码，目前来看实验结果也颇为不错。

详细介绍：https://kexue.fm/archives/8265

## 依赖

```
bert4keras 0.10.4
```

## 下载
- [chinese_roformer_L-12_H-768_A-12.zip](https://pan.baidu.com/s/1fiss862YsGCwf2HvU_Jm-g)(提取码：xy9x)

## 引用

Bibtex：

```tex
@techreport{zhuiyiroformer,
  title={RoFormer: Transformer with Rotary Position Embeddings - ZhuiyiAI},
  author={Jianlin Su},
  year={2021},
  url="https://github.com/ZhuiyiTechnology/roformer",
}
```

## 联系

邮箱：ai@wezhuiyi.com
追一科技：https://zhuiyi.ai
