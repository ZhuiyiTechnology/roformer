[[中文](https://github.com/ZhuiyiTechnology/roformer/blob/main/README_zh.md)|[English](https://github.com/ZhuiyiTechnology/roformer/blob/main/README.md)]

# Rotary Transformer

Rotary Transformer，简称RoFormer，是我们自研的语言模型之一，主要是为Transformer结构设计了新的旋转式位置编码（Rotary Position Embedding，RoPE）。RoPE具有良好的理论性质，且是目前唯一一种可以应用到线性Attention的绝对位置编码，目前来看实验结果也颇为不错。

详细介绍：https://kexue.fm/archives/8265

## 依赖

```
bert4keras 0.10.4
```

参考配置：在24G显存的3090上，跑maxlen=1024，batch_size能跑到8以上。

## 下载
- [chinese_roformer_L-12_H-768_A-12.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer_L-12_H-768_A-12.zip)
- [chinese_roformer_L-6_H-384_A-6.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer_L-6_H-384_A-6.zip)
- [chinese_roformer-char_L-12_H-768_A-12.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-char_L-12_H-768_A-12.zip)
- [chinese_roformer-char_L-6_H-384_A-6.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-char_L-6_H-384_A-6.zip)
- [chinese_roformer-gpt-char_L-12_H-768_A-12.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-gpt-char_L-12_H-768_A-12.zip)
- [chinese_roformer-sim-char_L-12_H-768_A-12.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-sim-char_L-12_H-768_A-12.zip)
- [chinese_roformer-sim-char_L-6_H-384_A-6.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-sim-char_L-6_H-384_A-6.zip)
- [chinese_roformer-sim-char-ft_L-12_H-768_A-12.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-sim-char-ft_L-12_H-768_A-12.zip)
- [chinese_roformer-sim-char-ft_L-6_H-384_A-6.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-sim-char-ft_L-6_H-384_A-6.zip)

## 其他

有热心网友转换了PyTorch版，有需要的朋友可以尝试：https://github.com/JunnYu/RoFormer_pytorch

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
