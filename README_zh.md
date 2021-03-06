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
- [chinese_roformer_L-12_H-768_A-12.zip](https://pan.baidu.com/s/1fiss862YsGCwf2HvU_Jm-g)(提取码：xy9x)
- [chinese_roformer_L-6_H-384_A-6.zip](https://pan.baidu.com/s/1iIXgZHHCgrYGXVRRSSCVPg)(提取码：gy97)
- [chinese_roformer-char_L-12_H-768_A-12.zip](https://pan.baidu.com/s/1Q1pq8F4Fsl6bTipUAkqeDQ)(提取码：bt94)
- [chinese_roformer-char_L-6_H-384_A-6.zip](https://pan.baidu.com/s/1cc281-M0Rsjlwws5phqzbQ)(提取码：a44c)
- [chinese_roformer-gpt-char_L-12_H-768_A-12.zip](https://pan.baidu.com/s/11YTnWLX0ThQr2P2yW0P7GA)(提取码：2nnn)
- [chinese_roformer-sim-char_L-12_H-768_A-12.zip](https://pan.baidu.com/s/1f1FB288nv1a6jYjsNCordg)(提取码：2cgz)
- [chinese_roformer-sim-char_L-6_H-384_A-6.zip](https://pan.baidu.com/s/1r0eJ7shGwQ0RzV9BTFFW4g)(提取码：h68q)
- [chinese_roformer-sim-char-ft_L-12_H-768_A-12.zip](https://pan.baidu.com/s/1Igh3tSvSu_ahDZmGaOlVoA)(提取码：w15n)
- [chinese_roformer-sim-char-ft_L-6_H-384_A-6.zip](https://pan.baidu.com/s/1G36x7YQF1b6nzW0OzyJS_Q)(提取码：gty5)

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
