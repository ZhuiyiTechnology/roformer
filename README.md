[[中文](https://github.com/ZhuiyiTechnology/roformer/blob/main/README_zh.md)|[English](https://github.com/ZhuiyiTechnology/roformer/blob/main/README.md)]

# Rotary Transformer

Rotary Transformer is an MLM pre-trained language model with rotary position embedding (RoPE). The RoPE is a relative position encoding method with promise theoretical properties. The main idea is to multiply the context embeddings (q,k in the Transformer) by rotation matrices depending on the absolute position.  One can prove that the inner product of the context embeddings will become only depending on the relative position. EleutherAI also posted a [blog](https://blog.eleuther.ai/rotary-embeddings/) that contains an intuitive explanation and experiments about RoPE.

To the best of our knowledge, RoPE is the only relative position embeddings that can be used in linear attentions.  For more details, please refer to our paper or the [original Blog (Chinese)](https://kexue.fm/archives/8265). EleutherAI also posted a [blog](https://blog.eleuther.ai/rotary-embeddings/) that contains an intuitive explanation and experiments of using RoPE in various models.

## Dependency

`bert4keras 0.10.4`

## Implementation

You can implement the RoPE with a few lines of changes in the self-attention layer. Here we provide the pseudo code for instruction.

```python
sinusoidal_pos.shape = [1, seq_len, hidden_size] # Sinusoidal position embeddings
qw.shape = [batch_size, seq_len, num_heads, hidden_size]  # query hiddens
kw.shape = [batch_size, seq_len, num_heads, hidden_size]  # key hiddens

cos_pos = repeat_elements(sinusoidal_pos[..., None, 1::2], rep=2, axis=-1)
sin_pos = repeat_elements(sinusoidal_pos[..., None, ::2], rep=2, axis=-1)
qw2 = stack([-qw[..., 1::2], qw[..., ::2]], 4)
qw2 = reshape(qw2, shape(qw))
qw = qw * cos_pos + qw2 * sin_pos
kw2 = K.stack([-kw[..., 1::2], kw[..., ::2]], 4)
kw2 = K.reshape(kw2, K.shape(kw))
kw = kw * cos_pos + kw2 * sin_pos

# Attention
a = tf.einsum('bjhd,bkhd->bhjk', qw, kw)
```

Or you can find the implementation in source code of [bert4keras](https://github.com/bojone/bert4keras).

## Download

- [chinese_roformer_L-12_H-768_A-12.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer_L-12_H-768_A-12.zip)
- [chinese_roformer_L-6_H-384_A-6.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer_L-6_H-384_A-6.zip)
- [chinese_roformer-char_L-12_H-768_A-12.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-char_L-12_H-768_A-12.zip)
- [chinese_roformer-char_L-6_H-384_A-6.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-char_L-6_H-384_A-6.zip)
- [chinese_roformer-gpt-char_L-12_H-768_A-12.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-gpt-char_L-12_H-768_A-12.zip)
- [chinese_roformer-sim-char_L-12_H-768_A-12.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-sim-char_L-12_H-768_A-12.zip)
- [chinese_roformer-sim-char_L-6_H-384_A-6.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-sim-char_L-6_H-384_A-6.zip)
- [chinese_roformer-sim-char-ft_L-12_H-768_A-12.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-sim-char-ft_L-12_H-768_A-12.zip)
- [chinese_roformer-sim-char-ft_L-6_H-384_A-6.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-sim-char-ft_L-6_H-384_A-6.zip)

## Other Implementations

- A pytorch implementation can be found [here](https://github.com/JunnYu/RoFormer_pytorch)

- [x-transformer](https://github.com/lucidrains/x-transformers), [GPT-Neo](https://github.com/EleutherAI/gpt-neo), [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) and [mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax) by EleutherAI

##  Citation

Bibtex:

```tex

@misc{su2021roformer,
      title={RoFormer: Enhanced Transformer with Rotary Position Embedding}, 
      author={Jianlin Su and Yu Lu and Shengfeng Pan and Bo Wen and Yunfeng Liu},
      year={2021},
      eprint={2104.09864},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

```



