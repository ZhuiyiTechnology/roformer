#! -*- coding: utf-8 -*-
# RoFormer-GPT 模型测试

import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import AutoRegressiveDecoder
from bert4keras.snippets import uniout

config_path = '/root/kg/bert/chinese_roformer-gpt-char_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_roformer-gpt-char_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_roformer-gpt-char_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='roformer',
    application='lm',
)  # 建立模型，加载权重


class ArticleCompletion(AutoRegressiveDecoder):
    """基于随机采样的文章续写
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids = np.concatenate([inputs[0], output_ids], 1)
        segment_ids = np.zeros_like(token_ids)
        return self.last_token(model).predict([token_ids, segment_ids])

    def generate(self, text, n=1, topp=0.95):
        token_ids = tokenizer.encode(text)[0][:-1]
        results = self.random_sample([token_ids], n, topp=topp)  # 基于随机采样
        return [text + tokenizer.decode(ids) for ids in results]


article_completion = ArticleCompletion(
    start_id=None, end_id=tokenizer.token_to_id(u'。'), maxlen=256, minlen=128
)

print(article_completion.generate(u'今天天气不错'))
