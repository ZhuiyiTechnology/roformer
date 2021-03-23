#! -*- coding: utf-8 -*-
# 词级别的中文RoFormer预训练
# MLM任务

import os
os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras

import json
import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.optimizers import extend_with_weight_decay
from bert4keras.optimizers import extend_with_piecewise_linear_lr
from bert4keras.optimizers import extend_with_gradient_accumulation
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator
from bert4keras.snippets import text_segmentate
import jieba
jieba.initialize()

# 基本参数
maxlen = 512
batch_size = 64
epochs = 100000

# bert配置
config_path = '/root/kg/bert/chinese_wobert_plus_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_wobert_plus_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_wobert_plus_L-12_H-768_A-12/vocab.txt'


def corpus():
    """语料生成器
    """
    while True:
        f = '/root/data_pretrain/data_shuf.json'
        with open(f) as f:
            for l in f:
                l = json.loads(l)
                for text in text_process(l['text']):
                    yield text


def text_process(text):
    """分割文本
    """
    texts = text_segmentate(text, 32, u'\n。')
    result, length = '', 0
    for text in texts:
        if result and len(result) + len(text) > maxlen * 1.5:
            yield result
            result, length = '', 0
        result += text
    if result:
        yield result

tokenizer = Tokenizer(
    dict_path,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
)


def random_masking(token_ids):
    """对输入进行随机mask
    """
    rands = np.random.random(len(token_ids))
    source, target = [], []
    for r, t in zip(rands, token_ids):
        if r < 0.15 * 0.8:
            source.append(tokenizer._token_mask_id)
            target.append(t)
        elif r < 0.15 * 0.9:
            source.append(t)
            target.append(t)
        elif r < 0.15:
            source.append(np.random.choice(tokenizer._vocab_size - 1) + 1)
            target.append(t)
        else:
            source.append(t)
            target.append(0)
    return source, target


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        for is_end, text in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            source, target = random_masking(token_ids)
            yield source, segment_ids, target


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = K.sum(accuracy * y_mask) / K.sum(y_mask)
        self.add_metric(accuracy, name='accuracy', aggregation='mean')
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


strategy = tf.distribute.MirroredStrategy()

with strategy.scope():

    bert = build_transformer_model(
        config_path,
        checkpoint_path=None,
        model='roformer',
        with_mlm='linear',
        ignore_invalid_weights=True,
        return_keras_model=False
    )
    model = bert.model

    # 训练用模型
    y_in = keras.layers.Input(shape=(None,), name='Input-Label')
    outputs = CrossEntropy(1)([y_in, model.output])

    train_model = keras.models.Model(model.inputs + [y_in], outputs)

    AdamW = extend_with_weight_decay(Adam, name='AdamW')
    AdamWLR = extend_with_piecewise_linear_lr(AdamW, name='AdamWLR')
    AdamWLRG = extend_with_gradient_accumulation(AdamWLR, name='AdamWLRG')
    optimizer = AdamWLRG(
        learning_rate=1e-5,
        weight_decay_rate=0.01,
        exclude_from_weight_decay=['Norm', 'bias'],
        grad_accum_steps=4,
        lr_schedule={20000: 1}
    )
    train_model.compile(optimizer=optimizer)
    train_model.summary()
    bert.load_weights_from_checkpoint(checkpoint_path)


class Evaluator(keras.callbacks.Callback):
    """训练回调
    """
    def on_epoch_end(self, epoch, logs=None):
        model.save_weights('bert_model.weights')  # 保存模型


if __name__ == '__main__':

    # 启动训练
    evaluator = Evaluator()
    train_generator = data_generator(corpus(), batch_size, 10**5)
    dataset = train_generator.to_dataset(
        types=('float32', 'float32', 'float32'),
        shapes=([None], [None], [None]),
        names=('Input-Token', 'Input-Segment', 'Input-Label'),
        padded_batch=True
    )

    train_model.fit(
        dataset, steps_per_epoch=1000, epochs=epochs, callbacks=[evaluator]
    )

else:

    model.load_weights('bert_model.weights')
