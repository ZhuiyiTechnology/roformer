#! -*- coding:utf-8 -*-
# 句子对分类任务
# 数据集：https://arxiv.org/abs/1911.08962

import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import GlobalAveragePooling1D, Dense
import jieba
jieba.initialize()

maxlen = 1024
batch_size = 8
config_path = '/root/kg/bert/chinese_roformer_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_roformer_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_roformer_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：(文本A, 文本B, 文本C)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            assert l['label'] in 'BC'
            if l['label'] == 'B':
                D.append((l['A'], l['B'], l['C']))
            else:
                D.append((l['A'], l['C'], l['B']))
    return D


# 加载数据集
train_data = load_data('CAIL2019-SCM/train.json')
valid_data = load_data('CAIL2019-SCM/valid.json')
test_data = load_data('CAIL2019-SCM/test.json')

# 建立分词器
tokenizer = Tokenizer(
    dict_path,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, text3) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                text1, text2, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([1])
            token_ids, segment_ids = tokenizer.encode(
                text1, text3, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([0])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
model = build_transformer_model(
    config_path=config_path, checkpoint_path=checkpoint_path, model='roformer'
)

output = GlobalAveragePooling1D()(model.output)
output = Dense(units=1, activation='sigmoid')(output)

model = keras.models.Model(model.input, output)
model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(6e-6),
    metrics=['accuracy'],
)

# 转换数据集
train_generator = data_generator(train_data, batch_size // 2)
valid_generator = data_generator(valid_data, batch_size // 2)
test_generator = data_generator(test_data, batch_size // 2)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true)[:, 0]
        total += len(y_pred) // 2
        right += (y_pred[::2] > y_pred[1::2]).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model_scm.weights')
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=20,
        callbacks=[evaluator]
    )

    model.load_weights('best_model_scm.weights')
    print(u'final test acc: %05f\n' % (evaluate(test_generator)))

else:

    model.load_weights('best_model_scm.weights')
