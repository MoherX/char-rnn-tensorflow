# coding=utf-8
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import numpy as np


class Model():
    def __init__(self, args, training=True):
        self.args = args
        # 在测试状态下（inference）才用如下选项
        if not training:
            args.batch_size = 1
            args.seq_length = 1
        # 几种备选的rnn类型
        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))
        # multiple-layer,cells=[basic.cell,..,]
        # add layers
        cells = []
        # 固定格式，有几层rnn:num_layers
        for _ in range(args.num_layers):
            # 固定格式是例：cell = rnn_cell.GRUCelll(rnn_size)
            # rnn_size指的是每个rnn单元中的神经元个数（虽然RNN途中只有一个圆圈代表，但这个圆圈代表了rnn_size个神经元）
            cell = cell_fn(args.rnn_size)
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=args.input_keep_prob,
                                          output_keep_prob=args.output_keep_prob)
            cells.append(cell)
        # 这里state_is_tuple根据官网解释，每个cell返回的h和c状态是储存在一个list里还是两个tuple里，官网建议设置为true
        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)
        # #input_data&target(标签)格式：[batch_size, seq_length]
        self.input_data = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        # cell的初始状态设为0，因为在前面设置cell时，cell_size已经设置好了，因此这里只需给出batch_size即可
        # （一个batch内有batch_size个sequence的输入）
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        # rnnlm = recurrent neural network language model
        # variable_scope就是变量的作用域
        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w",
                                        [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
        # embedding矩阵是将输入转换到了cell_size，因此这样的大小设置:词向量矩阵的维度应该是 vocab_size * rnn_size
        # 即每一行代表一个词，列数就是我们需要自己定义的词向量维度
        embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
        # 关于tf.nn.embedding_lookup(embedding, self.input_data)：
        # 调用tf.nn.embedding_lookup，索引与train_dataset对应的向量，相当于用train_dataset作为一个id，去检索矩阵中与这个id对应的embedding
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        # embeddinglookup得到的look_up尺寸是[batch_size, seq_length, rnn_size]，这里是[50,50,128]
        # dropout beta testing: double check which one should affect next line
        if training and args.output_keep_prob:
            inputs = tf.nn.dropout(inputs, args.output_keep_prob)
        # 将第三个参数，在第1维度，切成seq_length长短的片段
        # 将上面的[50,50,128]切开，得到50个[50,1,128]的inputs
        inputs = tf.split(inputs, args.seq_length, 1)
        # https://www.tensorflow.org/api_docs/python/tf/squeeze
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        # 之后将 1 squeeze掉,50个[50,128]

        # 在infer的时候方便查看
        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        # seq2seq.rnn_decoder基于schedule sampling实现，相当于一个黑盒子，可以直接调用
        # 该函数实现了一个简单的多层rnn模型。上面的MultiRNNCell函数构造了一个时间步的多层rnn，本函数则实现将其循环num_steps个时间步
        # https://www.tensorflow.org/api_docs/python/tf/contrib/legacy_seq2seq/rnn_decoder
        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')
        # 得到的两个参数shape均为50个50*128的张量，和输入是一样的
        # tf.concat()转换后的outputs的shape为(batch_size * sequence_size, rnn_size)
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])
        # 将outputsreshape在一起，形成[2500,128]的张量
        # logits和probs的大小都是[2500,65]([2500,128]*[128,65])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        # 得到length为2500的loss（即每一个batch的sequence中的每一个单词输入，都会最终产生一个loss，50*50=2500）
        loss = legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])])
        # 得到一个batch的cost后面用于求梯度
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        # 将state转换一下，便于下一次继续训练
        self.final_state = last_state
        # 具体的learning_rate是由train.py中args参数传过来的，这里只是初始化设了一个0
        self.lr = tf.Variable(0.0, trainable=False)
        # 返回了包括前面的softmax_w/softmax_b/embedding等所有变量
        tvars = tf.trainable_variables()
        # 求grads要使用clip避免梯度爆炸，这里设置的阈值是5（见args）
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        # 使用adam优化方法
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
        # 将计算出的梯度应用到变量上，是函数minimize()的第二部分，返回一个应用指定的梯度的操作Operation，对global_step做自增操作
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # instrument tensorboard
        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)
        # weight = [0.1,0.2,0.3,0.4]
        # (分布函数)t = [0.1,0.3,0.6,1]
        # s = 1
        # 为什么这样pick还不是太懂
        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            # 注意！！这里的probs是长度是1*65的，前面在训练的时候因为batch_size和seq_length都是50
            # 所以是2500*65之后用了这2500组预测结果来求loss，再BPTT，
            # 这里只是根据一个输入求一个输出，batch_size和seq_length都是1，因此是1*65
            # 所以p就是代表了由长度为65的一个数组，每一位代表着预测为该位的概率值
            p = probs[0]
            print probs[0]
            if sampling_type == 0:
                # 第一种方法，直接取最大的prob的索引值
                sample = np.argmax(p)
            elif sampling_type == 2:
                # 第二种方法，如果输入是空格，则wighted_pick
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    # 否则取最大prob的索引
                    sample = np.argmax(p)
            else:  # sampling_type == 1 default:
                # 一直使用weighted_pick方法
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret
