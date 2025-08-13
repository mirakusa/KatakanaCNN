import numpy as np
import sys
import os
sys.path.append('common')

from layers import *
from optimizer import *
from loss import *
import pickle

class KatakanaCNN:
   """
   カタカナ認識用のCNNモデル
   """
   def __init__(self):
       filter_num = 32
       filter_size = 5
       filter_num2 = 64
       filter_size2 = 3
       hidden_size = 256
       output_size = 15
       
       # 重みの初期化（He）
       self.params = {}
       self.params['W1'] = np.random.randn(filter_num, 1, filter_size, filter_size) * np.sqrt(2.0 / (1 * filter_size * filter_size))
       self.params['b1'] = np.zeros(filter_num)
       
       self.params['W2'] = np.random.randn(filter_num2, filter_num, filter_size2, filter_size2) * np.sqrt(2.0 / (filter_num * filter_size2 * filter_size2))
       self.params['b2'] = np.zeros(filter_num2)
       
       # Conv->Pool後のサイズを計算
       # 28 -> 28(conv, pad=2) -> 14(pool) -> 12(conv) -> 6(pool)
       conv_output_size = filter_num2 * 6 * 6
       
       self.params['W3'] = np.random.randn(conv_output_size, hidden_size) * np.sqrt(2.0 / conv_output_size)
       self.params['b3'] = np.zeros(hidden_size)
       
       self.params['W4'] = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
       self.params['b4'] = np.zeros(output_size)
       
       # レイヤーの生成
       self.layers = []
       self.layers.append(Convolution(self.params['W1'], self.params['b1'], stride=1, pad=2))
       self.layers.append(ReLU())
       self.layers.append(MaxPooling(pool_h=2, pool_w=2, stride=2))
       
       self.layers.append(Convolution(self.params['W2'], self.params['b2'], stride=1, pad=0))
       self.layers.append(ReLU())
       self.layers.append(MaxPooling(pool_h=2, pool_w=2, stride=2))
       
       self.layers.append(Affine(self.params['W3'], self.params['b3']))
       self.layers.append(ReLU())
       self.layers.append(Dropout(0.3))
       
       self.layers.append(Affine(self.params['W4'], self.params['b4']))
       
       self.last_layer = SoftmaxWithLoss()
       
   def predict(self, x, train_flg=False):
       # Dropoutのtrain_flgを設定
       for layer in self.layers:
           if isinstance(layer, Dropout):
               layer.train_flg = train_flg
           x = layer.forward(x)
       return x
   
   def loss(self, x, t):
       y = self.predict(x, train_flg=True)
       return self.last_layer.forward(y, t)
   
   def accuracy(self, x, t):
       y = self.predict(x, train_flg=False)
       y = np.argmax(y, axis=1)
       if t.ndim == 2:  # one-hot
           t = np.argmax(t, axis=1)
       accuracy = np.mean(y == t)
       return accuracy
   
   def gradient(self, x, t):
       # forward
       self.loss(x, t)
       
       # backward
       dout = 1
       dout = self.last_layer.backward(dout)
       
       for layer in reversed(self.layers):
           dout = layer.backward(dout)
       
       # 勾配の設定
       grads = {}
       grads['W1'] = self.layers[0].dW
       grads['b1'] = self.layers[0].db
       grads['W2'] = self.layers[3].dW
       grads['b2'] = self.layers[3].db
       grads['W3'] = self.layers[6].dW
       grads['b3'] = self.layers[6].db
       grads['W4'] = self.layers[9].dW
       grads['b4'] = self.layers[9].db
       
       return grads