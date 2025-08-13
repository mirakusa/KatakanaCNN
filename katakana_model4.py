import numpy as np
import sys
import os
sys.path.append('common')

from layers import *
from optimizer import *
from loss import *
import pickle

class KatakanaCNN4:
    """
    カタカナ認識用のCNNモデル（3層、第1層フィルタサイズ5、第2,3層フィルタサイズ3）
    """
    def __init__(self):
        # ネットワーク構造の定義
        filter_num1 = 32    # 第1層のフィルタ数
        filter_num2 = 64    # 第2層のフィルタ数
        filter_num3 = 128   # 第3層のフィルタ数
        filter_size1 = 5    # 第1層のフィルタサイズ
        filter_size2 = 3    # 第2,3層のフィルタサイズ
        hidden_size = 256   # 全結合層のユニット数
        output_size = 15    # 出力クラス数（カタカナ15種類）
        
        # 重みの初期化（He初期化）
        self.params = {}
        # 第1層: Conv1
        self.params['W1'] = np.random.randn(filter_num1, 1, filter_size1, filter_size1) * np.sqrt(2.0 / (1 * filter_size1 * filter_size1))
        self.params['b1'] = np.zeros(filter_num1)
        
        # 第2層: Conv2
        self.params['W2'] = np.random.randn(filter_num2, filter_num1, filter_size2, filter_size2) * np.sqrt(2.0 / (filter_num1 * filter_size2 * filter_size2))
        self.params['b2'] = np.zeros(filter_num2)
        
        # 第3層: Conv3
        self.params['W3'] = np.random.randn(filter_num3, filter_num2, filter_size2, filter_size2) * np.sqrt(2.0 / (filter_num2 * filter_size2 * filter_size2))
        self.params['b3'] = np.zeros(filter_num3)
        
        # Conv->Pool後のサイズを計算
        # 入力: 28x28
        # Conv1(5x5, pad=2) -> 28x28 -> Pool(2x2) -> 14x14
        # Conv2(3x3, pad=1) -> 14x14 -> Pool(2x2) -> 7x7
        # Conv3(3x3, pad=1) -> 7x7 -> Pool(2x2) -> 3x3
        conv_output_size = filter_num3 * 3 * 3
        
        # 第4層: 全結合層1
        self.params['W4'] = np.random.randn(conv_output_size, hidden_size) * np.sqrt(2.0 / conv_output_size)
        self.params['b4'] = np.zeros(hidden_size)
        
        # 第5層: 全結合層2（出力層）
        self.params['W5'] = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.params['b5'] = np.zeros(output_size)
        
        # レイヤーの生成
        self.layers = []
        
        # 第1畳み込み層ブロック
        self.layers.append(Convolution(self.params['W1'], self.params['b1'], stride=1, pad=2))
        self.layers.append(ReLU())
        self.layers.append(MaxPooling(pool_h=2, pool_w=2, stride=2))
        
        # 第2畳み込み層ブロック
        self.layers.append(Convolution(self.params['W2'], self.params['b2'], stride=1, pad=1))
        self.layers.append(ReLU())
        self.layers.append(MaxPooling(pool_h=2, pool_w=2, stride=2))
        
        # 第3畳み込み層ブロック
        self.layers.append(Convolution(self.params['W3'], self.params['b3'], stride=1, pad=1))
        self.layers.append(ReLU())
        self.layers.append(MaxPooling(pool_h=2, pool_w=2, stride=2))
        
        # 全結合層ブロック
        self.layers.append(Affine(self.params['W4'], self.params['b4']))
        self.layers.append(ReLU())
        self.layers.append(Dropout(0.5))
        
        # 出力層
        self.layers.append(Affine(self.params['W5'], self.params['b5']))
        
        self.last_layer = SoftmaxWithLoss()
        
    def predict(self, x, train_flg=False):
        # Dropout のtrain_flgを設定
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
        # Conv1の勾配（層インデックス: 0）
        grads['W1'] = self.layers[0].dW
        grads['b1'] = self.layers[0].db
        
        # Conv2の勾配（層インデックス: 3）
        grads['W2'] = self.layers[3].dW
        grads['b2'] = self.layers[3].db
        
        # Conv3の勾配（層インデックス: 6）
        grads['W3'] = self.layers[6].dW
        grads['b3'] = self.layers[6].db
        
        # Affine1の勾配（層インデックス: 9）
        grads['W4'] = self.layers[9].dW
        grads['b4'] = self.layers[9].db
        
        # Affine2の勾配（層インデックス: 12）
        grads['W5'] = self.layers[12].dW
        grads['b5'] = self.layers[12].db
        
        return grads