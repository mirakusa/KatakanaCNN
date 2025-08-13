import numpy as np
import glob
import os,sys
import util
import pickle
from katakana_model import KatakanaCNN

def makedataset():
    """
    データセットをつくる関数です。
    自由に編集してください。
    """
    
    # 次の行は変更しないこと
    test_data= util.loaddata()
    
    # 以下は自由に編集しても構いません
    # 必要な前処理をここに記述してください  
    
    # 正規化（学習時と同じ方法）
    test_data = test_data.astype('float32') / 255.0

    return test_data


def func_predict(test_data, test_label):
    """
    予測する関数
    data : 画像データ
    return loss, accuracy
    引数とreturn以外は、自由に編集してください    
    """
    
    # 以下を自由に編集してください
    model = KatakanaCNN()
    with open("params.pickle", "rb") as f:
        params = pickle.load(f)
        model.params = params
        # レイヤーの重みも更新
        model.layers[0].W = params['W1']
        model.layers[0].b = params['b1']
        model.layers[3].W = params['W2']
        model.layers[3].b = params['b2']
        model.layers[6].W = params['W3']
        model.layers[6].b = params['b3']
        model.layers[9].W = params['W4']
        model.layers[9].b = params['b4']
    
    accuracy = model.accuracy(test_data, test_label)
    loss  = model.loss(test_data, test_label)
    
    return loss, accuracy # 編集不可


def main():
    """
    編集しないでください。
    """
    # テスト用データをつくる
    test_data = makedataset()

    # 予測し精度を算出する
    util.accuracy(func_predict, test_data)
    
    return


if __name__=="__main__":
    main()