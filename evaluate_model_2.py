import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append('common')

from katakana_model import KatakanaCNN
from katakana_model2 import KatakanaCNN2
from preprocessing import preprocess_data

def main():
    # データ読み込み
    data = np.load('1_data/train_data.npy').astype(np.float32) / 255.0
    labels = np.load('1_data/train_label.npy')
    
    # 前処理（コントラスト強化）
    print("前処理（コントラスト強化）を適用中...")
    data = preprocess_data(data, enhance_contrast_flag=True)
    print("前処理完了")
    
    # モデル読み込み
    model = KatakanaCNN2()
    with open('params2.pickle', 'rb') as f:
        params = pickle.load(f)
        model.params = params
        # レイヤーの重みも更新（KatakanaCNN2用のインデックス）
        model.layers[0].W = params['W1']  # Conv1
        model.layers[0].b = params['b1']
        model.layers[3].W = params['W2']  # Conv2
        model.layers[3].b = params['b2']
        model.layers[6].W = params['W3']  # Conv3
        model.layers[6].b = params['b3']
        model.layers[9].W = params['W4']  # Affine1
        model.layers[9].b = params['b4']
        model.layers[12].W = params['W5'] # Affine2
        model.layers[12].b = params['b5']
    
    print("全学習データの評価を開始...")
    print(f"データサイズ: {len(data)}枚")
    
    # 予測実行
    predictions = model.predict(data, train_flg=False)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)
    
    # 精度計算
    accuracy = np.mean(pred_classes == true_classes)
    print(f"全体精度: {accuracy:.4f}")
    
    # 誤判定検出
    misclassified_indices = np.where(pred_classes != true_classes)[0]
    print(f"誤判定数: {len(misclassified_indices)}枚 ({len(misclassified_indices)/len(data)*100:.2f}%)")
    
    # クラス番号（0-14）
    
    if len(misclassified_indices) > 0:
        print("\n誤判定された画像をプロット中...")
        
        # 誤判定画像の可視化
        n_plots = min(len(misclassified_indices), 100)  # 最大50枚まで表示
        n_cols = 10
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2*n_rows))
        
        # axesが1次元の場合に対処
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_plots):
            idx = misclassified_indices[i]
            row, col = i // n_cols, i % n_cols
            
            axes[row, col].imshow(data[idx, 0], cmap='gray')
            true_label = true_classes[idx]
            pred_label = pred_classes[idx]
            axes[row, col].set_title(f'True: {true_label}\nPred: {pred_label}', fontsize=8)
            axes[row, col].axis('off')
        
        # 余った subplot を非表示
        for i in range(n_plots, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('misclassified_samples.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("誤判定画像を保存: misclassified_samples.png")
        
        # クラス別の誤判定統計
        print("\n=== クラス別誤判定統計 ===")
        for i in range(15):
            true_i_indices = np.where(true_classes == i)[0]
            misclassified_i = np.intersect1d(true_i_indices, misclassified_indices)
            if len(true_i_indices) > 0:
                error_rate = len(misclassified_i) / len(true_i_indices) * 100
                print(f"Class {i:2d}: {len(misclassified_i)}/{len(true_i_indices)}枚 ({error_rate:.1f}%)")
        
        # 混同行列風の統計（誤判定のみ）
        print("\n=== 主な誤判定パターン ===")
        confusion_count = {}
        for idx in misclassified_indices:
            true_class = true_classes[idx]
            pred_class = pred_classes[idx]
            key = (true_class, pred_class)
            confusion_count[key] = confusion_count.get(key, 0) + 1
        
        # 多い順にソート
        sorted_confusion = sorted(confusion_count.items(), key=lambda x: x[1], reverse=True)
        for (true_label, pred_label), count in sorted_confusion[:10]:
            print(f"{true_label:2d} → {pred_label:2d}: {count}回")
    
    else:
        print("誤判定はありませんでした！")

if __name__ == "__main__":
    main()