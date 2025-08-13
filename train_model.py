import numpy as np
import sys
import pickle
sys.path.append('common')

from optimizer import Adam
from katakana_model import KatakanaCNN
from katakana_model2 import KatakanaCNN2
from data_augmentation import augment_data
from visualize_augmentation import visualize_augmentation_samples
from preprocessing import preprocess_data

def main():
    # データ準備
    data = np.load('1_data/train_data.npy').astype(np.float32) / 255.0
    labels = np.load('1_data/train_label.npy')
    
    # 前処理（コントラスト強化）
    print("前処理（コントラスト強化）を適用中...")
    data = preprocess_data(data, enhance_contrast_flag=True)
    print("前処理完了")
    
    # シャッフル・分割
    indices = np.random.permutation(len(data))
    data, labels = data[indices], labels[indices]
    val_size = int(len(data) * 0.2)
    val_data, val_labels = data[:val_size], labels[:val_size]
    train_data, train_labels = data[val_size:], labels[val_size:]
    
    # データ拡張前のオリジナル画像を保存
    original_train_data = train_data.copy()
    original_train_labels = train_labels.copy()
    
    # データ拡張
    print(f"拡張前: {len(train_data)}枚")
    train_data, train_labels, aug_types = augment_data(train_data, train_labels)
    print(f"拡張後: {len(train_data)}枚")
    
    # 拡張結果サンプル表示（外部関数を呼び出し）
    visualize_augmentation_samples(
        original_train_data, 
        original_train_labels,
        train_data,
        train_labels,
        aug_types,
        save_path='augmentation_samples.png'
    )
    print()
    
    # モデル・最適化設定
    model = KatakanaCNN()
    optimizer = Adam(lr=0.001)
    batch_size = 128
    best_val_acc = 0
    patience = 0
    
    # 訓練
    for epoch in range(100):
        # 学習率減衰
        if epoch in [30, 60, 80]:
            optimizer.lr *= 0.5
        
        # ミニバッチ学習
        loss = 0
        for _ in range(len(train_data) // batch_size):
            idx = np.random.choice(len(train_data), batch_size)
            grads = model.gradient(train_data[idx], train_labels[idx])
            optimizer.update(model.params, grads)
            loss += model.loss(train_data[idx], train_labels[idx])
        
        # 評価
        val_acc = model.accuracy(val_data, val_labels)
        print(f"Epoch {epoch+1:3d} - Loss: {loss/(len(train_data)//batch_size):.4f}, Val Acc: {val_acc:.4f}")
        
        # モデル保存（重みのみ）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            with open('params.pickle', 'wb') as f:
                pickle.dump(model.params, f)
        else:
            patience += 1
            if patience >= 10:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # 最終評価
    print(f"\nベスト検証精度: {best_val_acc:.4f}")
    all_data = np.load('1_data/train_data.npy').astype(np.float32) / 255.0
    print(f"全データ精度: {model.accuracy(all_data, np.load('1_data/train_label.npy')):.4f}")

if __name__ == "__main__":
    main()