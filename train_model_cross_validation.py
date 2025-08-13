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

def train_fold(train_data, train_labels, val_data, val_labels, fold_num):
    """単一foldの訓練を実行"""
    print(f"\n=== Fold {fold_num} 開始 ===")
    
    # データ拡張前のオリジナル画像を保存
    original_train_data = train_data.copy()
    original_train_labels = train_labels.copy()
    
    # データ拡張
    print(f"拡張前: {len(train_data)}枚")
    train_data, train_labels, aug_types = augment_data(train_data, train_labels)
    print(f"拡張後: {len(train_data)}枚")
    
    # 最初のfoldのみ拡張結果を可視化
    if fold_num == 1:
        visualize_augmentation_samples(
            original_train_data, 
            original_train_labels,
            train_data,
            train_labels,
            aug_types,
            save_path=f'augmentation_samples_fold{fold_num}.png'
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
        
        # 10エポックごとに進捗表示
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} - Loss: {loss/(len(train_data)//batch_size):.4f}, Val Acc: {val_acc:.4f}")
        
        # モデル保存（重みのみ）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            with open(f'params_fold{fold_num}.pickle', 'wb') as f:
                pickle.dump(model.params, f)
        else:
            patience += 1
            if patience >= 10:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    print(f"Fold {fold_num} ベスト検証精度: {best_val_acc:.4f}")
    return best_val_acc, model

def main():
    # データ準備
    data = np.load('1_data/train_data.npy').astype(np.float32) / 255.0
    labels = np.load('1_data/train_label.npy')
    
    # 前処理（コントラスト強化）
    print("前処理（コントラスト強化）を適用中...")
    data = preprocess_data(data, enhance_contrast_flag=True)
    print("前処理完了")
    
    # シャッフル
    indices = np.random.permutation(len(data))
    data, labels = data[indices], labels[indices]
    
    # k-fold交差検証の設定
    k = 5
    fold_size = len(data) // k
    fold_scores = []
    
    print(f"\n{k}-fold交差検証を開始します")
    print(f"データ数: {len(data)}, 各foldのサイズ: 約{fold_size}")
    
    # 各foldで訓練と評価
    for fold in range(k):
        # 検証データのインデックス範囲
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < k - 1 else len(data)
        
        # 検証データと訓練データを分割
        val_indices = list(range(val_start, val_end))
        train_indices = list(range(0, val_start)) + list(range(val_end, len(data)))
        
        val_data = data[val_indices]
        val_labels = labels[val_indices]
        train_data = data[train_indices]
        train_labels = labels[train_indices]
        
        # foldの訓練
        val_acc, model = train_fold(train_data, train_labels, val_data, val_labels, fold + 1)
        fold_scores.append(val_acc)
    
    # 結果のサマリー
    print("\n" + "="*50)
    print("交差検証の結果:")
    for i, score in enumerate(fold_scores):
        print(f"  Fold {i+1}: {score:.4f}")
    print(f"\n平均精度: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"最良fold: Fold {np.argmax(fold_scores)+1} ({max(fold_scores):.4f})")
    
    # 最良モデルの重みをメインファイルとして保存
    best_fold = np.argmax(fold_scores) + 1
    with open(f'params_fold{best_fold}.pickle', 'rb') as f:
        best_params = pickle.load(f)
    with open('params.pickle', 'wb') as f:
        pickle.dump(best_params, f)
    print(f"\n最良モデル（Fold {best_fold}）の重みを params.pickle として保存しました")

if __name__ == "__main__":
    main()