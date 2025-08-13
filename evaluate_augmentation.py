import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append('common')

from katakana_model import KatakanaCNN
from data_augmentation import augment_data
from preprocessing import preprocess_data

def main():
    # データ準備
    data = np.load('1_data/train_data.npy').astype(np.float32) / 255.0
    labels = np.load('1_data/train_label.npy')
    
    # 前処理（コントラスト強化）
    print("前処理（コントラスト強化）を適用中...")
    data = preprocess_data(data, enhance_contrast_flag=True)
    print("前処理完了")
    
    # シャッフル・分割（学習データのみ使用）
    indices = np.random.permutation(len(data))
    data, labels = data[indices], labels[indices]
    val_size = int(len(data) * 0.2)
    train_data, train_labels = data[val_size:], labels[val_size:]
    
    # データ拡張前のオリジナル画像を保存
    original_train_data = train_data.copy()
    original_train_labels = train_labels.copy()
    
    # データ拡張（拡張タイプとシャッフル前データを取得）
    print(f"拡張前: {len(train_data)}枚")
    augmented_data, augmented_labels, aug_types, unshuffled = augment_data(
        train_data, train_labels, return_unshuffled=True
    )
    print(f"拡張後: {len(augmented_data)}枚")
    
    # シャッフル前のデータを使用（拡張タイプ情報付き）
    unshuffled_images, unshuffled_labels, unshuffled_types = unshuffled
    
    # モデル読み込み
    model = KatakanaCNN()
    try:
        with open('params.pickle', 'rb') as f:
            params = pickle.load(f)
            model.params = params
            # レイヤーの重みも更新
            model.layers[0].W = params['W1']  # Conv1
            model.layers[0].b = params['b1']
            model.layers[3].W = params['W2']  # Conv2  
            model.layers[3].b = params['b2']
            model.layers[6].W = params['W3']  # Affine1
            model.layers[6].b = params['b3']
            model.layers[8].W = params['W4']  # Affine2
            model.layers[8].b = params['b4']
        print("モデル読み込み完了")
    except FileNotFoundError:
        print("エラー: params.pickle が見つかりません。先にtrain_model.pyを実行してください。")
        return
    
    print("データ拡張画像の評価を開始...")
    print(f"評価データサイズ: {len(unshuffled_images)}枚")
    
    # 予測実行
    predictions = model.predict(unshuffled_images, train_flg=False)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(unshuffled_labels, axis=1)
    
    # 精度計算
    accuracy = np.mean(pred_classes == true_classes)
    print(f"拡張データ全体精度: {accuracy:.4f}")
    
    # 誤判定検出
    misclassified_indices = np.where(pred_classes != true_classes)[0]
    print(f"誤判定数: {len(misclassified_indices)}枚 ({len(misclassified_indices)/len(unshuffled_images)*100:.2f}%)")
    
    if len(misclassified_indices) > 0:
        print("\n誤判定された拡張画像をプロット中...")
        
        # 誤判定画像の可視化
        n_plots = min(len(misclassified_indices), 100)  # 最大100枚まで表示
        n_cols = 10
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2*n_rows))
        
        # axesが1次元の場合に対処
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_plots):
            idx = misclassified_indices[i]
            row, col = i // n_cols, i % n_cols
            
            axes[row, col].imshow(unshuffled_images[idx, 0], cmap='gray')
            true_label = true_classes[idx]
            pred_label = pred_classes[idx]
            aug_type = unshuffled_types[idx]
            axes[row, col].set_title(f'True: {true_label}\nPred: {pred_label}\nAug: {aug_type}', fontsize=8)
            axes[row, col].axis('off')
        
        # 余った subplot を非表示
        for i in range(n_plots, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('augmentation_misclassified_samples.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("誤判定拡張画像を保存: augmentation_misclassified_samples.png")
        
        # 拡張タイプ別の誤判定統計
        print("\n=== 拡張タイプ別誤判定統計 ===")
        aug_type_names = ['original', 'rotation', 'shift', 'contrast', 'noise']
        for aug_type_idx, aug_type_name in enumerate(aug_type_names):
            aug_type_indices = np.where(unshuffled_types == aug_type_idx)[0]
            misclassified_aug = np.intersect1d(aug_type_indices, misclassified_indices)
            if len(aug_type_indices) > 0:
                error_rate = len(misclassified_aug) / len(aug_type_indices) * 100
                print(f"{aug_type_name:10s}: {len(misclassified_aug):4d}/{len(aug_type_indices):4d}枚 ({error_rate:.1f}%)")
        
        # クラス別の誤判定統計
        print("\n=== クラス別誤判定統計 ===")
        for i in range(15):
            true_i_indices = np.where(true_classes == i)[0]
            misclassified_i = np.intersect1d(true_i_indices, misclassified_indices)
            if len(true_i_indices) > 0:
                error_rate = len(misclassified_i) / len(true_i_indices) * 100
                print(f"Class {i:2d}: {len(misclassified_i):3d}/{len(true_i_indices):3d}枚 ({error_rate:.1f}%)")
        
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
        
        # 拡張タイプ別の誤判定パターン分析
        print("\n=== 拡張タイプ別誤判定パターン詳細 ===")
        for aug_type_idx, aug_type_name in enumerate(aug_type_names):
            aug_misclassified = []
            for idx in misclassified_indices:
                if unshuffled_types[idx] == aug_type_idx:
                    aug_misclassified.append(idx)
            
            if len(aug_misclassified) > 0:
                print(f"\n{aug_type_name} での誤判定:")
                aug_confusion = {}
                for idx in aug_misclassified:
                    true_class = true_classes[idx]
                    pred_class = pred_classes[idx]
                    key = (true_class, pred_class)
                    aug_confusion[key] = aug_confusion.get(key, 0) + 1
                
                sorted_aug_confusion = sorted(aug_confusion.items(), key=lambda x: x[1], reverse=True)
                for (true_label, pred_label), count in sorted_aug_confusion[:5]:
                    print(f"  {true_label:2d} → {pred_label:2d}: {count}回")
    
    else:
        print("拡張データで誤判定はありませんでした！")

if __name__ == "__main__":
    main()