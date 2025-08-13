import numpy as np
import matplotlib.pyplot as plt

def visualize_augmentation_samples(original_data, original_labels, augmented_data, augmented_labels, aug_types, save_path='augmentation_samples.png'):
    """
    データ拡張のサンプル画像を可視化して保存
    シャッフル前のデータを想定（元画像1枚につき4枚の拡張画像が連続して並んでいる）
    
    Args:
        original_data: オリジナルの画像データ
        original_labels: オリジナルのラベル
        augmented_data: 拡張後の画像データ（シャッフル前）
        augmented_labels: 拡張後のラベル
        aug_types: 拡張タイプの配列
        save_path: 保存先のパス
    """
    # 表示設定
    n_samples = 10  # 表示するサンプル数
    n_aug_types = 4  # 拡張タイプ数（Original除く）
    
    # 各クラスから1つずつランダムにサンプルを選択
    sample_indices = []
    for i in range(n_samples):
        # オリジナル画像から選択
        orig_class_indices = np.where(np.argmax(original_labels, axis=1) == i % 15)[0]
        if len(orig_class_indices) > 0:
            sample_indices.append(np.random.choice(orig_class_indices))
    
    # プロット作成
    fig, axes = plt.subplots(n_aug_types + 1, len(sample_indices), figsize=(14, 10))
    
    # 1次元配列の場合に対処
    if len(sample_indices) == 1:
        axes = axes.reshape(-1, 1)
    
    for col_idx, orig_idx in enumerate(sample_indices):
        # オリジナル画像（一番上の行）
        axes[0, col_idx].imshow(original_data[orig_idx, 0], cmap='gray')
        class_id = np.argmax(original_labels[orig_idx])
        axes[0, col_idx].set_title(f'Original\nClass: {class_id}', fontsize=10)
        axes[0, col_idx].axis('off')
        
        # シャッフル前のデータ構造を利用
        # 各元画像に対して4枚の拡張画像が連続して格納されている
        # orig_idx * 4: Original, orig_idx * 4 + 1: Elastic, 
        # orig_idx * 4 + 2: Rotate+Elastic, orig_idx * 4 + 3: Cutout
        aug_base_idx = orig_idx * 4
        
        # 各拡張タイプの画像を表示
        aug_type_list = ['Elastic', 'Rotate+Elastic', 'Cutout', 'Elastic+Cutout']
        type_to_offset = {'Original': 0, 'Elastic': 1, 'Rotate+Elastic': 2, 'Cutout': 3, 'Elastic+Cutout': 3}
        
        for row_idx, aug_type in enumerate(aug_type_list[:n_aug_types], 1):
            if aug_type in type_to_offset:
                # 対応する拡張画像のインデックスを計算
                aug_idx = aug_base_idx + type_to_offset[aug_type]
                if aug_idx < len(augmented_data) and aug_types[aug_idx] == aug_type:
                    axes[row_idx, col_idx].imshow(augmented_data[aug_idx, 0], cmap='gray')
                    axes[row_idx, col_idx].set_title(f'{aug_type}', fontsize=9)
            axes[row_idx, col_idx].axis('off')
    
    plt.suptitle('Data Augmentation Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"拡張サンプル保存: {save_path}")

def visualize_simple_grid(data, labels, aug_types, n_samples=30, save_path='augmentation_grid.png'):
    """
    シンプルなグリッド形式でサンプル画像を表示
    
    Args:
        data: 画像データ
        labels: ラベル
        aug_types: 拡張タイプの配列
        n_samples: 表示するサンプル数
        save_path: 保存先のパス
    """
    sample_idx = np.random.choice(len(data), min(n_samples, len(data)))
    
    n_cols = 5
    n_rows = (len(sample_idx) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.4 * n_rows))
    
    # 1行の場合に対処
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(sample_idx):
        row, col = i // n_cols, i % n_cols
        axes[row, col].imshow(data[idx, 0], cmap='gray')
        class_id = np.argmax(labels[idx])
        axes[row, col].set_title(f'{aug_types[idx]}\nClass: {class_id}', fontsize=8)
        axes[row, col].axis('off')
    
    # 余ったsubplotを非表示
    for i in range(len(sample_idx), n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"グリッドサンプル保存: {save_path}")