import numpy as np
import cv2
import random


def cutout(image, n_holes=1, length=8):
    """Cutoutデータ拡張
    画像にランダムな正方形のマスクを適用する
    Args:
        image: 入力画像 (H, W)または(C, H, W)
        n_holes: マスクの数
        length: マスクの一辺の長さ
    """
    img = image.copy()
    
    # 画像の形状を取得
    if img.ndim == 3:
        _, h, w = img.shape
    else:
        h, w = img.shape
    
    for _ in range(n_holes):
        # マスクの中心座標をランダムに選択
        y = np.random.randint(h)
        x = np.random.randint(w)
        
        # マスク領域の計算
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        
        # マスク適用（1.0（白）で埋める）
        if img.ndim == 3:
            img[:, y1:y2, x1:x2] = 1.0
        else:
            img[y1:y2, x1:x2] = 1.0
    
    return img

def elastic_transform(image):
    """弾性変形"""
    image = np.array(image)
    shape = image.shape
    
    # パラメータをランダムに設定
    alpha = random.randint(60, 80)
    sigma = random.choice([12, 13, 14])
    
    # ランダムな変位フィールドを生成（-1から1の範囲）
    dx = (np.random.random(shape[:2]) * 2 - 1) * alpha
    dy = (np.random.random(shape[:2]) * 2 - 1) * alpha
    
    # OpenCVでガウシアンフィルタを適用
    ksize = int(sigma * 6 + 1)
    if ksize % 2 == 0:
        ksize += 1
    
    dx = cv2.GaussianBlur(dx.astype(np.float32), (ksize, ksize), sigma)
    dy = cv2.GaussianBlur(dy.astype(np.float32), (ksize, ksize), sigma)
    
    # 座標グリッドを作成
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    
    # 変位を適用（reflectモードのため、clipではなくミラーリング）
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    
    # cv2.remapで変形を適用（BORDER_REFLECTでreflectモードを再現）
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def augment_data(images, labels, return_unshuffled=False):
    """データ拡張
    
    Args:
        images: 画像データの配列
        labels: ラベルの配列
        return_unshuffled: Trueの場合、シャッフル前のデータも返す
    
    Returns:
        aug_images: 拡張後の画像
        aug_labels: 拡張後のラベル
        aug_types: 拡張タイプの配列
        (return_unshuffled=Trueの場合) unshuffled_data: シャッフル前のデータのタプル
    """
    aug_images, aug_labels, aug_types = [], [], []
    
    for img, lbl in zip(images, labels):
        # 元画像
        aug_images.append(img)
        aug_labels.append(lbl)
        aug_types.append('Original')
        
        # 弾性変形
        aug_images.append(elastic_transform(img[0]).reshape(1, 28, 28))
        aug_labels.append(lbl)
        aug_types.append('Elastic')
        
        # 回転＋弾性変形
        angle = np.random.uniform(-5, 5)
        M = cv2.getRotationMatrix2D((14, 14), angle, 0.9)
        rotated = cv2.warpAffine(img[0], M, (28, 28), borderMode=cv2.BORDER_REFLECT)
        aug_images.append(elastic_transform(rotated).reshape(1, 28, 28))
        aug_labels.append(lbl)
        aug_types.append('Rotate+Elastic')
        
        # Cutout
        cutout_img = cutout(img, n_holes=1, length=np.random.randint(6, 10))
        aug_images.append(cutout_img)
        aug_labels.append(lbl)
        aug_types.append('Cutout')
    
    # 配列に変換
    aug_images = np.array(aug_images)
    aug_labels = np.array(aug_labels)
    aug_types = np.array(aug_types)
    
    # シャッフル前のデータを保存
    if return_unshuffled:
        unshuffled_data = (aug_images.copy(), aug_labels.copy(), aug_types.copy())
    
    # シャッフル
    indices = np.random.permutation(len(aug_images))
    shuffled_images = aug_images[indices]
    shuffled_labels = aug_labels[indices]
    shuffled_types = aug_types[indices]
    
    if return_unshuffled:
        return shuffled_images, shuffled_labels, shuffled_types, unshuffled_data
    else:
        return shuffled_images, shuffled_labels, shuffled_types

def augment_data_advanced(images, labels, augmentation_config=None):
    """拡張可能なデータ拡張関数
    
    Args:
        images: 画像データの配列
        labels: ラベルの配列
        augmentation_config: 拡張設定の辞書
    
    Returns:
        aug_images: 拡張後の画像
        aug_labels: 拡張後のラベル
        aug_types: 拡張タイプの配列
    """
    if augmentation_config is None:
        # デフォルト設定
        augmentation_config = {
            'elastic': True,
            'rotate_elastic': True,
            'cutout': True,
            'elastic_cutout': False,
            'rotate_angle_range': (-5, 5),
            'cutout_length_range': (6, 10),
            'elastic_alpha_range': (30, 50),
            'elastic_sigma_choices': [8, 9, 10]
        }
    
    aug_images, aug_labels, aug_types = [], [], []
    
    for img, lbl in zip(images, labels):
        # 元画像
        aug_images.append(img)
        aug_labels.append(lbl)
        aug_types.append('Original')
        
        # 弾性変形
        if augmentation_config.get('elastic', True):
            aug_images.append(elastic_transform(img[0]).reshape(1, 28, 28))
            aug_labels.append(lbl)
            aug_types.append('Elastic')
        
        # 回転＋弾性変形
        if augmentation_config.get('rotate_elastic', True):
            angle_range = augmentation_config.get('rotate_angle_range', (-5, 5))
            angle = np.random.uniform(angle_range[0], angle_range[1])
            M = cv2.getRotationMatrix2D((14, 14), angle, 0.9)
            rotated = cv2.warpAffine(img[0], M, (28, 28), borderMode=cv2.BORDER_REFLECT)
            aug_images.append(elastic_transform(rotated).reshape(1, 28, 28))
            aug_labels.append(lbl)
            aug_types.append('Rotate+Elastic')
        
        # Cutout
        if augmentation_config.get('cutout', True):
            length_range = augmentation_config.get('cutout_length_range', (6, 10))
            cutout_img = cutout(img, n_holes=1, length=np.random.randint(length_range[0], length_range[1]))
            aug_images.append(cutout_img)
            aug_labels.append(lbl)
            aug_types.append('Cutout')
        
        # 弾性変形 + Cutout
        if augmentation_config.get('elastic_cutout', False):
            elastic_img = elastic_transform(img[0]).reshape(1, 28, 28)
            cutout_elastic = cutout(elastic_img, n_holes=1, length=np.random.randint(5, 8))
            aug_images.append(cutout_elastic)
            aug_labels.append(lbl)
            aug_types.append('Elastic+Cutout')
    
    # シャッフル
    aug_images = np.array(aug_images)
    aug_labels = np.array(aug_labels)
    aug_types = np.array(aug_types)
    indices = np.random.permutation(len(aug_images))
    return aug_images[indices], aug_labels[indices], aug_types[indices]