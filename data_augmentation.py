import numpy as np
import cv2


def cutout(image, n_holes=1, length=8):
    """画像にランダムな正方形マスクを適用"""
    img = image.copy()
    h, w = img.shape[-2:] if img.ndim == 3 else img.shape
    
    for _ in range(n_holes):
        y, x = np.random.randint(h), np.random.randint(w)
        y1, y2 = np.clip([y - length//2, y + length//2], 0, h)
        x1, x2 = np.clip([x - length//2, x + length//2], 0, w)
        
        if img.ndim == 3:
            img[:, y1:y2, x1:x2] = 1.0
        else:
            img[y1:y2, x1:x2] = 1.0
    return img


def elastic_transform(image, alpha_range=(60, 80), sigma_choices=[12, 13, 14]):
    """弾性変形"""
    img = np.array(image)
    h, w = img.shape[:2]
    
    alpha = np.random.randint(alpha_range[0], alpha_range[1])
    sigma = np.random.choice(sigma_choices)
    
    # 変位フィールド生成
    dx = (np.random.random((h, w)) * 2 - 1) * alpha
    dy = (np.random.random((h, w)) * 2 - 1) * alpha
    
    # ガウシアンフィルタ適用
    ksize = int(sigma * 6 + 1) | 1  # 奇数にする
    dx = cv2.GaussianBlur(dx.astype(np.float32), (ksize, ksize), sigma)
    dy = cv2.GaussianBlur(dy.astype(np.float32), (ksize, ksize), sigma)
    
    # 座標グリッドに変位を適用
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def shear_transform(image, shear_range=(-0.2, 0.2)):
    """せん断変形"""
    img = image[0] if image.ndim == 3 else image
    h, w = img.shape
    
    shear_x = np.random.uniform(*shear_range)
    shear_y = np.random.uniform(*shear_range)
    
    M = np.array([[1, shear_x, -shear_x * w/2],
                  [shear_y, 1, -shear_y * h/2]], dtype=np.float32)
    
    result = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return result.reshape(1, h, w) if image.ndim == 3 else result


def gaussian_noise(image, mean=0, std=0.03):
    """ガウシアンノイズを追加"""
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    return np.clip(image + noise, 0, 1)


def rotate_image(image, angle_range=(-5, 5), scale=0.9):
    """画像を回転"""
    angle = np.random.uniform(*angle_range)
    M = cv2.getRotationMatrix2D((14, 14), angle, scale)
    return cv2.warpAffine(image, M, (28, 28), borderMode=cv2.BORDER_REFLECT)


def augment_data(images, labels, return_unshuffled=False):
    """データ拡張（メイン関数）"""
    augmentations = [
        ('Original', lambda img: img),
        ('Elastic', lambda img: elastic_transform(img[0]).reshape(1, 28, 28)),
        ('Rotate+Elastic', lambda img: elastic_transform(rotate_image(img[0])).reshape(1, 28, 28)),
        ('Cutout', lambda img: cutout(img, n_holes=1, length=np.random.randint(6, 10))),
        ('Shear', lambda img: shear_transform(img)),
        ('GaussianNoise', lambda img: gaussian_noise(img))
    ]
    
    aug_images, aug_labels, aug_types = [], [], []
    
    for img, lbl in zip(images, labels):
        for aug_type, aug_func in augmentations:
            aug_images.append(aug_func(img))
            aug_labels.append(lbl)
            aug_types.append(aug_type)
    
    # numpy配列に変換
    aug_images = np.array(aug_images)
    aug_labels = np.array(aug_labels)
    aug_types = np.array(aug_types)
    
    # シャッフル前のデータを保存（必要な場合）
    unshuffled_data = (aug_images.copy(), aug_labels.copy(), aug_types.copy()) if return_unshuffled else None
    
    # シャッフル
    indices = np.random.permutation(len(aug_images))
    shuffled = aug_images[indices], aug_labels[indices], aug_types[indices]
    
    return (*shuffled, unshuffled_data) if return_unshuffled else shuffled