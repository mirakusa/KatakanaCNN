import numpy as np
import cv2

def enhance_contrast(image):
    """薄い文字を濃くするコントラスト強化
    CLAHEで局所的なコントラスト強化を行う
    
    Args:
        image: 入力画像 (H, W) または (C, H, W)
    
    Returns:
        enhanced: コントラスト強化後の画像
    """
    # 入力画像の形状を確認
    if image.ndim == 3:
        # (C, H, W) の場合、最初のチャンネルを処理
        img = image[0]
        needs_reshape = True
    else:
        img = image
        needs_reshape = False
    
    # 0-1の範囲を0-255に変換（uint8型が必要）
    if img.max() <= 1.0:
        img_uint8 = (img * 255).astype(np.uint8)
    else:
        img_uint8 = img.astype(np.uint8)
    
    # CLAHEで局所的なコントラスト強化
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img_uint8)
    
    # 元の範囲に戻す
    if img.max() <= 1.0:
        enhanced = enhanced.astype(np.float32) / 255.0
    else:
        enhanced = enhanced.astype(np.float32)
    
    # 必要に応じて形状を戻す
    if needs_reshape:
        return enhanced.reshape(1, enhanced.shape[0], enhanced.shape[1])
    else:
        return enhanced

def preprocess_data(data, enhance_contrast_flag=True):
    """データの前処理を適用
    
    Args:
        data: 画像データの配列
        enhance_contrast_flag: コントラスト強化を適用するか
    
    Returns:
        preprocessed_data: 前処理後のデータ
    """
    if not enhance_contrast_flag:
        return data
    
    preprocessed = []
    for img in data:
        enhanced = enhance_contrast(img)
        preprocessed.append(enhanced)
    
    return np.array(preprocessed)

def normalize_data(data):
    """データの正規化（0-255 -> 0-1）
    
    Args:
        data: 画像データ
    
    Returns:
        normalized: 正規化後のデータ
    """
    return data.astype(np.float32) / 255.0