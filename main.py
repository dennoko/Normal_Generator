import numpy as np
import cv2
import argparse
from enum import Enum
import os

class ProfileType(Enum):
    LINEAR = 1
    LOGARITHMIC = 2
    EXPONENTIAL = 3

class MaskToNormalMap:
    def __init__(self):
        pass
    
    def detect_edges(self, mask_img):
        """エッジ検出を行う"""
        # 水平・垂直エッジの検出
        sobel_x = cv2.Sobel(mask_img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(mask_img, cv2.CV_64F, 0, 1, ksize=3)
        
        # 斜め方向のエッジ検出 - フィルタを使用
        kernel_diag1 = np.array([[-1, -2, 0], [-2, 0, 2], [0, 2, 1]], dtype=np.float32)
        kernel_diag2 = np.array([[0, -2, -1], [2, 0, -2], [1, 2, 0]], dtype=np.float32)
        
        sobel_diag1 = cv2.filter2D(mask_img, cv2.CV_64F, kernel_diag1)
        sobel_diag2 = cv2.filter2D(mask_img, cv2.CV_64F, kernel_diag2)
        
        # 結果の合成
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2 + sobel_diag1**2 + sobel_diag2**2)
        
        # 正規化して反転（白背景に黒のエッジ）
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        edges = 255 - magnitude.astype(np.uint8)
        
        return edges
    
    def apply_blur_profile(self, edges, radius, profile_type):
        """エッジに対して指定されたプロファイルでぼかしを適用する（パディングを使用）"""
        height, width = edges.shape
        
        # 画像の周囲にradiusピクセル分のパディングを追加
        padded_edges = cv2.copyMakeBorder(
            edges, 
            radius, radius, radius, radius, 
            cv2.BORDER_CONSTANT, 
            value=255
        )
        
        # 出力用の画像を作成（白で初期化）- パディングサイズも考慮
        padded_blurred = np.ones((height + 2*radius, width + 2*radius), dtype=np.float32) * 255
        
        # エッジピクセルの位置を取得（パディング後の座標）
        edge_pixels = np.where(padded_edges < 128)
        edge_coords = list(zip(edge_pixels[0], edge_pixels[1]))
        
        # 各エッジピクセルに対して円形のぼかしを適用
        for y, x in edge_coords:
            edge_value = padded_edges[y, x]
            
            # 円の範囲内のピクセルを処理
            for cy in range(max(0, y - radius), min(padded_edges.shape[0], y + radius + 1)):
                for cx in range(max(0, x - radius), min(padded_edges.shape[1], x + radius + 1)):
                    # 中心からの距離を計算
                    distance = np.sqrt((cy - y)**2 + (cx - x)**2)
                    
                    if distance <= radius:
                        # 距離に基づいてグラデーションを計算
                        normalized_dist = distance / radius
                        
                        if profile_type == ProfileType.LINEAR:
                            # 線形プロファイル
                            intensity = normalized_dist * 255
                        elif profile_type == ProfileType.LOGARITHMIC:
                            # 対数プロファイル（凸型）
                            intensity = 255 * (np.log(1 + 9 * normalized_dist) / np.log(10))
                        elif profile_type == ProfileType.EXPONENTIAL:
                            # 指数プロファイル（凹型）
                            intensity = 255 * (np.exp(normalized_dist * 3) - 1) / (np.exp(3) - 1)
                        else:
                            intensity = normalized_dist * 255
                            
                        # 既存のピクセル値と比較して、より暗い値を採用
                        if intensity < padded_blurred[cy, cx]:
                            padded_blurred[cy, cx] = intensity
        
        # パディングを取り除いて元のサイズに戻す
        blurred = padded_blurred[radius:radius+height, radius:radius+width]
        
        return blurred.astype(np.uint8)
    
    def apply_blur_profile_optimized(self, edges, radius, profile_type):
        """
        エッジに対して指定されたプロファイルでぼかしを適用する最適化版
        距離変換とルックアップテーブルを使用して高速化
        """
        height, width = edges.shape
        
        # 画像の周囲にradiusピクセル分のパディングを追加
        padded_edges = cv2.copyMakeBorder(
            edges, 
            radius, radius, radius, radius, 
            cv2.BORDER_CONSTANT, 
            value=255
        )
        
        # エッジマスクを作成（0:エッジピクセル、255:非エッジピクセル）
        edge_mask = (padded_edges < 128).astype(np.uint8) * 255
        
        # 距離変換を使用してエッジからの距離を計算
        dist = cv2.distanceTransform(255 - edge_mask, cv2.DIST_L2, 5)
        
        # 距離の最大値をradiusに制限
        dist = np.minimum(dist, radius)
        
        # 距離を正規化
        normalized_dist = dist / radius
        
        # プロファイルに応じた強度を計算
        if profile_type == ProfileType.LINEAR:
            # 線形プロファイル
            intensity = normalized_dist * 255
        elif profile_type == ProfileType.LOGARITHMIC:
            # 対数プロファイル（凸型）
            intensity = 255 * (np.log(1 + 9 * normalized_dist) / np.log(10))
        elif profile_type == ProfileType.EXPONENTIAL:
            # 指数プロファイル（凹型）
            intensity = 255 * (np.exp(normalized_dist * 3) - 1) / (np.exp(3) - 1)
        else:
            intensity = normalized_dist * 255
            
        # パディングを取り除いて元のサイズに戻す
        blurred = intensity[radius:radius+height, radius:radius+width]
        
        return blurred.astype(np.uint8)
    
    def combine_mask_and_blur(self, mask_img, blurred_edges):
        """マスク画像とぼかしたエッジを合成する"""
        # マスク画像が白黒逆（黒地に白）の場合は反転
        if np.mean(mask_img) < 127:
            mask_img = 255 - mask_img
            
        # エッジのぼかしを適用
        height_map = cv2.min(mask_img, blurred_edges)
        
        return height_map
    
    def generate_normal_map(self, height_map, strength=1.0):
        """ハイトマップからノーマルマップを生成する"""
        # 画像境界での問題を避けるためにパディングを追加
        padded_height_map = cv2.copyMakeBorder(height_map, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        
        # 縦横の勾配を計算
        sobel_x = cv2.Sobel(padded_height_map, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(padded_height_map, cv2.CV_32F, 0, 1, ksize=3)
        
        # パディングを取り除く
        sobel_x = sobel_x[1:-1, 1:-1]
        sobel_y = sobel_y[1:-1, 1:-1]
        
        # 強度の調整
        sobel_x = sobel_x * strength / 255.0
        sobel_y = sobel_y * strength / 255.0
        
        # 法線ベクトルの計算
        normal_map = np.zeros((height_map.shape[0], height_map.shape[1], 3), dtype=np.uint8)
        
        # X成分（赤チャンネル）- Xの勾配を反転
        normal_map[:, :, 2] = np.clip(127.5 - sobel_x * 127.5, 0, 255).astype(np.uint8)
        
        # Y成分（緑チャンネル）- Yの勾配を反転
        normal_map[:, :, 1] = np.clip(127.5 - sobel_y * 127.5, 0, 255).astype(np.uint8)
        
        # Z成分（青チャンネル）は常に正
        normal_map[:, :, 0] = 255
        
        return normal_map
    
    def process(self, input_path, output_path, profile_type=ProfileType.LINEAR, radius=10, strength=1.0, use_optimized=True):
        """マスク画像からノーマルマップを生成する全体のプロセス"""
        # 画像の読み込み
        mask_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise ValueError(f"入力画像を読み込めませんでした: {input_path}")
            
        # エッジの検出
        edges = self.detect_edges(mask_img)
        
        # エッジのぼかし（最適化版か通常版を選択）
        if use_optimized:
            blurred_edges = self.apply_blur_profile_optimized(edges, radius, profile_type)
        else:
            blurred_edges = self.apply_blur_profile(edges, radius, profile_type)
        
        # マスクとぼかしの合成
        height_map = self.combine_mask_and_blur(mask_img, blurred_edges)
        
        # ノーマルマップの生成
        normal_map = self.generate_normal_map(height_map, strength)
        
        # 結果の保存
        cv2.imwrite(output_path, normal_map)
        
        # 中間結果も保存（デバッグ用）
        base_name = os.path.splitext(output_path)[0]
        cv2.imwrite(f"{base_name}_edges.png", edges)
        cv2.imwrite(f"{base_name}_blurred.png", blurred_edges)
        cv2.imwrite(f"{base_name}_height.png", height_map)
        
        return normal_map, edges, blurred_edges, height_map

def parse_args():
    parser = argparse.ArgumentParser(description='マスク画像からノーマルマップを生成します。')
    parser.add_argument('input', help='入力マスク画像のパス')
    parser.add_argument('output', help='出力ノーマルマップのパス')
    parser.add_argument('--profile', type=int, choices=[1, 2, 3], default=1,
                        help='断面形状のプロファイル: 1=LINEAR, 2=LOGARITHMIC, 3=EXPONENTIAL')
    parser.add_argument('--radius', type=int, default=10, help='ぼかしの半径')
    parser.add_argument('--strength', type=float, default=1.0, help='ノーマルマップの強度')
    parser.add_argument('--no-optimize', action='store_true', help='最適化版のぼかし処理を使用しない')
    return parser.parse_args()

def main():
    args = parse_args()
    
    profile_map = {
        1: ProfileType.LINEAR,
        2: ProfileType.LOGARITHMIC,
        3: ProfileType.EXPONENTIAL
    }
    
    processor = MaskToNormalMap()
    
    try:
        normal_map, edges, blurred, height_map = processor.process(
            args.input,
            args.output,
            profile_type=profile_map[args.profile],
            radius=args.radius,
            strength=args.strength,
            use_optimized=not args.no_optimize
        )
        
        print(f"処理が完了しました。結果は {args.output} に保存されました。")
        print(f"中間結果も保存されました: {args.output}_edges.png, {args.output}_blurred.png, {args.output}_height.png")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
    