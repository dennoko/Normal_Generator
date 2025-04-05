import argparse
import os
import numpy as np
import cv2
from PIL import Image
from enum import Enum


class ProfileType(Enum):
    LINEAR = 1
    LOGARITHMIC = 2
    EXPONENTIAL = 3


class NormalMapType(Enum):
    DX = 1  # DirectX (Y+)
    GL = 2  # OpenGL (Y-)


class MaskToNormalMap:
    def detect_edges(self, mask_img):
        """エッジ検出を行う"""
        sobel_x = cv2.Sobel(mask_img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(mask_img, cv2.CV_64F, 0, 1, ksize=3)
        kernel_diag1 = np.array([[-1, -2, 0], [-2, 0, 2], [0, 2, 1]], dtype=np.float32)
        kernel_diag2 = np.array([[0, -2, -1], [2, 0, -2], [1, 2, 0]], dtype=np.float32)
        sobel_diag1 = cv2.filter2D(mask_img, cv2.CV_64F, kernel_diag1)
        sobel_diag2 = cv2.filter2D(mask_img, cv2.CV_64F, kernel_diag2)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2 + sobel_diag1**2 + sobel_diag2**2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        edges = 255 - magnitude.astype(np.uint8)
        return edges

    def apply_blur_profile_optimized(self, edges, radius, profile_type):
        height, width = edges.shape
        padded_edges = cv2.copyMakeBorder(edges, radius, radius, radius, radius, cv2.BORDER_CONSTANT, value=255)
        edge_mask = (padded_edges < 128).astype(np.uint8) * 255
        dist = cv2.distanceTransform(255 - edge_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        dist = np.minimum(dist, radius)
        normalized_dist = dist / radius
        if profile_type == ProfileType.LINEAR:
            intensity = normalized_dist * 255
        elif profile_type == ProfileType.LOGARITHMIC:
            intensity = 255 * (np.log(1 + 9 * normalized_dist) / np.log(10))
        elif profile_type == ProfileType.EXPONENTIAL:
            intensity = 255 * (np.exp(normalized_dist * 2.5) - 1) / (np.exp(2.5) - 1)
        else:
            intensity = normalized_dist * 255
        intensity = cv2.GaussianBlur(intensity, (3, 3), 0)
        blurred = intensity[radius:radius+height, radius:radius+width]
        return blurred.astype(np.uint8)

    def combine_mask_and_blur(self, mask_img, blurred_edges, invert_mask):
        if invert_mask:
            mask_img = 255 - mask_img
        height_map = cv2.min(mask_img, blurred_edges)
        return height_map

    def generate_normal_map(self, height_map, strength=1.0, normal_map_type=NormalMapType.DX):
        padded_height_map = cv2.copyMakeBorder(height_map, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        sobel_x = cv2.Sobel(padded_height_map, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(padded_height_map, cv2.CV_32F, 0, 1, ksize=3)
        sobel_x = sobel_x[1:-1, 1:-1] * strength / 255.0
        sobel_y = sobel_y[1:-1, 1:-1] * strength / 255.0
        sobel_z = np.sqrt(1 - np.clip(sobel_x**2 + sobel_y**2, 0, 1))
        normal_map = np.zeros((height_map.shape[0], height_map.shape[1], 3), dtype=np.uint8)
        normal_map[:, :, 0] = np.clip(127.5 + sobel_z * 127.5, 0, 255).astype(np.uint8)
        normal_map[:, :, 2] = np.clip(127.5 - sobel_x * 127.5, 0, 255).astype(np.uint8)
        if normal_map_type == NormalMapType.DX:
            normal_map[:, :, 1] = np.clip(127.5 - sobel_y * 127.5, 0, 255).astype(np.uint8)
        else:
            normal_map[:, :, 1] = np.clip(127.5 + sobel_y * 127.5, 0, 255).astype(np.uint8)
        return normal_map

    def process(self, input_path, output_path, profile_type, radius, strength, normal_map_type, save_intermediates, invert_mask, disable_blurring):
        pil_image = Image.open(input_path).convert("L")
        mask_img = np.array(pil_image)
        edges = self.detect_edges(mask_img)
        blurred_edges = self.apply_blur_profile_optimized(edges, radius, profile_type)
        height_map = mask_img if disable_blurring else self.combine_mask_and_blur(mask_img, blurred_edges, invert_mask)
        normal_map = self.generate_normal_map(height_map, strength, normal_map_type)
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        pil_normal_map = cv2.cvtColor(normal_map, cv2.COLOR_BGR2RGB)
        pil_normal_map = Image.fromarray(pil_normal_map)
        pil_normal_map.save(output_path, format="PNG")
        if save_intermediates:
            processing_dir = os.path.join(os.path.dirname(input_path), "processing")
            if not os.path.exists(processing_dir):
                os.makedirs(processing_dir)
            base_name = os.path.basename(input_path).split('.')[0]
            cv2.imwrite(os.path.join(processing_dir, f"{base_name}_edges.png"), edges)
            cv2.imwrite(os.path.join(processing_dir, f"{base_name}_blurred.png"), blurred_edges)
            cv2.imwrite(os.path.join(processing_dir, f"{base_name}_height.png"), height_map)


def main():
    parser = argparse.ArgumentParser(description="Generate normal maps from mask images.")
    parser.add_argument("input", help="Path to the input mask image.")
    parser.add_argument("output", help="Path to save the output normal map.")
    parser.add_argument("--profile", type=int, choices=[1, 2, 3], default=1, help="Slope profile type: 1=Linear, 2=Logarithmic, 3=Exponential.")
    parser.add_argument("--radius", type=int, default=15, help="Radius for slope generation.")
    parser.add_argument("--strength", type=float, default=1.0, help="Strength of the normal map.")
    parser.add_argument("--type", type=int, choices=[1, 2], default=1, help="Normal map type: 1=DirectX (Y+), 2=OpenGL (Y-).")
    parser.add_argument("--save-intermediates", action="store_true", help="Save intermediate processing results.")
    parser.add_argument("--invert", action="store_true", help="Invert the mask image.")
    parser.add_argument("--disable-blurring", action="store_true", help="Disable slope generation.")
    args = parser.parse_args()

    processor = MaskToNormalMap()
    processor.process(
        input_path=args.input,
        output_path=args.output,
        profile_type=ProfileType(args.profile),
        radius=args.radius,
        strength=args.strength,
        normal_map_type=NormalMapType(args.type),
        save_intermediates=args.save_intermediates,
        invert_mask=args.invert,
        disable_blurring=args.disable_blurring
    )
    print(f"Normal map saved to {args.output}")


if __name__ == "__main__":
    main()    