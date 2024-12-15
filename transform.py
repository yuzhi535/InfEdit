import os
import numpy as np
import cv2
from typing import Dict, Tuple


class PanoramaConverter:
    def __init__(self, output_size: int = 1024):
        self.output_size = output_size
        self.faces = {
            "front": (0, 0, -1),  # 前
            "right": (1, 0, 0),  # 右
            "back": (0, 0, 1),  # 后
            "left": (-1, 0, 0),  # 左
            "up": (0, 1, 0),  # 上
            "down": (0, -1, 0),  # 下
        }

    def _create_face_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """创建面的坐标网格"""
        xs = np.linspace(-1, 1, self.output_size)
        ys = np.linspace(-1, 1, self.output_size)
        x, y = np.meshgrid(xs, ys)
        return x, y

    def _calculate_xyz(
        self, face_direction: Tuple[int, int, int], x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算3D坐标"""
        if face_direction[0]:  # 左右面
            z = x * face_direction[0]
            x = face_direction[0]
            y = y
        elif face_direction[1]:  # 上下面
            z = y * face_direction[1]
            x = x
            y = face_direction[1]
        else:  # 前后面
            z = face_direction[2]
            x = x * -face_direction[2]
            y = y
        return x, y, z

    def convert(self, panorama: np.ndarray) -> Dict[str, np.ndarray]:
        """转换全景图为立方体贴图"""
        h, w = panorama.shape[:2]
        x_grid, y_grid = self._create_face_coordinates()
        result = {}

        for face_name, direction in self.faces.items():
            # 计算3D坐标
            x, y, z = self._calculate_xyz(direction, x_grid, y_grid)

            # 计算球面坐标
            phi = np.arctan2(z, x)
            theta = np.arctan2(y, np.sqrt(x * x + z * z))

            # 映射到全景图
            u = ((phi + np.pi) / (2 * np.pi)) * w
            v = ((theta + np.pi / 2) / np.pi) * h

            # 双线性插值采样
            # TODO: 这里可以优化为更高效的插值方法
            u = np.clip(u, 0, w - 1)
            v = np.clip(v, 0, h - 1)
            face = cv2.remap(
                panorama, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR
            )

            result[face_name] = face

        return result

    def inverse_convert(self, faces: Dict[str, np.ndarray]) -> np.ndarray:
        """将立方体贴图转换回全景图"""
        h, w = self.output_size * 2, self.output_size * 4
        panorama = np.zeros((h, w, 3), dtype=np.uint8)
        x_grid, y_grid = self._create_face_coordinates()

        for face_name, direction in self.faces.items():
            face = faces[face_name]
            x, y, z = self._calculate_xyz(direction, x_grid, y_grid)

            # 计算球面坐标
            phi = np.arctan2(z, x)
            theta = np.arctan2(y, np.sqrt(x * x + z * z))

            # 映射到全景图
            u = ((phi + np.pi) / (2 * np.pi)) * w
            v = ((theta + np.pi / 2) / np.pi) * h

            # 双线性插值采样
            u = np.clip(u, 0, w - 1).astype(np.int32)
            v = np.clip(v, 0, h - 1).astype(np.int32)

            panorama[v, u] = face[y_grid.astype(np.int32), x_grid.astype(np.int32)]

        return panorama


def main():
    # 测试代码
    image_path = "images/urban_street_01.jpg"
    panorama = cv2.imread(image_path)
    converter = PanoramaConverter(output_size=1024)
    cubemap = converter.convert(panorama)

    output_fd = "cubemaps"
    if not os.path.exists(output_fd):
        os.makedirs(output_fd)

    # 保存结果
    for face_name, face_img in cubemap.items():
        cv2.imwrite(os.path.join(output_fd, f"cubemap_{face_name}.jpg"), face_img)


if __name__ == "__main__":
    main()
