import numpy as np
import cv2
import open3d as o3d
from Logger import logger, Logger

def visualize_pnp_result(object_points: np.ndarray, result: np.ndarray):
    """
    object_points（ワールド）の各点に異なる色とインデックス番号を表示。
    カメラ視線方向を矢印で表示。

    Args:
        object_points (np.ndarray): (n, 3) の3D点群（ワールド座標系）
        result (np.ndarray): (6,) or (6,1) の [rvec; tvec]
    """
    assert object_points.shape[1] == 3, "object_points must be of shape (n, 3)"
    result = result.flatten()
    assert result.shape[0] == 6, "result must be of shape (6,) or (6,1)"

    rvec = result[:3].reshape(3, 1)
    tvec = result[3:].reshape(3, 1)

    # 回転行列
    R, _ = cv2.Rodrigues(rvec)

    # カメラのワールド座標位置
    cam_pos = -R.T @ tvec

    # 各点に異なる色を割り当て
    colors = [
        [1.0, 0.0, 0.0],  # 赤
        [0.0, 1.0, 0.0],  # 緑
        [0.0, 0.0, 1.0],  # 青
        [1.0, 1.0, 0.0],  # 黄
        [1.0, 0.0, 1.0],  # マゼンタ
        [0.0, 1.0, 1.0],  # シアン
        [0.5, 0.5, 0.0],  # オリーブ
        [0.5, 0.0, 0.5],  # パープル
    ]

    geometries = []

    for idx, pt in enumerate(object_points):
        # 球体を作成
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.paint_uniform_color(colors[idx % len(colors)])
        sphere.translate(pt)
        geometries.append(sphere)

        # インデックス番号を表示（printで代用）
        print(f"Point {idx}: {pt}")

    # カメラの視線方向を示す矢印
    arrow_length = 0.2
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.005,
        cone_radius=0.01,
        cylinder_height=arrow_length * 0.8,
        cone_height=arrow_length * 0.2
    )
    arrow.paint_uniform_color([0, 1, 0])  # 緑
    arrow.rotate(R.T, center=np.zeros(3))
    arrow.translate(cam_pos.flatten())
    geometries.append(arrow)           
    logger.log(Logger.DEBUG, f"pose in pnp :{cam_pos.flatten()}, {R.T}")
            

    # 座標フレームを追加
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    geometries.append(camera_frame)

    # 可視化
    o3d.visualization.draw_geometries(geometries)
