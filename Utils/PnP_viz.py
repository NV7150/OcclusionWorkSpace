import numpy as np
import cv2
import open3d as o3d
from Logger import logger, Logger

# グローバル変数: Trueの場合OpenCV座標系、Falseの場合Open3D座標系を使用
is_opencv = False

# 座標系変換のヘルパー関数
def convert_coordinates(points, vectors=None):
    """
    座標系の変換を行う関数
    
    Args:
        points (np.ndarray): 変換する点の座標
        vectors (np.ndarray, optional): 変換するベクトル
        
    Returns:
        tuple: (変換後の点, 変換後のベクトル)
    """
    # OpenCVとOpen3Dの座標系変換行列
    open3d_to_opencv = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    
    if not is_opencv:
        # すでにOpenCV座標系の場合は変換不要
        return points, vectors
    else:
        # Open3D座標系からOpenCV座標系への変換
        converted_points = points @ open3d_to_opencv.T if points is not None else None
        converted_vectors = vectors @ open3d_to_opencv.T if vectors is not None else None
        return converted_points, converted_vectors

def visualize_pnp_result(object_points: np.ndarray, result: np.ndarray, marker_pose: dict):
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

    open3d_to_opencv = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    
    rvec = result[:3].reshape(3, 1)
    tvec = result[3:].reshape(3, 1)
    
    R, _ = cv2.Rodrigues(rvec)
    cam_pos = -R.T @ tvec

    # Convert coordinates if needed (result is always in OpenCV coordinates)
    if not is_opencv:
        # Convert from OpenCV to OpenGL coordinate system
        opencv_to_opengl = open3d_to_opencv.T  # Inverse of open3d_to_opencv
        
        # Convert translation vector
        # tvec_opengl = (-R.T @ tvec).flatten()
        
        R = opencv_to_opengl @ R 
        
        tvec_opengl = opencv_to_opengl @ cam_pos
        # tvec_opengl[1:] *= -1
        # tvec_opengl = -R.T @ tvec
        cam_pos = tvec_opengl
        
        # Convert object points from OpenCV to OpenGL coordinate system
        object_points = object_points @ opencv_to_opengl.T
    

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
    
    model = o3d.io.read_point_cloud("../LocalData/DepthIMUData2/Env_3DModels/on_the_desk.ply")

    
    if is_opencv:
        model.rotate(open3d_to_opencv, center=(0, 0, 0))
    geometries.append(model)
    # マーカーIDに基づく色を定義
    marker_colors = {
        '0': [1.0, 0.0, 0.0],    # 赤
        '1': [0.0, 0.0, 1.0],    # 青
        '2': [0.0, 1.0, 0.0],    # 緑
        '3': [1.0, 1.0, 0.0],    # 黄
        '4': [1.0, 0.0, 1.0],    # マゼンタ
        '5': [0.0, 1.0, 1.0],    # シアン
    }
    
    for i,  (marker_id, marker_data) in enumerate(marker_pose.items()):
        if "pos" in marker_data:
            # マーカーの位置データを取得し、必要に応じて座標変換を適用
            position = np.array(marker_data["pos"])
            position_opencv, _ = convert_coordinates(position, None)
            
            # マーカー位置にボックスを配置
            box = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.05, depth=0.01)
            
            # マーカーIDに基づいて色を割り当て
            color = marker_colors.get(str(i), [0.5, 0.5, 0.5])  # デフォルトはグレー
            box.paint_uniform_color(color)
            box.scale(2, center=box.get_center())
            
            # マーカーの位置と向きに基づいてボックスを配置
            if "norm" in marker_data and "tangent" in marker_data:
                norm = np.array(marker_data["norm"])
                tangent = np.array(marker_data["tangent"])
                
                # 法線と接線ベクトルを変換
                _, converted_vectors = convert_coordinates(None, np.vstack([norm, tangent]))
                norm = converted_vectors[0]
                tangent = converted_vectors[1]
                
                norm = norm / np.linalg.norm(norm)  # 正規化
                
                # z軸をnormに、x軸をtangentに合わせる回転行列を計算
                z_axis = norm
                
                # tangentからnormに垂直な成分を取り出す
                x_axis = tangent - np.dot(tangent, z_axis) * z_axis
                x_axis = x_axis / np.linalg.norm(x_axis)  # 正規化
                
                # 右手系を保つためにy_axisを計算
                y_axis = np.cross(z_axis, x_axis)
                
                # 回転行列を構築
                rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
                
                # ボックスを回転させて配置
                box.rotate(rotation_matrix, center=[0, 0, 0])
                box.translate(position_opencv)
            else:
                # 回転情報がない場合は単純に位置だけを設定
                box.translate(position_opencv)
            
            geometries.append(box)
            
            # マーカー番号を表示するためのテキスト
            print(f"Marker {marker_id} placed at position: {position_opencv} (original: {position})")

    # 可視化
    o3d.visualization.draw_geometries(geometries)
