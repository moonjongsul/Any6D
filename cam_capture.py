import cv2
import numpy as np
import os
import time
from scipy.interpolate import griddata

from ketisdk.sensor.femto import FemtoSensor
from ketisdk.sensor.realsense_sensor import RSSensor


def interpolate_depth_cv2(depth: np.ndarray, radius: int = 3) -> np.ndarray:
    """
    Interpolate missing depth regions (value == 0) using OpenCV inpainting.

    Args:
        depth (np.ndarray): Input depth map (H, W), uint16 or float32.
        radius (int): Inpainting radius (pixel neighborhood).

    Returns:
        np.ndarray: Interpolated depth map (same shape, dtype float32).
    """
    # convert to float
    depth = depth.astype(np.float32)

    # mask: 0인 부분만 보간
    mask = (depth == 0).astype(np.uint8)

    # 보간 (TELEA 방식)
    interpolated_depth = cv2.inpaint(depth, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)

    return interpolated_depth

def interpolate_depth_scipy(depth: np.ndarray, method='linear') -> np.ndarray:
    """
    Interpolate missing depth regions using scipy.interpolate.griddata.

    Args:
        depth (np.ndarray): Input depth map (H, W), float32 or uint16.
        method (str): Interpolation method ('linear', 'nearest', 'cubic').

    Returns:
        np.ndarray: Interpolated depth map (same shape, dtype float32).
    """
    depth = depth.astype(np.float32)
    depth[depth == 0] = np.nan  # 0을 NaN으로 치환

    h, w = depth.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    # 유효 픽셀만 선택
    valid = ~np.isnan(depth)
    points = np.stack([xx[valid], yy[valid]], axis=-1)
    values = depth[valid]

    # 선형 보간
    interpolated = griddata(points, values, (xx, yy), method=method)

    # 남은 NaN은 최근접 이웃으로 보완
    if np.isnan(interpolated).any():
        nearest = griddata(points, values, (xx, yy), method='nearest')
        interpolated[np.isnan(interpolated)] = nearest[np.isnan(interpolated)]

    return interpolated

def visualize_color_depth(color, depth, window_name="Color & Depth"):
    """
    Color와 Depth 이미지를 나란히 시각화하는 함수
    """
    # Depth 이미지를 시각화 가능한 형태로 변환
    depth_vis = cv2.applyColorMap(
        cv2.convertScaleAbs(depth, alpha=255.0/depth.max() if depth.max() > 0 else 1), 
        cv2.COLORMAP_JET
    )
    
    # 이미지 크기 맞추기
    h, w = color.shape[:2]
    depth_vis = cv2.resize(depth_vis, (w, h))
    
    # 두 이미지를 나란히 합치기
    combined = np.hstack([color[:, :, ::-1], depth_vis])
    
    # 텍스트 추가
    cv2.putText(combined, "Color", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Depth", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 구분선 추가
    cv2.line(combined, (w, 0), (w, h), (255, 255, 255), 2)
    
    return combined


def main():
    camera = RSSensor()

    camera.start()
    time.sleep(1)

    print("Color & Depth 시각화 시작...")
    print("ESC 키를 눌러 종료하세요.")
    print("D 키를 눌러 color, depth 이미지를 저장하세요.")

    # 저장할 디렉토리 생성
    save_dir = "captured_images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"저장 디렉토리 생성: {save_dir}")

    frame_count = 0

    while True:
        color, depth = camera.get_data()
        # depth = interpolate_depth_cv2(depth)
        # Color와 Depth를 함께 시각화
        combined = visualize_color_depth(color, depth)
        
        # 윈도우에 표시
        cv2.imshow('Color & Depth Visualization', combined)
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 키
            break
        elif key == ord('d') or key == ord('D'):  # D 키
            # 타임스탬프 생성
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # 파일명 생성
            color_filename = os.path.join(save_dir, f"color_{timestamp}_{frame_count:04d}.png")
            depth_filename = os.path.join(save_dir, f"depth_{timestamp}_{frame_count:04d}.png")
            
            # Color 이미지 저장 (BGR -> RGB 변환)
            color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            cv2.imwrite(color_filename, color_bgr)
            
            # Depth 이미지 저장 (unchanged 포맷)
            cv2.imwrite(depth_filename, depth)
            
            print(f"이미지 저장 완료:")
            print(f"  Color: {color_filename}")
            print(f"  Depth: {depth_filename}")
            
            frame_count += 1

    camera.stop()
    cv2.destroyAllWindows()
    print("시각화 종료")


if __name__ == "__main__":
    main()   