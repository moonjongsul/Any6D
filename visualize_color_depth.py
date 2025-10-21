#!/usr/bin/env python3
"""
Color와 Depth 이미지를 함께 시각화하는 독립 스크립트
"""

import cv2
import numpy as np
import os
import argparse
from PIL import Image


def visualize_color_depth(color_path, depth_path, output_path=None, show_window=True):
    """
    Color와 Depth 이미지를 나란히 시각화하는 함수
    
    Args:
        color_path (str): Color 이미지 경로
        depth_path (str): Depth 이미지 경로
        output_path (str, optional): 출력 이미지 저장 경로
        show_window (bool): 윈도우에 표시할지 여부
    
    Returns:
        np.ndarray: 합쳐진 이미지
    """
    
    # 이미지 로드
    color = cv2.imread(color_path)
    if color is None:
        raise ValueError(f"Color 이미지를 로드할 수 없습니다: {color_path}")
    
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise ValueError(f"Depth 이미지를 로드할 수 없습니다: {depth_path}")
    
    # Color 이미지를 RGB로 변환
    color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    
    # Depth 이미지를 시각화 가능한 형태로 변환
    if depth.dtype != np.uint8:
        # Depth 값을 0-255 범위로 정규화
        depth_normalized = cv2.convertScaleAbs(
            depth, 
            alpha=255.0/depth.max() if depth.max() > 0 else 1
        )
    else:
        depth_normalized = depth
    
    # 컬러맵 적용 (JET 컬러맵 사용)
    depth_vis = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    # 이미지 크기 맞추기
    h, w = color_rgb.shape[:2]
    depth_vis = cv2.resize(depth_vis, (w, h))
    
    # 두 이미지를 나란히 합치기
    combined = np.hstack([color_rgb, depth_vis])
    
    # 텍스트 추가
    cv2.putText(combined, "Color", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Depth", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 구분선 추가
    cv2.line(combined, (w, 0), (w, h), (255, 255, 255), 2)
    
    # 출력 이미지 저장
    if output_path:
        output_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_bgr)
        print(f"시각화 결과 저장: {output_path}")
    
    # 윈도우에 표시
    if show_window:
        cv2.imshow('Color & Depth Visualization', 
                   cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print("ESC 키를 눌러 종료하세요.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return combined


def main():
    parser = argparse.ArgumentParser(description="Color와 Depth 이미지를 함께 시각화")
    parser.add_argument("--color", type=str, required=True, help="Color 이미지 경로")
    parser.add_argument("--depth", type=str, required=True, help="Depth 이미지 경로")
    parser.add_argument("--output", type=str, help="출력 이미지 저장 경로")
    parser.add_argument("--no-show", action="store_true", help="윈도우에 표시하지 않음")
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    if not os.path.exists(args.color):
        print(f"Error: Color 이미지 파일이 존재하지 않습니다: {args.color}")
        return
    
    if not os.path.exists(args.depth):
        print(f"Error: Depth 이미지 파일이 존재하지 않습니다: {args.depth}")
        return
    
    try:
        # 시각화 실행
        combined = visualize_color_depth(
            args.color, 
            args.depth, 
            args.output, 
            not args.no_show
        )
        print("시각화 완료!")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
