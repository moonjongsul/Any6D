#!/usr/bin/env python3
"""
SAM GUI 테스트 스크립트
"""

import os
import sys

def test_gui_availability():
    """GUI 실행 가능 여부 테스트"""
    print("=== SAM GUI 테스트 ===")
    
    # 필요한 모듈 확인
    try:
        import cv2
        print("✓ OpenCV 사용 가능")
    except ImportError:
        print("✗ OpenCV 없음 - pip install opencv-python")
        return False
    
    try:
        import tkinter
        print("✓ Tkinter 사용 가능")
    except ImportError:
        print("✗ Tkinter 없음 - GUI 버전 사용 불가")
    
    try:
        import torch
        print(f"✓ PyTorch 사용 가능 (버전: {torch.__version__})")
    except ImportError:
        print("✗ PyTorch 없음 - pip install torch")
        return False
    
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("✓ SAM2 모듈 사용 가능")
    except ImportError:
        print("✗ SAM2 모듈 없음 - SAM2 설치 필요")
        return False
    
    # 체크포인트 파일 확인
    checkpoint_path = "sam2/checkpoints/sam2_hiera_large.pt"
    if os.path.exists(checkpoint_path):
        print(f"✓ SAM2 체크포인트 발견: {checkpoint_path}")
    else:
        print(f"✗ SAM2 체크포인트 없음: {checkpoint_path}")
        print("  SAM2 체크포인트를 다운로드하세요.")
        return False
    
    config_path = "sam2/sam2_hiera_l.yaml"
    if os.path.exists(config_path):
        print(f"✓ SAM2 설정 파일 발견: {config_path}")
    else:
        print(f"✗ SAM2 설정 파일 없음: {config_path}")
        return False
    
    return True

def main():
    """메인 함수"""
    if test_gui_availability():
        print("\n=== GUI 실행 옵션 ===")
        print("1. 고급 GUI (Tkinter): python sam_gui.py")
        print("2. 간단한 GUI (OpenCV): python simple_sam_gui.py")
        print("\n간단한 GUI를 실행하시겠습니까? (y/n): ", end="")
        
        choice = input().lower()
        if choice == 'y':
            print("간단한 GUI 실행 중...")
            os.system("python simple_sam_gui.py")
        else:
            print("GUI를 종료합니다.")
    else:
        print("\n필요한 의존성을 설치한 후 다시 시도하세요.")

if __name__ == "__main__":
    main()

