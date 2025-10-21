#!/usr/bin/env python3
"""
화면 크기 문제 해결 스크립트
"""

import tkinter as tk
import os

def check_display_settings():
    """화면 설정 확인"""
    print("=== 화면 설정 확인 ===")
    
    # Tkinter로 화면 정보 확인
    root = tk.Tk()
    root.withdraw()  # 윈도우 숨기기
    
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    screen_dpi = root.winfo_fpixels('1i')
    
    print(f"화면 해상도: {screen_width} x {screen_height}")
    print(f"화면 DPI: {screen_dpi}")
    
    # 현재 스케일링 확인
    current_scaling = root.tk.call('tk', 'scaling')
    print(f"현재 Tkinter 스케일링: {current_scaling}")
    
    root.destroy()
    
    # 환경 변수 확인
    print("\n=== 환경 변수 확인 ===")
    env_vars = [
        "GDK_SCALE",
        "GDK_DPI_SCALE", 
        "QT_AUTO_SCREEN_SCALE_FACTOR",
        "QT_SCALE_FACTOR",
        "DISPLAY"
    ]
    
    for var in env_vars:
        value = os.environ.get(var, "설정되지 않음")
        print(f"{var}: {value}")
    
    return screen_width, screen_height, screen_dpi, current_scaling

def suggest_fixes(screen_width, screen_height, screen_dpi, current_scaling):
    """해결책 제안"""
    print("\n=== 해결책 제안 ===")
    
    if screen_dpi > 120:
        print("고해상도 화면 감지됨")
        print("권장 설정:")
        print("1. Tkinter 스케일링: 2.0")
        print("2. 환경 변수:")
        print("   export GDK_SCALE=2")
        print("   export GDK_DPI_SCALE=0.5")
        print("   export QT_SCALE_FACTOR=2")
        print("3. 큰 화면 GUI 사용: python large_sam_gui.py")
        
    elif screen_dpi > 96:
        print("중간 해상도 화면 감지됨")
        print("권장 설정:")
        print("1. Tkinter 스케일링: 1.5")
        print("2. 환경 변수:")
        print("   export GDK_SCALE=1.5")
        print("   export QT_SCALE_FACTOR=1.5")
        
    else:
        print("일반 해상도 화면 감지됨")
        print("권장 설정:")
        print("1. Tkinter 스케일링: 1.0")
        print("2. 큰 화면 GUI 사용: python large_sam_gui.py")

def create_launch_script():
    """실행 스크립트 생성"""
    script_content = '''#!/bin/bash
# SAM GUI 실행 스크립트

echo "=== SAM GUI 실행 스크립트 ==="

# 환경 변수 설정
export GDK_SCALE=2
export GDK_DPI_SCALE=0.5
export QT_AUTO_SCREEN_SCALE_FACTOR=1
export QT_SCALE_FACTOR=2

echo "환경 변수 설정 완료"

# GUI 선택
echo "실행할 GUI를 선택하세요:"
echo "1. 고급 GUI (Tkinter)"
echo "2. 간단한 GUI (OpenCV)"
echo "3. 큰 화면 GUI (OpenCV)"
echo "4. 종료"

read -p "선택 (1-4): " choice

case $choice in
    1)
        echo "고급 GUI 실행 중..."
        python sam_gui.py
        ;;
    2)
        echo "간단한 GUI 실행 중..."
        python simple_sam_gui.py
        ;;
    3)
        echo "큰 화면 GUI 실행 중..."
        python large_sam_gui.py
        ;;
    4)
        echo "종료"
        exit 0
        ;;
    *)
        echo "잘못된 선택입니다."
        exit 1
        ;;
esac
'''
    
    with open('run_sam_gui.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('run_sam_gui.sh', 0o755)
    print("\n실행 스크립트 생성 완료: run_sam_gui.sh")
    print("사용법: ./run_sam_gui.sh")

def main():
    """메인 함수"""
    print("SAM GUI 화면 크기 문제 해결 도구")
    print("=" * 50)
    
    # 화면 설정 확인
    screen_width, screen_height, screen_dpi, current_scaling = check_display_settings()
    
    # 해결책 제안
    suggest_fixes(screen_width, screen_height, screen_dpi, current_scaling)
    
    # 실행 스크립트 생성
    create_launch_script()
    
    print("\n=== 추가 팁 ===")
    print("1. 큰 화면에서 사용하려면: python large_sam_gui.py")
    print("2. Tkinter GUI가 작다면: python sam_gui.py (자동 스케일링 적용)")
    print("3. 문제가 계속되면: ./run_sam_gui.sh")

if __name__ == "__main__":
    main()
