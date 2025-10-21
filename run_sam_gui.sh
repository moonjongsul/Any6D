#!/bin/bash
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
