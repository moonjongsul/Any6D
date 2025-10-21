# SAM GUI 애플리케이션

SAM (Segment Anything Model)을 사용한 물체 분할 GUI 애플리케이션입니다.

## 🎯 기능

- **이미지 로드**: PNG, JPG, JPEG, BMP, TIFF 형식 지원
- **인터랙티브 클릭**: 마우스 클릭으로 물체/배경 점 지정
- **실시간 분할**: SAM2를 사용한 즉시 물체 분할
- **다중 마스크**: 여러 분할 후보 중 선택 가능
- **마스크 저장**: PNG 형식으로 마스크 저장

## 📁 파일 구조

```
sam_gui.py          # 고급 GUI (Tkinter 기반)
simple_sam_gui.py   # 간단한 GUI (OpenCV 기반)
test_sam_gui.py     # GUI 테스트 스크립트
```

## 🚀 사용법

### 1. 테스트 실행
```bash
python test_sam_gui.py
```

### 2. 간단한 GUI 실행
```bash
python simple_sam_gui.py
```

**키보드 단축키:**
- `i`: 이미지 로드
- `f`: 물체 점 추가 모드
- `b`: 배경 점 추가 모드
- `s`: SAM 실행
- `n`: 다음 마스크
- `p`: 이전 마스크
- `c`: 점들 초기화
- `q`: 종료
- `h`: 도움말

### 3. 고급 GUI 실행
```bash
python sam_gui.py
```

**기능:**
- 그래픽 인터페이스
- 마우스 클릭으로 점 추가
- 드롭다운으로 마스크 선택
- 정보 패널
- 마스크 저장 기능

## 📋 요구사항

### 필수 의존성
```bash
pip install opencv-python
pip install torch
pip install matplotlib
pip install pillow
```

### SAM2 설치
```bash
# SAM2 저장소 클론 및 설치
git clone https://github.com/facebookresearch/segment-anything-2.git sam2
cd sam2
pip install -e .
```

### 체크포인트 다운로드
```bash
# SAM2 체크포인트 다운로드
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -O sam2/checkpoints/sam2_hiera_large.pt
```

## 🎮 사용 예시

### 간단한 GUI 사용법
1. `python simple_sam_gui.py` 실행
2. `i` 키를 눌러 이미지 로드
3. `f` 키를 눌러 물체 점 추가 모드
4. 이미지에서 물체를 클릭하여 점 추가
5. `s` 키를 눌러 SAM 실행
6. `n`/`p` 키로 마스크 선택
7. `q` 키로 종료

### 고급 GUI 사용법
1. `python sam_gui.py` 실행
2. "이미지 로드" 버튼 클릭
3. "점 추가 (물체)" 버튼 클릭
4. 이미지에서 물체를 클릭하여 점 추가
5. "SAM 실행" 버튼 클릭
6. 드롭다운에서 마스크 선택
7. "마스크 저장" 버튼으로 저장

## 🔧 문제 해결

### CUDA 메모리 부족
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### SAM2 모델 로딩 실패
- 체크포인트 파일 경로 확인
- 모델 설정 파일 경로 확인
- GPU 메모리 확인

### GUI 실행 실패
- Tkinter 설치 확인: `python -c "import tkinter"`
- OpenCV 설치 확인: `python -c "import cv2"`

## 📝 주의사항

- 첫 번째 실행 시 SAM2 모델 로딩에 시간이 걸릴 수 있습니다
- GPU 사용 시 충분한 VRAM이 필요합니다 (최소 8GB 권장)
- 이미지 크기가 클 경우 처리 시간이 오래 걸릴 수 있습니다

## 🎨 커스터마이징

### 색상 변경
`simple_sam_gui.py`에서 색상 코드 수정:
```python
color = (0, 0, 255) if label == 1 else (255, 0, 0)  # BGR 형식
```

### 마스크 투명도 조정
```python
display_img = cv2.addWeighted(display_img, 0.7, mask_colored, 0.3, 0)
```

## 📞 지원

문제가 발생하면 다음을 확인하세요:
1. 모든 의존성이 설치되었는지
2. SAM2 체크포인트가 올바른 위치에 있는지
3. 이미지 파일이 올바른 형식인지
4. GPU 메모리가 충분한지

