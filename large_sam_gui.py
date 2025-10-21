#!/usr/bin/env python3
"""
큰 화면 SAM GUI 애플리케이션
OpenCV 기반으로 큰 화면에서 사용하기 편한 GUI
"""

import cv2
import numpy as np
import torch
import os
from sam2.sam2.build_sam import build_sam2
from sam2.sam2.sam2_image_predictor import SAM2ImagePredictor


class LargeSAMGUI:
    def __init__(self):
        self.image = None
        self.image_rgb = None
        self.predictor = None
        self.points = []
        self.labels = []
        self.masks = []
        self.scores = []
        self.current_mask_idx = 0
        self.click_mode = 1  # 1: foreground, 0: background
        
        # 화면 크기 설정
        self.window_width = 1600
        self.window_height = 1000
        
        # SAM2 모델 초기화
        self.init_sam2()
        
        # OpenCV 윈도우 설정
        cv2.namedWindow('Large SAM GUI', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Large SAM GUI', self.window_width, self.window_height)
        cv2.setMouseCallback('Large SAM GUI', self.mouse_callback)
        
        # 초기 화면 표시
        self.show_instructions()
        
        print("=== 큰 화면 SAM GUI ===")
        print("사용법:")
        print("- 'i': 이미지 로드")
        print("- 'f': 물체 점 추가 모드")
        print("- 'b': 배경 점 추가 모드")
        print("- 's': SAM 실행")
        print("- 'n': 다음 마스크")
        print("- 'p': 이전 마스크")
        print("- 'c': 점들 초기화")
        print("- 'q': 종료")
        print("- 'h': 도움말")
    
    def init_sam2(self):
        """SAM2 모델 초기화"""
        try:
            print("SAM2 모델 로딩 중...")
            
            sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            
            print(f"SAM2 모델 로딩 완료! 디바이스: {device}")
            
        except Exception as e:
            print(f"SAM2 모델 로딩 실패: {str(e)}")
            self.predictor = None
    
    def show_instructions(self):
        """사용법 안내 화면 표시"""
        # 검은 화면 생성
        instructions_img = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        
        # 텍스트 설정
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        color = (255, 255, 255)
        thickness = 2
        
        # 제목
        title = "Large SAM GUI - Segment Anything Model"
        title_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
        title_x = (self.window_width - title_size[0]) // 2
        cv2.putText(instructions_img, title, (title_x, 100), font, font_scale, color, thickness)
        
        # 사용법
        instructions = [
            "사용법:",
            "i - 이미지 로드",
            "f - 물체 점 추가 모드",
            "b - 배경 점 추가 모드",
            "s - SAM 실행",
            "n - 다음 마스크",
            "p - 이전 마스크",
            "c - 점들 초기화",
            "q - 종료",
            "h - 도움말"
        ]
        
        y_start = 200
        for i, instruction in enumerate(instructions):
            y = y_start + i * 50
            cv2.putText(instructions_img, instruction, (50, y), font, 1.0, color, 2)
        
        # 현재 상태
        status = f"현재 모드: {'물체' if self.click_mode == 1 else '배경'} 점 추가"
        cv2.putText(instructions_img, status, (50, self.window_height - 100), font, 1.0, color, 2)
        
        cv2.imshow('Large SAM GUI', instructions_img)
    
    def load_image(self):
        """이미지 로드"""
        file_path = input("이미지 파일 경로를 입력하세요: ").strip()
        
        if not file_path or not os.path.exists(file_path):
            print("파일을 찾을 수 없습니다.")
            return
        
        try:
            self.image = cv2.imread(file_path)
            if self.image is None:
                print("이미지를 로드할 수 없습니다.")
                return
            
            self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.points = []
            self.labels = []
            self.masks = []
            self.scores = []
            self.current_mask_idx = 0
            
            print(f"이미지 로드 완료: {file_path}")
            self.display_image()
            
        except Exception as e:
            print(f"이미지 로드 실패: {str(e)}")
    
    def mouse_callback(self, event, x, y, flags, param):
        """마우스 콜백"""
        if event == cv2.EVENT_LBUTTONDOWN and self.image is not None:
            self.points.append((x, y))
            self.labels.append(self.click_mode)
            
            mode_text = "물체" if self.click_mode == 1 else "배경"
            print(f"점 추가: ({x}, {y}) - {mode_text}")
            
            self.display_image()
    
    def display_image(self):
        """이미지 표시"""
        if self.image is None:
            self.show_instructions()
            return
        
        # 이미지 크기 조정
        h, w = self.image.shape[:2]
        scale = min(self.window_width / w, self.window_height / h) * 0.9
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        display_img = cv2.resize(self.image, (new_w, new_h))
        
        # 큰 화면에 맞게 패딩 추가
        padded_img = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        start_y = (self.window_height - new_h) // 2
        start_x = (self.window_width - new_w) // 2
        padded_img[start_y:start_y+new_h, start_x:start_x+new_w] = display_img
        
        # 점들 표시 (스케일 조정)
        for i, (point, label) in enumerate(zip(self.points, self.labels)):
            scaled_x = int(point[0] * scale) + start_x
            scaled_y = int(point[1] * scale) + start_y
            color = (0, 0, 255) if label == 1 else (255, 0, 0)  # BGR
            cv2.circle(padded_img, (scaled_x, scaled_y), 8, color, -1)
            cv2.putText(padded_img, str(i+1), (scaled_x+15, scaled_y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        
        # 마스크 표시
        if len(self.masks) > 0:
            mask = self.masks[self.current_mask_idx]
            mask_resized = cv2.resize(mask.astype(np.uint8), (new_w, new_h))
            mask_colored = cv2.applyColorMap(mask_resized * 255, cv2.COLORMAP_JET)
            
            # 마스크를 큰 화면에 맞게 패딩
            mask_padded = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
            mask_padded[start_y:start_y+new_h, start_x:start_x+new_w] = mask_colored
            
            padded_img = cv2.addWeighted(padded_img, 0.7, mask_padded, 0.3, 0)
            
            score = self.scores[self.current_mask_idx]
            cv2.putText(padded_img, f"Mask {self.current_mask_idx + 1} (Score: {score:.3f})", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # 현재 상태 표시
        mode_text = f"Mode: {'Foreground' if self.click_mode == 1 else 'Background'}"
        cv2.putText(padded_img, mode_text, (50, self.window_height - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        cv2.imshow('Large SAM GUI', padded_img)
    
    def run_sam(self):
        """SAM 실행"""
        if len(self.points) == 0 or self.predictor is None:
            print("점을 추가하고 SAM2 모델이 로드되었는지 확인하세요.")
            return
        
        try:
            print("SAM 실행 중...")
            
            self.predictor.set_image(self.image_rgb)
            
            input_points = np.array(self.points)
            input_labels = np.array(self.labels)
            
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
            
            self.masks = masks
            self.scores = scores
            self.current_mask_idx = 0
            
            print(f"SAM 실행 완료! 생성된 마스크: {len(masks)}개")
            for i, score in enumerate(scores):
                print(f"  마스크 {i+1}: 점수 {score:.3f}")
            
            self.display_image()
            
        except Exception as e:
            print(f"SAM 실행 실패: {str(e)}")
    
    def next_mask(self):
        """다음 마스크"""
        if len(self.masks) > 1:
            self.current_mask_idx = (self.current_mask_idx + 1) % len(self.masks)
            self.display_image()
            print(f"마스크 {self.current_mask_idx + 1} 표시")
        else:
            print("마스크가 없거나 1개뿐입니다.")
    
    def prev_mask(self):
        """이전 마스크"""
        if len(self.masks) > 1:
            self.current_mask_idx = (self.current_mask_idx - 1) % len(self.masks)
            self.display_image()
            print(f"마스크 {self.current_mask_idx + 1} 표시")
        else:
            print("마스크가 없거나 1개뿐입니다.")
    
    def clear_points(self):
        """점들 초기화"""
        self.points = []
        self.labels = []
        self.masks = []
        self.scores = []
        self.current_mask_idx = 0
        print("점들이 초기화되었습니다.")
        self.display_image()
    
    def save_mask(self):
        """현재 마스크 저장"""
        if len(self.masks) == 0:
            print("저장할 마스크가 없습니다.")
            return
        
        filename = f"mask_{self.current_mask_idx + 1}_score_{self.scores[self.current_mask_idx]:.3f}.png"
        mask = self.masks[self.current_mask_idx]
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        cv2.imwrite(filename, mask_uint8)
        print(f"마스크 저장 완료: {filename}")
    
    def run(self):
        """메인 루프"""
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('i'):
                self.load_image()
            elif key == ord('f'):
                self.click_mode = 1
                print("물체 점 추가 모드")
            elif key == ord('b'):
                self.click_mode = 0
                print("배경 점 추가 모드")
            elif key == ord('s'):
                self.run_sam()
            elif key == ord('n'):
                self.next_mask()
            elif key == ord('p'):
                self.prev_mask()
            elif key == ord('c'):
                self.clear_points()
            elif key == ord('h'):
                print("\n=== 도움말 ===")
                print("'i': 이미지 로드")
                print("'f': 물체 점 추가 모드")
                print("'b': 배경 점 추가 모드")
                print("'s': SAM 실행")
                print("'n': 다음 마스크")
                print("'p': 이전 마스크")
                print("'c': 점들 초기화")
                print("'q': 종료")
                print("'h': 도움말")
        
        cv2.destroyAllWindows()


def main():
    """메인 함수"""
    gui = LargeSAMGUI()
    gui.run()


if __name__ == "__main__":
    main()
