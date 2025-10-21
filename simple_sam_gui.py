#!/usr/bin/env python3
"""
간단한 SAM GUI 애플리케이션
OpenCV를 사용한 간단한 인터페이스
"""

import cv2
import numpy as np
import torch
import os
from sam2.sam2.build_sam import build_sam2
from sam2.sam2.sam2_image_predictor import SAM2ImagePredictor


class SimpleSAMGUI:
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
        
        # SAM2 모델 초기화
        self.init_sam2()
        self.file_path = ""
        
        # OpenCV 윈도우 설정
        cv2.namedWindow('SAM GUI', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('SAM GUI', self.mouse_callback)
        
        print("=== 간단한 SAM GUI ===")
        print("사용법:")
        print("- 'i': 이미지 로드")
        print("- 'f': 물체 점 추가 모드")
        print("- 'b': 배경 점 추가 모드")
        print("- 's': SAM 실행 (최고 점수 마스크 자동 선택 및 저장)")
        print("- 'n': 다음 마스크")
        print("- 'p': 이전 마스크")
        print("- 'c': 점들 초기화")
        print("- 'q': 종료")
        print("- 'h': 도움말")
        print("\n특징:")
        print("- SAM 실행 시 최고 점수 마스크 자동 선택")
        print("- 마스크는 255(물체)/0(배경) 형태로 저장")
        print("- 자동 저장: auto_best_mask_score_XXX.png")
    
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
    
    def load_image(self):
        """이미지 로드"""
        file_path = input("이미지 파일 경로를 입력하세요: ").strip()
        self.file_path = file_path
        if not f"captured_images/{file_path}" or not os.path.exists(f"captured_images/{file_path}"):
            print("파일을 찾을 수 없습니다.")
            return
        
        try:
            self.image = cv2.imread(f"captured_images/{file_path}")
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
            return
        
        display_img = self.image.copy()
        
        # 점들 표시
        for i, (point, label) in enumerate(zip(self.points, self.labels)):
            color = (0, 0, 255) if label == 1 else (255, 0, 0)  # BGR
            cv2.circle(display_img, point, 5, color, -1)
            cv2.putText(display_img, str(i+1), (point[0]+10, point[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 마스크 표시
        if len(self.masks) > 0:
            mask = self.masks[self.current_mask_idx]
            mask_colored = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
            display_img = cv2.addWeighted(display_img, 0.7, mask_colored, 0.3, 0)
            
            score = self.scores[self.current_mask_idx]
            cv2.putText(display_img, f"Mask {self.current_mask_idx + 1} (Score: {score:.3f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('SAM GUI', display_img)
    
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
            
            # 최고 점수 마스크 자동 선택
            self.current_mask_idx = np.argmax(scores)
            best_score = scores[self.current_mask_idx]
            
            print(f"SAM 실행 완료! 생성된 마스크: {len(masks)}개")
            for i, score in enumerate(scores):
                marker = " ← 최고 점수" if i == self.current_mask_idx else ""
                print(f"  마스크 {i+1}: 점수 {score:.3f}{marker}")
            
            print(f"자동으로 최고 점수 마스크 {self.current_mask_idx + 1} 선택됨 (점수: {best_score:.3f})")
            
            # 자동으로 최고 점수 마스크 저장
            self.auto_save_best_mask()
            
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
        """현재 마스크 저장 (255/0 형태)"""
        if len(self.masks) == 0:
            print("저장할 마스크가 없습니다.")
            return
        
        mask = self.masks[self.current_mask_idx]
        score = self.scores[self.current_mask_idx]
        
        # 마스크를 255/0 형태로 변환
        mask_binary = np.where(mask, 255, 0).astype(np.uint8)
        
        filename = f"best_mask_score_{score:.3f}.png"
        cv2.imwrite(filename, mask_binary)
        print(f"최고 점수 마스크 저장 완료: {filename}")
        print(f"마스크 형태: 물체 부분=255, 배경 부분=0")
    
    def auto_save_best_mask(self):
        """자동으로 최고 점수 마스크 저장"""
        if len(self.masks) == 0:
            return
        
        mask = self.masks[self.current_mask_idx]
        score = self.scores[self.current_mask_idx]
        
        # 마스크를 255/0 형태로 변환
        mask_binary = np.where(mask, 255, 0).astype(np.uint8)
        
        filename = f"captured_images/mask_{self.file_path}.png"
        cv2.imwrite(filename, mask_binary)
        print(f"자동 저장 완료: {filename}")
    
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
                print("'s': SAM 실행 (최고 점수 마스크 자동 선택 및 저장)")
                print("'n': 다음 마스크")
                print("'p': 이전 마스크")
                print("'c': 점들 초기화")
                print("'q': 종료")
                print("'h': 도움말")
                print("\n특징:")
                print("- SAM 실행 시 최고 점수 마스크 자동 선택")
                print("- 마스크는 255(물체)/0(배경) 형태로 저장")
                print("- 자동 저장: auto_best_mask_score_XXX.png")
        
        cv2.destroyAllWindows()


def main():
    """메인 함수"""
    gui = SimpleSAMGUI()
    gui.run()


if __name__ == "__main__":
    main()

