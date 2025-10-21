#!/usr/bin/env python3
"""
SAM GUI 애플리케이션
이미지를 로드하고 클릭한 포인트로 SAM을 실행하는 GUI
"""
import os
# DPI 스케일링 설정
os.environ["GDK_SCALE"] = "2"
os.environ["GDK_DPI_SCALE"] = "0.5"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCALE_FACTOR"] = "2"

import sys
import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# SAM2 import
from sam2.sam2.build_sam import build_sam2
from sam2.sam2.sam2_image_predictor import SAM2ImagePredictor


class SAMGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SAM GUI - Segment Anything Model")
        
        # 화면 크기 감지 및 윈도우 크기 설정
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # 화면 크기의 80% 사용
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        
        # 중앙에 윈도우 배치
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(800, 600)  # 최소 크기 설정
        
        # 변수 초기화
        self.image = None
        self.image_rgb = None
        self.original_image = None
        self.predictor = None
        self.points = []
        self.labels = []
        self.masks = []
        self.scores = []
        self.current_mask_idx = 0
    
    def safe_bool_check(self, obj):
        """안전한 boolean 체크"""
        if obj is None:
            return False
        if hasattr(obj, '__len__'):
            return len(obj) > 0
        return bool(obj)
        
        # GUI 구성
        self.setup_ui()
        
        # SAM2 모델 초기화
        self.init_sam2()
    
    def setup_ui(self):
        """UI 구성"""
        # 메인 프레임
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 왼쪽 패널 (이미지 표시)
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 이미지 표시용 Canvas
        self.canvas_frame = ttk.Frame(left_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Matplotlib Figure - 더 큰 크기로 설정
        self.fig = Figure(figsize=(12, 9), dpi=120)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("이미지를 로드하고 물체를 클릭하세요", fontsize=16)
        self.ax.axis('off')
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 마우스 클릭 이벤트 바인딩
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
        # 오른쪽 패널 (컨트롤) - 더 넓게 설정
        right_frame = ttk.Frame(main_frame, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # 파일 로드 버튼
        ttk.Button(right_frame, text="이미지 로드", command=self.load_image).pack(pady=5, fill=tk.X)
        
        # 구분선
        ttk.Separator(right_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Annotation 컨트롤
        annotation_frame = ttk.LabelFrame(right_frame, text="Annotation 컨트롤")
        annotation_frame.pack(fill=tk.X, pady=5)
        
        # 점 추가/제거
        ttk.Button(annotation_frame, text="점 추가 (물체)", 
                  command=lambda: self.set_click_mode(1)).pack(pady=2, fill=tk.X)
        ttk.Button(annotation_frame, text="점 추가 (배경)", 
                  command=lambda: self.set_click_mode(0)).pack(pady=2, fill=tk.X)
        ttk.Button(annotation_frame, text="점 제거", 
                  command=self.remove_last_point).pack(pady=2, fill=tk.X)
        ttk.Button(annotation_frame, text="모든 점 제거", 
                  command=self.clear_points).pack(pady=2, fill=tk.X)
        
        # SAM 실행
        ttk.Button(annotation_frame, text="SAM 실행", 
                  command=self.run_sam).pack(pady=5, fill=tk.X)
        
        # 구분선
        ttk.Separator(right_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # 마스크 컨트롤
        mask_frame = ttk.LabelFrame(right_frame, text="마스크 컨트롤")
        mask_frame.pack(fill=tk.X, pady=5)
        
        # 마스크 선택
        ttk.Label(mask_frame, text="마스크 선택:").pack()
        self.mask_var = tk.StringVar(value="0")
        self.mask_combo = ttk.Combobox(mask_frame, textvariable=self.mask_var, 
                                      state="readonly", width=10)
        self.mask_combo.pack(pady=2)
        self.mask_combo.bind('<<ComboboxSelected>>', self.on_mask_change)
        
        # 마스크 저장
        ttk.Button(mask_frame, text="마스크 저장", 
                  command=self.save_mask).pack(pady=2, fill=tk.X)
        ttk.Button(mask_frame, text="모든 마스크 저장", 
                  command=self.save_all_masks).pack(pady=2, fill=tk.X)
        
        # 구분선
        ttk.Separator(right_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # 정보 표시
        info_frame = ttk.LabelFrame(right_frame, text="정보")
        info_frame.pack(fill=tk.X, pady=5)
        
        self.info_text = tk.Text(info_frame, height=10, width=40, font=("Arial", 10))
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # 스크롤바
        scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        # 초기 정보 표시
        self.update_info("SAM GUI가 준비되었습니다.\n이미지를 로드하세요.")
    
    def init_sam2(self):
        """SAM2 모델 초기화"""
        try:
            self.update_info("SAM2 모델 로딩 중...")
            self.root.update()
            
            # SAM2 설정
            sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 모델 빌드
            sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            
            self.update_info(f"SAM2 모델 로딩 완료!\n디바이스: {device}")
            
        except Exception as e:
            self.update_info(f"SAM2 모델 로딩 실패: {str(e)}")
            messagebox.showerror("에러", f"SAM2 모델을 로드할 수 없습니다:\n{str(e)}")
    
    def load_image(self):
        """이미지 로드"""
        file_path = filedialog.askopenfilename(
            title="이미지 선택",
            filetypes=[("이미지 파일", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                # 이미지 로드
                self.image = cv2.imread(file_path)
                if self.image is None:
                    raise ValueError("이미지를 로드할 수 없습니다.")
                
                # RGB로 변환
                self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                self.original_image = self.image_rgb.copy()
                
                # 이미지 표시
                self.display_image()
                
                # 점과 마스크 초기화
                self.points = []
                self.labels = []
                self.masks = []
                self.scores = []
                self.current_mask_idx = 0
                self.update_mask_combo()
                
                self.update_info(f"이미지 로드 완료: {os.path.basename(file_path)}\n크기: {self.image_rgb.shape}")
                
            except Exception as e:
                self.update_info(f"이미지 로드 실패: {str(e)}")
                messagebox.showerror("에러", f"이미지를 로드할 수 없습니다:\n{str(e)}")
    
    def display_image(self):
        """이미지 표시"""
        if self.image_rgb is not None:
            self.ax.clear()
            self.ax.imshow(self.image_rgb)
            self.ax.set_title("이미지를 로드하고 물체를 클릭하세요")
            self.ax.axis('off')
            
            # 점들 표시
            self.draw_points()
            
            self.canvas.draw()
    
    def draw_points(self):
        """클릭한 점들 표시"""
        for i, (point, label) in enumerate(zip(self.points, self.labels)):
            color = 'red' if label == 1 else 'blue'
            self.ax.plot(point[0], point[1], 'o', color=color, markersize=8)
            self.ax.text(point[0], point[1], str(i+1), color='white', 
                        fontsize=10, ha='center', va='center')
    
    def on_click(self, event):
        """마우스 클릭 이벤트"""
        if event.inaxes != self.ax or self.image_rgb is None:
            return
        
        if event.button == 1:  # 왼쪽 클릭
            x, y = int(event.xdata), int(event.ydata)
            
            # 현재 클릭 모드에 따라 라벨 설정
            if hasattr(self, 'click_mode'):
                self.points.append((x, y))
                self.labels.append(self.click_mode)
                
                # 이미지 다시 표시
                self.display_image()
                
                mode_text = "물체" if self.click_mode == 1 else "배경"
                self.update_info(f"점 추가: ({x}, {y}) - {mode_text}")
            else:
                self.update_info("먼저 '점 추가' 버튼을 클릭하세요.")
    
    def set_click_mode(self, label):
        """클릭 모드 설정"""
        self.click_mode = label
        mode_text = "물체" if label == 1 else "배경"
        self.update_info(f"클릭 모드: {mode_text} 점 추가")
    
    def remove_last_point(self):
        """마지막 점 제거"""
        if self.points:
            removed_point = self.points.pop()
            removed_label = self.labels.pop()
            mode_text = "물체" if removed_label == 1 else "배경"
            self.update_info(f"점 제거: {removed_point} - {mode_text}")
            self.display_image()
        else:
            self.update_info("제거할 점이 없습니다.")
    
    def clear_points(self):
        """모든 점 제거"""
        self.points = []
        self.labels = []
        self.masks = []
        self.scores = []
        self.current_mask_idx = 0
        self.update_mask_combo()
        self.display_image()
        self.update_info("모든 점이 제거되었습니다.")
    
    def run_sam(self):
        """SAM 실행"""
        if not self.points or not self.predictor:
            self.update_info("점을 추가하고 SAM2 모델이 로드되었는지 확인하세요.")
            return
        
        try:
            self.update_info("SAM 실행 중...")
            self.root.update()
            
            # SAM2 실행
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
            
            # 마스크 콤보박스 업데이트
            self.update_mask_combo()
            
            # 마스크 표시
            self.display_mask()
            
            self.update_info(f"SAM 실행 완료!\n생성된 마스크: {len(masks)}개")
            
        except Exception as e:
            self.update_info(f"SAM 실행 실패: {str(e)}")
            messagebox.showerror("에러", f"SAM 실행 중 오류가 발생했습니다:\n{str(e)}")
    
    def display_mask(self):
        """마스크 표시"""
        if len(self.masks) == 0:
            return
        
        mask = self.masks[self.current_mask_idx]
        score = self.scores[self.current_mask_idx]
        
        self.ax.clear()
        self.ax.imshow(self.image_rgb)
        self.ax.imshow(mask, alpha=0.6, cmap='Reds')
        self.ax.set_title(f"마스크 {self.current_mask_idx + 1} (점수: {score:.3f})")
        self.ax.axis('off')
        
        # 점들 표시
        self.draw_points()
        
        self.canvas.draw()
    
    def update_mask_combo(self):
        """마스크 콤보박스 업데이트"""
        if len(self.masks) > 0:
            mask_options = [f"마스크 {i+1} (점수: {score:.3f})" 
                           for i, score in enumerate(self.scores)]
            self.mask_combo['values'] = mask_options
            self.mask_combo.current(0)
        else:
            self.mask_combo['values'] = []
    
    def on_mask_change(self, event):
        """마스크 선택 변경"""
        if len(self.masks) > 0:
            self.current_mask_idx = self.mask_combo.current()
            self.display_mask()
    
    def save_mask(self):
        """현재 마스크 저장"""
        if len(self.masks) == 0:
            self.update_info("저장할 마스크가 없습니다.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="마스크 저장",
            defaultextension=".png",
            filetypes=[("PNG 파일", "*.png")]
        )
        
        if file_path:
            try:
                mask = self.masks[self.current_mask_idx]
                mask_uint8 = (mask * 255).astype(np.uint8)
                cv2.imwrite(file_path, mask_uint8)
                self.update_info(f"마스크 저장 완료: {os.path.basename(file_path)}")
            except Exception as e:
                self.update_info(f"마스크 저장 실패: {str(e)}")
                messagebox.showerror("에러", f"마스크 저장 중 오류가 발생했습니다:\n{str(e)}")
    
    def save_all_masks(self):
        """모든 마스크 저장"""
        if len(self.masks) == 0:
            self.update_info("저장할 마스크가 없습니다.")
            return
        
        dir_path = filedialog.askdirectory(title="마스크 저장 폴더 선택")
        
        if dir_path:
            try:
                for i, (mask, score) in enumerate(zip(self.masks, self.scores)):
                    filename = f"mask_{i+1}_score_{score:.3f}.png"
                    filepath = os.path.join(dir_path, filename)
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    cv2.imwrite(filepath, mask_uint8)
                
                self.update_info(f"모든 마스크 저장 완료: {dir_path}")
            except Exception as e:
                self.update_info(f"마스크 저장 실패: {str(e)}")
                messagebox.showerror("에러", f"마스크 저장 중 오류가 발생했습니다:\n{str(e)}")
    
    def update_info(self, message):
        """정보 텍스트 업데이트"""
        self.info_text.insert(tk.END, f"{message}\n")
        self.info_text.see(tk.END)
        self.root.update()


def main():
    """메인 함수"""
    root = tk.Tk()
    
    # DPI 스케일링 설정
    try:
        current_scaling = root.tk.call('tk', 'scaling')
        print(f"Current scaling: {current_scaling}")
        
        # 화면 DPI에 따라 스케일링 조정
        screen_dpi = root.winfo_fpixels('1i')
        print(f"Screen DPI: {screen_dpi}")
        
        if screen_dpi > 120:  # 고해상도 화면
            new_scaling = 2.0
        elif screen_dpi > 96:  # 중간 해상도
            new_scaling = 1.5
        else:  # 일반 해상도
            new_scaling = 1.0
            
        root.tk.call('tk', 'scaling', new_scaling)
        print(f"New scaling: {new_scaling}")
        
    except Exception as e:
        print(f"Scaling 설정 실패: {e}")
        # 기본값으로 설정
        root.tk.call('tk', 'scaling', 1.5)
    
    app = SAMGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
