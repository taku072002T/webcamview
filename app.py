import cv2
import numpy as np
import streamlit as st

def analyze_brightness(frame, levels=9):
    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 明るさの範囲を9段階に分割
    height, width = gray.shape
    cell_h, cell_w = height // 32, width // 32
    
    result = np.zeros_like(gray)
    
    for i in range(32):
        for j in range(32):
            cell = gray[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            avg_brightness = np.mean(cell)
            
            # 明るさを1-9の数字にマッピング
            brightness_level = int(avg_brightness / 255 * levels) + 1
            
            # 数字を描画
            cv2.putText(frame, str(brightness_level), 
                        (j*cell_w + cell_w//2 - 10, i*cell_h + cell_h//2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

# Webカメラを開く
cap = cv2.VideoCapture(1)
placeholder = st.empty()
while True:
    # フレームを読み込む
    ret, frame = cap.read()
    if not ret:
        break
    
    # フレームを左右反転
    frame = cv2.flip(frame, 1)
    
    # 明るさを解析し、数字を表示
    processed_frame = analyze_brightness(frame)
    # 画像を一旦初期化
    # 処理後の映像を表示
    placeholder.image(processed_frame)


# リソースを解放
cap.release()
cv2.destroyAllWindows()