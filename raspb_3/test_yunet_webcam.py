import cv2 as cv
import numpy as np


model = 'face_detection_yunet_2023mar.onnx'
input_size = (320, 320)  # Tamanho de entrada do modelo


face_detector = cv.FaceDetectorYN.create(
    model, "", input_size, score_threshold=0.9, nms_threshold=0.3,
    top_k=5000, backend_id=cv.dnn.DNN_BACKEND_OPENCV, target_id=cv.dnn.DNN_TARGET_CPU
)


"""ip = 'http://192.168.0.107'
stream_url = f"{ip}:81/stream"
cap = cv.VideoCapture(stream_url)"""

# Captura de vídeo da webcam
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Não foi possível conectar ao stream da ESP32-CAM")
    exit()

ret, frame = cap.read()
if not ret:
    print("Falha na captura de vídeo")
    exit()

frame_height, frame_width = frame.shape[:2]
face_detector.setInputSize((frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha na captura de vídeo")
        break
 
    faces = face_detector.detect(frame)

    if faces[1] is not None:
        for face in faces[1]:
            # Extrair as coordenadas da caixa delimitadora
            # Desenhar a caixa delimitadora ao redor do rosto
            # Desenhar os pontos dos olhos, nariz e boca
            
            x, y, w, h = face[:4].astype(int)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            landmarks = face[4:14].reshape((5, 2)).astype(int)
            
            
            for (lx, ly) in landmarks:
                cv.circle(frame, (lx, ly), 2, (0, 0, 255), -1)

    
    cv.imshow('Deteccaoo Facial com YuNet', frame)

    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
