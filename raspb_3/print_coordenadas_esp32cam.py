import cv2 as cv

model = 'face_detection_yunet_2023mar.onnx'
input_size = (320, 320)  # Tamanho de entrada do modelo

face_detector = cv.FaceDetectorYN.create(
    model, "", input_size, score_threshold=0.9, nms_threshold=0.3,
    top_k=5000, backend_id=cv.dnn.DNN_BACKEND_OPENCV, target_id=cv.dnn.DNN_TARGET_CPU
)

ip = 'http://192.168.0.107'
stream_url = f"{ip}:81/stream"
cap = cv.VideoCapture(stream_url)

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
            x, y, w, h = face[:4].astype(int)
            center_x = x + w / 2
            center_y = y + h / 2
            print(f"Centro da face: ({center_x}, {center_y})")
    else:
        print("Nenhuma face detectada.")

    # Pequena pausa para não sobrecarregar o sistema
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
