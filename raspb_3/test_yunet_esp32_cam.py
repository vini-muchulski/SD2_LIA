import cv2 as cv

model = 'face_detection_yunet_2023mar.onnx'
input_size = (320, 320)  # Tamanho de entrada do modelo

face_detector = cv.FaceDetectorYN.create(
    model, "", input_size, score_threshold=0.9, nms_threshold=0.3,
    top_k=5000, backend_id=cv.dnn.DNN_BACKEND_OPENCV, target_id=cv.dnn.DNN_TARGET_CPU
)

ip = 'http://192.168.0.120'
stream_url = f"{ip}:81/stream"
cap = cv.VideoCapture(stream_url)

if not cap.isOpened():
    print("Não foi possível conectar ao stream da ESP32-CAM")
    exit()

# Criar a janela antes do loop
cv.namedWindow('Deteccao_Facial', cv.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha na captura de vídeo")
        break

    # Redimensionar o quadro para o tamanho de entrada do modelo
    resized_frame = cv.resize(frame, input_size)

    # Detectar faces no quadro redimensionado
    faces = face_detector.detect(resized_frame)

    # Calcular as escalas de redimensionamento
    scale_x = frame.shape[1] / input_size[0]
    scale_y = frame.shape[0] / input_size[1]

    if faces[1] is not None:
        for face in faces[1]:
            x, y, w, h = face[:4]

            # Ajustar as coordenadas
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)

            # Desenhar a caixa delimitadora
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Desenhar os pontos de referência faciais
            landmarks = face[4:14].reshape((5, 2))
            landmarks[:, 0] *= scale_x
            landmarks[:, 1] *= scale_y
            landmarks = landmarks.astype(int)
            for (lx, ly) in landmarks:
                cv.circle(frame, (lx, ly), 2, (0, 0, 255), -1)

            # Desenhar o centro da face
            cv.circle(frame, (center_x, center_y), 3, (255, 0, 0), -1)

            # Exibir as coordenadas do centro na imagem
            cv.putText(frame, f"Centro: ({center_x}, {center_y})", (x, y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Imprimir as coordenadas no console
            print(f"Centro da face: ({center_x}, {center_y})")
    else:
        print("Nenhuma face detectada.")

    # Exibir o vídeo na janela criada
    cv.imshow('Deteccao_Facial', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
