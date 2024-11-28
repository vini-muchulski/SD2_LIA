import cv2 as cv
import numpy as np

# Carregar o modelo YuNet
model = 'face_detection_yunet_2023mar.onnx'
input_size = (640,640)


face_detector = cv.FaceDetectorYN.create(
    model, "", input_size, score_threshold=0.8, nms_threshold=0.3,
    top_k=5000, backend_id=cv.dnn.DNN_BACKEND_OPENCV, target_id=cv.dnn.DNN_TARGET_CPU
)

# Carregar a imagem
image_path = "img_lateral.jpeg"
frame = cv.imread(image_path)

if frame is None:
    print(f"Erro ao carregar a imagem {image_path}")
else:
    # Redimensionar a imagem para o tamanho de entrada do modelo
    resized_frame = cv.resize(frame, input_size)

    # Realizar a detecção facial na imagem redimensionada
    faces = face_detector.detect(resized_frame)

    if faces[1] is not None:
        for face in faces[1]:
            # Escalar as coordenadas de volta para o tamanho original da imagem
            face = face.astype(int)
            x, y, w, h = face[:4]
            x = int(x * frame.shape[1] / input_size[0])
            y = int(y * frame.shape[0] / input_size[1])
            w = int(w * frame.shape[1] / input_size[0])
            h = int(h * frame.shape[0] / input_size[1])

            # Desenhar a caixa delimitadora ao redor do rostoquero 
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Desenhar os pontos dos olhos, nariz e boca
            landmarks = face[4:14].reshape((5, 2))
            for (lx, ly) in landmarks:
                lx = int(lx * frame.shape[1] / input_size[0])
                ly = int(ly * frame.shape[0] / input_size[1])
                cv.circle(frame, (lx, ly), 2, (0, 0, 255), -1)

    # Exibir a imagem com as detecções
    cv.imshow('Deteccao Facial com YuNet', frame)
    cv.waitKey(0)
    cv.destroyAllWindows()
