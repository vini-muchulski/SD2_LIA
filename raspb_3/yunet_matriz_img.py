import cv2 as cv
import numpy as np

# Carregar o modelo YuNet
model = 'face_detection_yunet_2023mar.onnx'
input_size = (640, 640)

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
    # Obter as dimensões originais da imagem
    height, width = frame.shape[:2]

    # Redimensionar a imagem para o tamanho de entrada do modelo
    resized_frame = cv.resize(frame, input_size)

    # Realizar a detecção facial na imagem redimensionada
    faces = face_detector.detect(resized_frame)

    if faces[1] is not None:
        # Dividir a imagem em uma grade 3x3
        cell_width = width / 3
        cell_height = height / 3

        for face in faces[1]:
            # Escalar as coordenadas de volta para o tamanho original da imagem
            face = face.astype(int)
            x, y, w, h = face[:4]
            x = int(x * width / input_size[0])
            y = int(y * height / input_size[1])
            w = int(w * width / input_size[0])
            h = int(h * height / input_size[1])

            # Desenhar a caixa delimitadora ao redor do rosto
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Desenhar os pontos dos olhos, nariz e boca
            landmarks = face[4:14].reshape((5, 2))
            for (lx, ly) in landmarks:
                lx = int(lx * width / input_size[0])
                ly = int(ly * height / input_size[1])
                cv.circle(frame, (lx, ly), 2, (0, 0, 255), -1)

            # Calcular o centro do rosto
            face_center_x = x + w / 2
            face_center_y = y + h / 2

            # Determinar em qual célula da grade o centro do rosto está localizado
            col = int(face_center_x // cell_width) + 1  # colunas 1, 2 ou 3
            row = int(face_center_y // cell_height) + 1  # linhas 1, 2 ou 3

            # Garantir que os valores de linha e coluna não excedam 3
            col = min(col, 3)
            row = min(row, 3)

            print(f"Rosto localizado na célula da matriz: linha {row}, coluna {col}")
            text = f"Linha {row}, Coluna {col}"
            cv.putText(frame, text, (20, height -20), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

        # Desenhar as linhas da grade na imagem
        for i in range(1, 3):
            # Linhas verticais
            cv.line(frame, (int(cell_width * i), 0), (int(cell_width * i), height), (255, 0, 0), 1)
            # Linhas horizontais
            cv.line(frame, (0, int(cell_height * i)), (width, int(cell_height * i)), (255, 0, 0), 1)

    else:
        print("Nenhum rosto detectado.")

    # Exibir a imagem com as detecções e a grade
    cv.imshow('Deteccao Facial com YuNet', frame)
    cv.waitKey(0)
    cv.destroyAllWindows()
