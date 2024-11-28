import cv2 as cv

# Carregar o modelo YuNet
model = 'face_detection_yunet_2023mar.onnx'
input_size = (640, 640)

face_detector = cv.FaceDetectorYN.create(
    model, "", input_size, score_threshold=0.8, nms_threshold=0.3,
    top_k=5000, backend_id=cv.dnn.DNN_BACKEND_OPENCV, target_id=cv.dnn.DNN_TARGET_CPU
)

# Carregar o vídeo
video_path = "sora.mp4"  # Substitua pelo caminho do seu vídeo
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Erro ao abrir o vídeo {video_path}")
else:
    while True:
        ret, frame = cap.read()

        if not ret:
            break  # Fim do vídeo

        # Obter as dimensões do quadro
        height, width = frame.shape[:2]

        # Redimensionar o quadro para o tamanho de entrada do modelo
        resized_frame = cv.resize(frame, input_size)

        # Realizar a detecção facial no quadro redimensionado
        faces = face_detector.detect(resized_frame)

        if faces[1] is not None:
            # Dividir o quadro em uma grade 3x3
            cell_width = width / 3
            cell_height = height / 3

            for face in faces[1]:
                # Escalar as coordenadas de volta para o tamanho original do quadro
                face = face.astype(int)
                x, y, w, h = face[:4]
                x = int(x * width / input_size[0])
                y = int(y * height / input_size[1])
                w = int(w * width / input_size[0])
                h = int(h * height / input_size[1])

                # Desenhar a caixa delimitadora ao redor do rosto
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Calcular o centro do rosto
                face_center_x = x + w / 2
                face_center_y = y + h / 2

                # Determinar em qual célula da grade o centro do rosto está localizado
                col = int(face_center_x // cell_width) + 1  # colunas 1, 2 ou 3
                row = int(face_center_y // cell_height) + 1  # linhas 1, 2 ou 3

                # Garantir que os valores de linha e coluna não excedam 3
                col = min(col, 3)
                row = min(row, 3)

                # Imprimir a célula da matriz correspondente
                print(f"Rosto localizado na célula da matriz: linha {row}, coluna {col}")

            # Desenhar as linhas da grade no quadro
            for i in range(1, 3):
                # Linhas verticais
                cv.line(frame, (int(cell_width * i), 0), (int(cell_width * i), height), (255, 0, 0), 1)
                # Linhas horizontais
                cv.line(frame, (0, int(cell_height * i)), (width, int(cell_height * i)), (255, 0, 0), 1)

        # Exibir o quadro com as detecções e a grade
        cv.imshow('Deteccao Facial com YuNet', frame)

        # Parar o vídeo ao pressionar a tecla 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
