import cv2 as cv

model = 'face_detection_yunet_2023mar.onnx'
input_size = (640, 640)

face_detector = cv.FaceDetectorYN.create(
    model, "", input_size, score_threshold=0.8, nms_threshold=0.3,
    top_k=5000, backend_id=cv.dnn.DNN_BACKEND_OPENCV, target_id=cv.dnn.DNN_TARGET_CPU
)

video_path = "sora.mp4"
cap = cv.VideoCapture(0)

def get_position(row, col):
    switch = {
        (1, 1): "position(top_left)",
        (1, 2): "position(top_center)",
        (1, 3): "position(top_right)",
        (2, 1): "position(middle_left)",
        (2, 2): "position(middle_center)",
        (2, 3): "position(middle_right)",
        (3, 1): "position(bottom_left)",
        (3, 2): "position(bottom_center)",
        (3, 3): "position(bottom_right)"
    }
    return switch.get((row, col), "position(unknown)")

if not cap.isOpened():
    print(f"Erro ao abrir o vídeo {video_path}")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        resized_frame = cv.resize(frame, input_size)
        faces = face_detector.detect(resized_frame)

        if faces[1] is not None:
            cell_width = width / 3
            cell_height = height / 3

            for face in faces[1]:
                face = face.astype(int)
                x, y, w, h = face[:4]
                x = int(x * width / input_size[0])
                y = int(y * height / input_size[1])
                w = int(w * width / input_size[0])
                h = int(h * height / input_size[1])
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                face_center_x = x + w / 2
                face_center_y = y + h / 2
                col = int(face_center_x // cell_width) + 1
                row = int(face_center_y // cell_height) + 1
                col = min(col, 3)
                row = min(row, 3)

                position = get_position(row, col)
                print(f"Rosto localizado na célula da matriz: {position}")
                text = position
                cv.putText(frame, text, (20, height -20), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

            for i in range(1, 3):
                cv.line(frame, (int(cell_width * i), 0), (int(cell_width * i), height), (255, 0, 0), 1)
                cv.line(frame, (0, int(cell_height * i)), (width, int(cell_height * i)), (255, 0, 0), 1)

        cv.imshow('Deteccao Facial com YuNet', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
