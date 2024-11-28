import cv2 as cv

# URL do stream da câmera
ip = 'http://192.168.0.107'
stream_url = f"{ip}:81/stream"
     
cap = cv.VideoCapture(stream_url)


if not cap.isOpened():
    print("Não foi possível conectar ao stream da ESP32-CAM")
    exit()

# cv.destroyAllWindows()
# Criar a janela antes do loop
cv.namedWindow('Stream', cv.WINDOW_NORMAL)

# Loop de captura e exibição de vídeo
while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha na captura de vídeo")
        break

    
    cv.imshow('Stream', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv.destroyAllWindows()