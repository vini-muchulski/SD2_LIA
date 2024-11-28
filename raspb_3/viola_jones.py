# https://www.youtube.com/playlist?list=PLsyobOqUhkthjvmA_s7tTjb7V2EiwYYGC
#https://how2electronics.com/esp32-cam-based-object-detection-identification-with-opencv/

import cv2

# Carregar o classificador de face pré-treinado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


"""WebCamera = cv2.VideoCapture(0)

#REAJUSTE DE RESOLUCAO
WebCamera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
WebCamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)"""





# URL completa para acessar o stream de vídeo do ESP32-CAM
ip = "http://192.168.0.120"

stream_url = f"{ip}:81/stream"
#http://192.168.0.120:81/stream

# Inicializa a captura de vídeo com o URL do stream
WebCamera = cv2.VideoCapture(stream_url)

# Criar a janela antes do loop
cv2.namedWindow('Deteccao_Facial', cv2.WINDOW_NORMAL)

while True:
    conectado, img = WebCamera.read()

    #img = cv2.imread('imagem2.jpg')


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for i, (x, y, w, h) in enumerate(faces):
        
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Calcular o centro de massa
        center_x = x + w // 2
        center_y = y + h // 2
        
        
        print(f"Face {i+1} - Centro : (x={center_x}, y={center_y})")
        
        
        cross_size = min(w, h) // 4
        cv2.line(img, (center_x - cross_size, center_y), (center_x + cross_size, center_y), (0, 255, 0), 2)
        cv2.line(img, (center_x, center_y - cross_size), (center_x, center_y + cross_size), (0, 255, 0), 2)


    cv2.imshow('Face Detection - Alvo ', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
WebCamera.release()
cv2.destroyAllWindows()