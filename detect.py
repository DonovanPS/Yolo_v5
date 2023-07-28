#modelo = torch.jit.load('D:/U/8/Inteligencia/Proyecto/Codigo/marioDetect_2.pt', map_location=device)
import torch
import cv2
import numpy as np
import pyautogui

# Definir la resolución de entrada deseada para el modelo YOLOv5s
INPUT_RESOLUTION = (640, 480)  # Puedes ajustar esto según tus necesidades

def get_user_selection():
    print("Por favor, selecciona el área de la pantalla a capturar.")
    print("Presiona la tecla 'c' para confirmar la selección.")

    # Captura la pantalla completa
    full_screenshot = pyautogui.screenshot()
    full_screenshot = np.array(full_screenshot)

    # Crea una copia para dibujar el rectángulo seleccionado por el usuario
    img_copy = full_screenshot.copy()
    x1, y1, x2, y2 = -1, -1, -1, -1
    drawing = False

    def draw_rectangle(event, x, y, flags, param):
        nonlocal x1, y1, x2, y2, drawing, img_copy

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            x1, y1 = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x2, y2 = x, y
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.namedWindow('Selección de área')
    cv2.setMouseCallback('Selección de área', draw_rectangle)

    while True:
        cv2.imshow('Selección de área', img_copy)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            break

    cv2.destroyAllWindows()
    return x1, y1, x2, y2

# Obtiene las coordenadas del área seleccionada por el usuario
x1, y1, x2, y2 = get_user_selection()

print(f"Coordenadas seleccionadas: (x1, y1): ({x1}, {y1}), (x2, y2): ({x2}, {y2})")

# Verificar si CUDA está disponible y usarlo si es posible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Cargar el modelo personalizado
#modelo = torch.jit.load('D:/U/8/Inteligencia/Proyecto/Codigo/marioDetect_2.pt', map_location=device)
modelo = torch.hub.load('ultralytics/yolov5', 'custom', path='D:/U/8/Inteligencia/Proyecto/Codigo/marioDetect.pt', force_reload=True)

modelo = modelo.to(device)

while True:
    # Capturar la pantalla completa
    full_screenshot = pyautogui.screenshot()
    full_screenshot = np.array(full_screenshot)

    # Obtener la región seleccionada de la captura completa
    screenshot = full_screenshot[y1:y2, x1:x2]

    # Redimensionar la imagen para mejorar la eficiencia en la CPU
    screenshot = cv2.resize(screenshot, INPUT_RESOLUTION)

    # Convertir de RGB a BGR (OpenCV usa BGR por defecto)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

    # Detectar objetos utilizando YOLOv5
    results = modelo(screenshot)

    # Obtener las coordenadas y etiquetas de los objetos detectados
    detecciones = results.pandas().xyxy[0]

    # Dibujar rectángulos y etiquetas en la imagen
    for index, deteccion in detecciones.iterrows():
        x_min, y_min, x_max, y_max = int(deteccion['xmin']), int(deteccion['ymin']), int(deteccion['xmax']), int(
            deteccion['ymax'])
        etiqueta = deteccion['name']
        cv2.rectangle(screenshot, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(screenshot, etiqueta, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar la porción de la pantalla con las detecciones en una ventana
    cv2.imshow('Detección de Objetos', screenshot)

    # Detener el bucle cuando se presione la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerrar todas las ventanas abiertas
cv2.destroyAllWindows()
