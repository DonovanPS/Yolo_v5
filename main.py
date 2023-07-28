#pip install opencv-python
#pip install numpy
#pip install pyautogui

#pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt


import cv2
import numpy as np
import pyautogui

# Coordenadas de la porción de la pantalla que deseas capturar
x1, y1, x2, y2 = 300, 10, 500, 500

while True:
    # Capturar la porción de la pantalla
    screenshot = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1))
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

    # Mostrar la porción de la pantalla en una ventana
    cv2.imshow('Porción de la Pantalla', screenshot)

    # Verificar si se ha cerrado la ventana
    if cv2.waitKey(1) & 0xFF == 27:  # 27 es el código ASCII para la tecla 'Esc'
        break

# Cerrar todas las ventanas abiertas
cv2.destroyAllWindows()
