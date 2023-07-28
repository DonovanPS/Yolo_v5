import cv2
import numpy as np
import pyautogui


def get_user_selection():
    print("Por favor, selecciona el área de la pantalla a capturar.")
    print("Presiona la tecla 'c' para confirmar la selección.")

    # Captura la pantalla
    screenshot = pyautogui.screenshot()
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

    # Crea una copia para dibujar el rectángulo seleccionado por el usuario
    img_copy = screenshot.copy()
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

while True:
    # Captura la porción de la pantalla
    screenshot = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1))
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

    # Mostrar la porción de la pantalla en una ventana
    cv2.imshow('Porción de la Pantalla', screenshot)

    # Detener el bucle cuando se presione la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerrar todas las ventanas abiertas
cv2.destroyAllWindows()
