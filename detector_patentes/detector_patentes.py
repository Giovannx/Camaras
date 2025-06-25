import cv2
import pandas as pd
from ultralytics import YOLO
import easyocr
from datetime import datetime
import os
import argparse
import numpy as np
from mss import mss
import time
from collections import deque

# Constantes de rutas
VIDEO_PATH = 'video_camaras.mp4'
STOLEN_CSV = 'patentes_robadas.csv'
ALERTS_CSV = 'alertas_detectadas.csv'
CAPTURES_DIR = 'capturas_alertas'
# Duración en segundos del video que se guarda al detectar una patente robada
CLIP_SECONDS = 30
# Ruta del modelo entrenado para detectar patentes
MODEL_PATH = 'modelo_yolo.pt'  # Reemplazar por el peso entrenado para detectar patentes

def parse_args():
    """Procesa argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Sistema de detección de patentes")
    parser.add_argument('--video', default=VIDEO_PATH,
                        help='Ruta del video a procesar (ignorado si se usa --screen)')
    parser.add_argument('--model', default=MODEL_PATH,
                        help='Ruta del modelo YOLOv8')
    parser.add_argument('--stolen', default=STOLEN_CSV,
                        help='CSV con patentes robadas')
    parser.add_argument('--frame-skip', type=int, default=1,
                        help='Procesar cada N fotogramas')
    parser.add_argument('--screen', action='store_true',
                        help='Capturar la pantalla en tiempo real en lugar de un video')
    return parser.parse_args()


def load_stolen_plates(csv_path: str) -> set:
    """Carga las patentes robadas desde un CSV y las normaliza."""
    df = pd.read_csv(csv_path)
    return set(df['patente'].str.strip().str.upper())


def init_alerts_csv(csv_path: str):
    """Crea el archivo de alertas si no existe."""
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('fecha,hora,patente\n')


def clean_text(text: str) -> str:
    """Corrige errores comunes de OCR."""
    replacements = {
        '0': 'O',
        '1': 'I',
        '5': 'S',
        '6': 'G',
        '8': 'B'
    }
    cleaned = text.upper().replace(' ', '')
    for k, v in replacements.items():
        cleaned = cleaned.replace(k, v)
    # Solo mantener letras y números
    cleaned = ''.join(ch for ch in cleaned if ch.isalnum())
    return cleaned


def save_alert(frame, plate_text: str):
    """Guarda la imagen de la alerta y registra en CSV."""
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H-%M-%S')
    filename = f'{CAPTURES_DIR}/{plate_text}_{date_str}_{time_str}.jpg'
    cv2.imwrite(filename, frame)
    with open(ALERTS_CSV, 'a', encoding='utf-8') as f:
        f.write(f'{date_str},{time_str},{plate_text}\n')


def start_clip_writer(frame, plate_text: str, fps: int):
    """Crea un writer para guardar un clip de video."""
    now = datetime.now()
    ts = now.strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"{CAPTURES_DIR}/{plate_text}_{ts}.mp4"
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    return writer


def draw_label(frame, text, bbox, color=(0, 255, 0)):
    """Dibuja el rectángulo y la etiqueta en la imagen."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color, 2, cv2.LINE_AA)


def main():
    args = parse_args()

    stolen_plates = load_stolen_plates(args.stolen)
    init_alerts_csv(ALERTS_CSV)

    if not os.path.exists(CAPTURES_DIR):
        os.makedirs(CAPTURES_DIR, exist_ok=True)

    # Cargar modelo YOLO y OCR
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"No se encontró el modelo YOLO en {args.model}")
    model = YOLO(args.model)
    reader = easyocr.Reader(['en'], gpu=False)

    if args.screen:
        cap = None
        sct = mss()
        monitor = sct.monitors[1]
        fps = 20
    else:
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1:
            fps = 20
    frame_id = 0
    frame_skip = max(1, args.frame_skip)

    detection_enabled = False
    recording = False
    record_end = 0
    writer = None

    print("Presiona 'O' para iniciar detección y 'P' para salir")

    while True:
        if args.screen:
            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            ret, frame = cap.read()
            if not ret:
                break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('o'):
            detection_enabled = True
        elif key == ord('p'):
            break

        if not detection_enabled:
            cv2.imshow('Video', frame)
            continue

        if frame_id % frame_skip != 0:
            frame_id += 1
            if recording and writer is not None:
                writer.write(frame)
                if time.time() >= record_end:
                    writer.release()
                    writer = None
                    recording = False
            cv2.imshow('Video', frame)
            continue

        try:
            results = model(frame)
        except Exception as e:
            print(f"Error al procesar YOLO: {e}")
            break
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                plate_img = frame[int(y1):int(y2), int(x1):int(x2)]
                if plate_img.size == 0:
                    continue
                try:
                    ocr_result = reader.readtext(
                        plate_img, detail=0, paragraph=False
                    )
                except Exception as e:
                    print(f"Error de OCR: {e}")
                    continue
                if not ocr_result:
                    continue
                plate_text = clean_text(ocr_result[0])
                draw_label(frame, plate_text, (x1, y1, x2, y2))
                if plate_text in stolen_plates:
                    print(f"ALERTA: Patente robada detectada {plate_text}")
                    save_alert(frame, plate_text)
                    if not recording:
                        writer = start_clip_writer(frame, plate_text, int(fps))
                        record_end = time.time() + CLIP_SECONDS
                        recording = True

        if recording and writer is not None:
            writer.write(frame)
            if time.time() >= record_end:
                writer.release()
                writer = None
                recording = False

        cv2.imshow('Video', frame)
        frame_id += 1

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

# Instrucciones de instalación:
# Ejecuta: pip install -r requirements.txt
# Uso básico:
#     python detector_patentes.py --video mi_video.mp4 --stolen mis_patentes.csv
# Para capturar la pantalla en tiempo real:
#     python detector_patentes.py --screen
# Durante la ejecución presiona la tecla 'O' para iniciar la detección
# y la tecla 'P' para finalizar.
