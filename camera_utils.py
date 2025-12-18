import cv2
import numpy as np
import os
import time
from logger import logger

# === ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ ОПТИМИЗАЦИИ ДЛЯ RASPBERRY PI 5 ===
TARGET_RESOLUTION = (320, 240)  # Было (480, 320) → снижено для 4 камер
TARGET_FPS = 8                  # Оставлено 8 FPS — достаточно и безопасно
FACE_DETECTION_SCALE = 1.5      # Уменьшение кадра перед детекцией лиц (ускорение)
JPEG_QUALITY = 70               # Качество JPEG для потоковой передачи

def initialize_cameras(camera_indices, target_resolution=TARGET_RESOLUTION, target_fps=TARGET_FPS):
    """Инициализация камер с пониженным разрешением, MJPG и паузой для Raspberry Pi"""
    caps = []
    for idx in camera_indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)  # Явное указание V4L2 бэкенда
        if cap.isOpened():
            # 🔑 КЛЮЧЕВЫЕ ОПТИМИЗАЦИИ:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # MJPG
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_resolution[1])
            cap.set(cv2.CAP_PROP_FPS, target_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Минимальный буфер

            # Проверка реальных параметров
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            logger.success(f"Camera {idx} initialized: {w}x{h} @ {fps:.1f} FPS (MJPG)")

            # 🔑 Пауза между камерами — критично для USB-стабильности
            time.sleep(0.2)
        else:
            logger.error(f"Failed to initialize camera {idx}")
        caps.append(cap)
    return caps


def release_cameras(caps):
    """Освобождение ресурсов камер"""
    for cap in caps:
        if cap and cap.isOpened():
            cap.release()
    logger.info("Camera resources have been released")


def create_video_grid(frames, grid_size=(2, 2), output_size=(640, 480)):
    """Создание сетки из кадров (все кадры уже в TARGET_RESOLUTION)"""
    if not frames:
        return np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    
    # Рассчитываем размер ячейки
    cell_w = output_size[0] // grid_size[1]
    cell_h = output_size[1] // grid_size[0]
    
    resized_frames = []
    for frame in frames:
        # Пропорциональное изменение размера с обрезкой или отступами
        h, w = frame.shape[:2]
        scale = min(cell_w / w, cell_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Центрируем в ячейке
        canvas = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        y_offset = (cell_h - new_h) // 2
        x_offset = (cell_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        resized_frames.append(canvas)

    # Добавляем пустые кадры, если камер меньше, чем ячеек
    while len(resized_frames) < grid_size[0] * grid_size[1]:
        resized_frames.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

    # Формируем сетку
    rows = []
    for i in range(0, len(resized_frames), grid_size[1]):
        row = np.hstack(resized_frames[i:i + grid_size[1]])
        rows.append(row)
    
    grid = np.vstack(rows[:grid_size[0]])
    return grid


def get_no_signal_frame(camera_idx, size=TARGET_RESOLUTION):
    """Кадр 'Нет сигнала' в оптимальном разрешении"""
    h, w = size[1], size[0]
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    text = f"No signal cam {camera_idx}"
    font_scale = max(0.5, min(1.0, w / 640))
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.putText(frame, text, ((w - tw) // 2, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
    return frame


def get_waiting_frame(camera_idx, time_left=None, size=TARGET_RESOLUTION):
    """Кадр 'Ожидание движения'"""
    h, w = size[1], size[0]
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(frame, f"CAM {camera_idx}", (w // 2 - 50, h // 2 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, "WAITING FOR MOTION", (w // 2 - 90, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if time_left is not None:
        cv2.putText(frame, f"Next: {time_left}s", (w // 2 - 50, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return frame


class MultiMaskCreator:
    def create_mask(self, camera_index, mask_name="default"):
        if os.environ.get('SSH_CLIENT') or os.environ.get('SSH_TTY'):
            logger.error("Mask creation requires GUI (X11/VNC). Not available over SSH.")
            return None

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error(f"Couldn't open camera {camera_index}")
            return None

        print(f"Creating mask for camera {camera_index}")
        print("Instructions:")
        print("  's' - toggle drawing")
        print("  LMB - add point | RMB - remove last point")
        print("  'c' - clear polygon | 'n' - save polygon")
        print("  'q' - save mask | ESC - quit without saving")

        os.makedirs("masks", exist_ok=True)
        mask_path = f"masks/camera_{camera_index}_{mask_name}.png"

        polygons = []
        current_polygon = []
        drawing = False

        def mouse_callback(event, x, y, flags, param):
            nonlocal current_polygon, drawing
            if event == cv2.EVENT_LBUTTONDOWN and drawing:
                current_polygon.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN and current_polygon:
                current_polygon.pop()

        cv2.namedWindow("Create MultiMask", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Create MultiMask", mouse_callback)

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.critical("Failed to read frame")
                break

            display = frame.copy()
            for poly in polygons:
                pts = np.array(poly, np.int32)
                cv2.polylines(display, [pts], True, (0, 255, 0), 2)
            if len(current_polygon) > 1:
                pts = np.array(current_polygon, np.int32)
                cv2.polylines(display, [pts], False, (0, 255, 255), 2)
            for pt in current_polygon:
                cv2.circle(display, pt, 3, (255, 0, 0), -1)

            cv2.imshow("Create MultiMask", display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('s'):
                drawing = not drawing
            elif key == ord('c'):
                current_polygon = []
            elif key == ord('n'):
                if len(current_polygon) >= 3:
                    polygons.append(current_polygon.copy())
                    current_polygon = []
                    logger.success(f"Polygon added ({len(polygons)} total)")
                else:
                    logger.warning("Need >=3 points")
            elif key == ord('q'):
                if polygons:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    for poly in polygons:
                        pts = np.array(poly, np.int32)
                        cv2.fillPoly(mask, [pts], 255)
                    cv2.imwrite(mask_path, mask)
                    logger.success(f"Mask saved: {mask_path}")
                    break
                else:
                    logger.error("No polygons to save")
            elif key == 27:  # ESC
                logger.warning("Exit without saving")
                mask_path = None
                break

        cap.release()
        cv2.destroyAllWindows()
        return mask_path


def load_mask(mask_path):
    """Загрузка маски"""
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask
    return None


def overlay_mask(frame, mask, color=(0, 255, 0), alpha=0.3):
    """Наложение маски (только если она есть)"""
    if mask is None or mask.size == 0:
        return frame
    color_layer = np.full(frame.shape, color, dtype=np.uint8)
    mask_bool = mask.astype(bool)
    frame[mask_bool] = cv2.addWeighted(frame[mask_bool], 1 - alpha, color_layer[mask_bool], alpha, 0)
    return frame


def draw_bounding_box(frame, rect, label=None, color=(0, 255, 0)):
    """Рисование bounding box (упрощено)"""
    x, y, w, h = rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
    if label:
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def load_lbph_face_recognizer(model_path="face_model.yml", labels_path="labels.npy"):
    """Загрузка модели распознавания лиц"""
    if not (os.path.exists(model_path) and os.path.exists(labels_path)):
        logger.warning("Face model files not found")
        return None, None, None
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    label_dict = np.load(labels_path, allow_pickle=True).item()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return recognizer, label_dict, face_cascade


def detect_faces_only(frame, detection_scale=FACE_DETECTION_SCALE):
    """Детекция лиц БЕЗ распознавания (оптимизировано)"""
    h, w = frame.shape[:2]
    small_w, small_h = int(w / detection_scale), int(h / detection_scale)
    small_frame = cv2.resize(frame, (small_w, small_h))
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )

    face_boxes = []
    for (x, y, fw, fh) in faces:
        x, y = int(x * detection_scale), int(y * detection_scale)
        fw, fh = int(fw * detection_scale), int(fh * detection_scale)
        face_boxes.append([x, y, x + fw, y + fh])
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 0, 255), 1)
        cv2.putText(frame, "NE RASPOZNAN", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    return frame, face_boxes


def detect_and_recognize_faces(recognizer, label_dict, face_cascade, frame, confidence_threshold=80, detection_scale=FACE_DETECTION_SCALE):
    """Распознавание лиц с оптимизацией под Pi"""
    h, w = frame.shape[:2]
    small_w, small_h = int(w / detection_scale), int(h / detection_scale)
    small_frame = cv2.resize(frame, (small_w, small_h))
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )

    face_boxes = []
    for (x, y, fw, fh) in faces:
        x, y = int(x * detection_scale), int(y * detection_scale)
        fw, fh = int(fw * detection_scale), int(fh * detection_scale)
        roi = cv2.cvtColor(frame[y:y+fh, x:x+fw], cv2.COLOR_BGR2GRAY)

        label_id, confidence = recognizer.predict(roi)

        if confidence < confidence_threshold:
            name = label_dict.get(label_id, "Unknown")
            color = (0, 255, 0)
            label_text = f"{name} ({int(confidence)})"
        else:
            color = (0, 0, 255)
            label_text = f"NE RASPOZNAN ({int(confidence)})"

        face_boxes.append([x, y, x + fw, y + fh])
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 1)
        cv2.putText(frame, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return frame, face_boxes
