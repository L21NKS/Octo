#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import time
import os
import queue
import threading
import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory, Response
from flask_socketio import SocketIO, emit
from werkzeug.security import check_password_hash, generate_password_hash
from functools import wraps
import numpy as np

# === ИМПОРТЫ С ОПТИМИЗИРОВАННЫМ camera_utils ===
from motion_detection import detect_motion, draw_motion_visualization
from camera_utils import (
    initialize_cameras, release_cameras, create_video_grid,
    get_no_signal_frame, get_waiting_frame,
    MultiMaskCreator, load_mask, overlay_mask, load_lbph_face_recognizer,
    detect_and_recognize_faces, detect_faces_only,
    TARGET_FPS, TARGET_RESOLUTION  # ключевые константы!
)
from logger import motion_logger
from loguru import logger
from AI_face import learning
from config import CAMERA_INDICES, SYSTEM

# === КОНСТАНТЫ ОПТИМИЗАЦИИ ===
POST_MOTION_DURATION = 5  # секунд записи после движения
PRE_RECORD_SECONDS = 4    # секунды презаписи
MAX_FRAME_QUEUE_SIZE = int(TARGET_FPS * 10)  # 10 сек буфера
JPEG_QUALITY = 70         # качество JPEG для потока

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Используем threading режим для стабильности на Pi
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

@app.after_request
def add_header(response):
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

# === БАЗА ПОЛЬЗОВАТЕЛЕЙ ===
USERS = {
    'admin': {'password': generate_password_hash('admin123'), 'role': 'Admin'},
    'user': {'password': generate_password_hash('user123'), 'role': 'User'}
}

# === ГЛОБАЛЬНОЕ СОСТОЯНИЕ ===
system_state = {
    'running': False,
    'system': None,
    'camera_settings': {i: {'faces': False, 'motion': False, 'recording': True, 'triggered': False} for i in CAMERA_INDICES},
    'timeouts': {i: 10 for i in CAMERA_INDICES},
    'motion_sensitivity': {i: 25 for i in CAMERA_INDICES}
}

# Буферы для видеопотоков (маленькие!)
video_buffers = {i: queue.Queue(maxsize=2) for i in CAMERA_INDICES}

# Презапись — динамический размер под TARGET_FPS
pre_record_buffers = {
    i: queue.Queue(maxsize=int(TARGET_FPS * PRE_RECORD_SECONDS))
    for i in CAMERA_INDICES
}

# === ДЕКОРАТОРЫ ===
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session or session.get('role') != 'Admin':
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated_function

# === МАРШРУТЫ ===
@app.route('/')
def index():
    return redirect(url_for('login') if 'user' not in session else url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in USERS and check_password_hash(USERS[username]['password'], password):
            session['user'] = username
            session['role'] = USERS[username]['role']
            return jsonify({'success': True, 'role': USERS[username]['role']})
        return jsonify({'success': False, 'error': 'Неверный логин или пароль'}), 401
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', role=session.get('role'), camera_indices=CAMERA_INDICES)

# === API: УПРАВЛЕНИЕ СИСТЕМОЙ ===
@app.route('/api/system/start', methods=['POST'])
@admin_required
def start_system():
    if system_state['running']:
        return jsonify({'error': 'Система уже запущена'}), 400

    try:
        from octo_cli import SurveillanceSystem
        system = SurveillanceSystem()
        
        # Применяем настройки
        system.camera_faces = [i for i in CAMERA_INDICES if system_state['camera_settings'][i]['faces']]
        system.camera_motion = [i for i in CAMERA_INDICES if system_state['camera_settings'][i]['motion']]
        system.camera_recording = CAMERA_INDICES.copy()
        system.camera_triggered = [i for i in CAMERA_INDICES if system_state['camera_settings'][i]['triggered']]
        system.MOTION_TIMEOUTS = system_state['timeouts'].copy()
        system.MOTION_THRESHOLDS = system_state['motion_sensitivity'].copy()
        system.FPS = TARGET_FPS  # важно!

        logger.info(f"Настройки: faces={system.camera_faces}, motion={system.camera_motion}")

        system.initialize(skip_settings=True)
        system_state['system'] = system
        system_state['running'] = True

        threading.Thread(target=process_cameras_loop, daemon=True).start()
        motion_logger.log_system_event("Система запущена через веб")
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Ошибка запуска: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/stop', methods=['POST'])
@admin_required
def stop_system():
    if not system_state['running']:
        return jsonify({'error': 'Система не запущена'}), 400
    try:
        if system_state['system']:
            system_state['system'].cleanup()
            system_state['system'] = None
        system_state['running'] = False
        motion_logger.log_system_event("Система остановлена")
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Ошибка остановки: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/status', methods=['GET'])
@login_required
def system_status():
    return jsonify({
        'running': system_state['running'],
        'settings': system_state['camera_settings']
    })

# === НАСТРОЙКИ КАМЕР ===
@app.route('/api/settings/cameras', methods=['GET', 'POST'])
@admin_required
def camera_settings():
    if request.method == 'GET':
        return jsonify({
            'settings': system_state['camera_settings'],
            'timeouts': system_state['timeouts'],
            'motion_sensitivity': system_state['motion_sensitivity']
        })
    
    data = request.json
    camera_id = data.get('camera_id')
    setting_type = data.get('setting_type')
    value = data.get('value')
    timeout = data.get('timeout')
    
    if camera_id is not None and setting_type in ['faces', 'motion', 'recording', 'triggered']:
        system_state['camera_settings'][camera_id][setting_type] = bool(value)
    elif setting_type == 'timeout' and timeout is not None:
        system_state['timeouts'][camera_id] = int(timeout)
    
    return jsonify({'success': True})

@app.route('/api/settings/sensitivity', methods=['POST'])
@admin_required
def set_sensitivity():
    data = request.json
    camera_id = data.get('camera_id')
    sensitivity = data.get('sensitivity')
    if camera_id is not None and sensitivity is not None:
        sens = max(5, min(100, int(sensitivity)))
        system_state['motion_sensitivity'][camera_id] = sens
        if system_state['running'] and system_state['system']:
            system_state['system'].MOTION_THRESHOLDS[camera_id] = sens
        motion_logger.log_settings({f'camera_{camera_id}_sensitivity': sens})
        return jsonify({'success': True, 'sensitivity': sens})
    return jsonify({'error': 'Нет данных'}), 400

# === МАСКИ И ЛОГИ ===
@app.route('/api/masks/list', methods=['GET'])
@admin_required
def list_masks():
    masks_dir = "masks"
    masks = {}
    if os.path.exists(masks_dir):
        for filename in os.listdir(masks_dir):
            if filename.startswith("camera_") and filename.endswith(".png"):
                try:
                    parts = filename.split('_')
                    camera_idx = int(parts[1])
                    if camera_idx not in masks:
                        masks[camera_idx] = []
                    masks[camera_idx].append({'filename': filename})
                except (ValueError, IndexError):
                    continue
    return jsonify({'masks': masks})

@app.route('/api/masks/delete', methods=['POST'])
@admin_required
def delete_mask():
    data = request.json
    filename = data.get('filename')
    if not filename:
        return jsonify({'error': 'Filename required'}), 400
    mask_path = os.path.join("masks", filename)
    if os.path.exists(mask_path):
        os.remove(mask_path)
        return jsonify({'success': True})
    return jsonify({'error': 'Mask not found'}), 404

@app.route('/api/masks/create', methods=['POST'])
@admin_required
def create_mask():
    data = request.json
    camera_id = data.get('camera_id')
    if camera_id is None:
        return jsonify({'error': 'Camera ID required'}), 400
    # Возвращаем сообщение — создание маски требует GUI
    logger.warning("Создание маски доступно только в терминальной версии")
    return jsonify({
        'success': True,
        'message': 'Используйте терминальную версию для создания маски'
    })

@app.route('/api/logs', methods=['GET'])
@login_required
def get_logs():
    logs_dir = "logs"
    logs = []
    if os.path.exists(logs_dir):
        log_files = sorted([f for f in os.listdir(logs_dir) if f.endswith('.log')], reverse=True)
        for log_file in log_files[:1]:
            log_path = os.path.join(logs_dir, log_file)
            try:
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    logs = [line.strip() for line in f.readlines()[-100:]]
            except Exception as e:
                logger.error(f"Ошибка чтения лога: {e}")
    return jsonify({'logs': logs})

@app.route('/api/biometric/train', methods=['POST'])
@admin_required
def train_model():
    try:
        learning()
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Ошибка обучения: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/biometric/upload', methods=['POST'])
@admin_required
def upload_photos():
    if 'files' not in request.files:
        return jsonify({'error': 'No files selected'}), 400
    files = request.files.getlist('files')
    user_name = request.form.get('user_name', 'unknown')
    if not files:
        return jsonify({'error': 'No files provided'}), 400
    user_dir = os.path.join('dataset', user_name)
    os.makedirs(user_dir, exist_ok=True)
    saved_count = 0
    for file in files:
        if file.filename and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(user_dir, file.filename)
            file.save(filepath)
            saved_count += 1
    return jsonify({'success': True, 'message': f'Загружено {saved_count} фото'})

# === ВИДЕОПОТОК ===
@app.route('/video_feed/<int:camera_id>')
@login_required
def video_feed(camera_id):
    return Response(
        generate_video_stream(camera_id),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def generate_video_stream(camera_id):
    while True:
        try:
            if camera_id in video_buffers:
                frame = video_buffers[camera_id].get(timeout=2)
                # Сжимаем с пониженным качеством
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except queue.Empty:
            # Отправляем кадр "ожидание" в правильном разрешении
            wait_frame = get_waiting_frame(camera_id, size=TARGET_RESOLUTION)
            ret, buffer = cv2.imencode('.jpg', wait_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            logger.error(f"Поток {camera_id} ошибка: {e}")
            time.sleep(0.5)

# === ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ ===
def process_cameras_loop():
    system = system_state['system']
    if not system:
        return

    last_motion_state = {i: False for i in CAMERA_INDICES}
    motion_stop_time = {i: 0 for i in CAMERA_INDICES}

    while system_state['running']:
        try:
            current_time = time.time()

            for idx, cap in enumerate(system.caps):
                if not cap or not cap.isOpened():
                    camera_idx = system.camera_indices[idx]
                    no_signal = get_no_signal_frame(camera_idx, size=TARGET_RESOLUTION)
                    try:
                        video_buffers[camera_idx].put_nowait(no_signal)
                    except queue.Full:
                        pass
                    continue

                ret, raw_frame = cap.read()
                if not ret:
                    camera_idx = system.camera_indices[idx]
                    no_signal = get_no_signal_frame(camera_idx, size=TARGET_RESOLUTION)
                    try:
                        video_buffers[camera_idx].put_nowait(no_signal)
                    except queue.Full:
                        pass
                    continue

                camera_idx = system.camera_indices[idx]

                # === Презапись ===
                if camera_idx in system.camera_recording:
                    try:
                        pre_record_buffers[camera_idx].put_nowait(raw_frame.copy())
                    except queue.Full:
                        try:
                            pre_record_buffers[camera_idx].get_nowait()
                            pre_record_buffers[camera_idx].put_nowait(raw_frame.copy())
                        except queue.Empty:
                            pass

                # === Обработка кадра ===
                processed_frame = system.process_camera_frame(camera_idx, raw_frame, current_time)

                # === Управление записью ===
                if camera_idx in system.camera_recording:
                    motion_now = system.motion_detected.get(camera_idx, False)

                    if motion_now and camera_idx not in system.video_writers:
                        prerecord = []
                        while not pre_record_buffers[camera_idx].empty():
                            prerecord.append(pre_record_buffers[camera_idx].get())
                        if prerecord:
                            start_recording_with_prerecord(system, camera_idx, prerecord)

                    if camera_idx in system.video_writers:
                        if motion_now:
                            motion_stop_time[camera_idx] = current_time + POST_MOTION_DURATION
                            last_motion_state[camera_idx] = True
                        elif last_motion_state[camera_idx]:
                            motion_stop_time[camera_idx] = current_time + POST_MOTION_DURATION
                            last_motion_state[camera_idx] = False

                        if current_time >= motion_stop_time[camera_idx]:
                            system.stop_recording(camera_idx)
                            motion_stop_time[camera_idx] = 0

                        try:
                            system.frame_queues[camera_idx].put_nowait(raw_frame.copy())
                        except (queue.Full, KeyError):
                            pass

                # === Веб-поток ===
                try:
                    video_buffers[camera_idx].put_nowait(processed_frame)
                except queue.Full:
                    try:
                        video_buffers[camera_idx].get_nowait()
                        video_buffers[camera_idx].put_nowait(processed_frame)
                    except queue.Empty:
                        pass

            # Сон в соответствии с TARGET_FPS
            time.sleep(1.0 / TARGET_FPS)

        except Exception as e:
            logger.error(f"Ошибка в цикле: {e}")
            time.sleep(1)

# === ЗАПИСЬ С ПРЕЗАПИСЬЮ ===
def start_recording_with_prerecord(system, camera_idx, pre_record_frames):
    if camera_idx in system.video_writers:
        return

    now = datetime.datetime.now()
    camera_dir = os.path.join("recordings", now.strftime("%Y-%m-%d"), "motion_detected", f"cam{camera_idx}")
    os.makedirs(camera_dir, exist_ok=True)
    filepath = os.path.join(camera_dir, f"recording_{now.strftime('%H-%M-%S')}.avi")

    h, w = pre_record_frames[0].shape[:2]
    # Используем MJPG — быстрее на Pi
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(filepath, fourcc, TARGET_FPS, (w, h))

    for frame in pre_record_frames:
        writer.write(frame)

    frame_queue = queue.Queue(maxsize=MAX_FRAME_QUEUE_SIZE)
    frame_queue.put(pre_record_frames[-1].copy())

    system.video_writers[camera_idx] = writer
    system.recording_start_time[camera_idx] = time.time()
    system.frame_queues[camera_idx] = frame_queue

    thread = threading.Thread(target=system._write_video_thread, args=(camera_idx,), daemon=True)
    thread.start()
    system.recording_threads[camera_idx] = thread

    motion_logger.log_system_event(f"Запись начата: {filepath}")

# === ЗАПУСК ===
def main():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('masks', exist_ok=True)
    os.makedirs('dataset', exist_ok=True)
    os.makedirs('recordings', exist_ok=True)

    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Сервер запущен: http://<IP>:{port}/login")
    
    # Отключаем use_reloader — стабильность на Pi
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True, use_reloader=False)

if __name__ == '__main__':
    main()
