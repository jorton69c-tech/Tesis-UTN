import os, time, datetime, socketserver, signal, sys
from http import server
from threading import Thread, Condition
import collections

import cv2
import smbus2
import pandas as pd
import numpy as np
import RPi.GPIO as GPIO

from picamera2 import Picamera2
from ultralytics import YOLO

# =====================================================
# CONFIGURACIÓN
# =====================================================
PORT = 7123
RESOLUTION = (480, 360)
FPS = 40

MODEL_PATH = "/home/jorge123/tesis_final/yolo11n_ncnn_model"
CONF_TH = 0.5
YOLO_EVERY_N = 3

LIDAR_ADDR = 0x62
ALERTA_UMBRAL = 2.0

LOG_DIR = "/home/jorge123/tesis_final/lidar_logs"
os.makedirs(LOG_DIR, exist_ok=True)

CLASES_YOLO = {
    "person": "Persona",
    "car": "Auto",
    "motorcycle": "Motocicleta",
    "bus": "Bus",
    "truck": "Camión",
    "bicycle": "Bicicleta"
}

# =====================================================
# CONFIGURACIÓN FILTRO LIDAR
# =====================================================
LIDAR_FILTER_WINDOW_SIZE = 7  # Tamaño de ventana para filtro de mediana
LIDAR_MAX_VALID_DISTANCE = 15.0  # Distancia máxima válida (metros)
LIDAR_MIN_VALID_DISTANCE = 0.05  # Distancia mínima válida (metros)
LIDAR_MAX_JUMP = 2.0  # Cambio máximo permitido entre lecturas (metros)

# =====================================================
# CONFIGURACIÓN BUZZER
# =====================================================
BUZZER_PIN = 18  # GPIO18 (pin físico 12)
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.output(BUZZER_PIN, GPIO.LOW)  # Asegurar que empiece apagado

# =====================================================
# ESTADO GLOBAL
# =====================================================
running = True
recording = False
last_lidar_distance = -1.0
buzzer_muted = False

# Variables para compartir distancia entre hilos (con lock)
buzzer_distance = -1.0
buzzer_lock = Condition()

# Buffer para filtro de LiDAR
lidar_buffer = collections.deque(maxlen=LIDAR_FILTER_WINDOW_SIZE)
lidar_buffer_lock = Condition()

frames_jpg = []
registros = []

current_video_path = None
current_excel_path = None

# =====================================================
# FILTRO LIDAR
# =====================================================
def apply_lidar_filter(raw_distance):
    """Aplica filtros a la lectura del LiDAR para eliminar ruido"""
    global lidar_buffer, last_lidar_distance
    
    # 1. Validación de rango
    if (raw_distance < LIDAR_MIN_VALID_DISTANCE or 
        raw_distance > LIDAR_MAX_VALID_DISTANCE):
        return last_lidar_distance if last_lidar_distance > 0 else -1.0
    
    # 2. Detección de saltos bruscos
    if last_lidar_distance > 0:
        distance_jump = abs(raw_distance - last_lidar_distance)
        if distance_jump > LIDAR_MAX_JUMP:
            # Ignorar lectura si es un salto muy brusco
            return last_lidar_distance
    
    # 3. Agregar a buffer para filtro de mediana
    with lidar_buffer_lock:
        lidar_buffer.append(raw_distance)
        
        # Si no hay suficientes muestras, usar el valor actual
        if len(lidar_buffer) < 3:
            filtered_distance = raw_distance
        else:
            # Aplicar filtro de mediana
            sorted_buffer = sorted(lidar_buffer)
            filtered_distance = sorted_buffer[len(sorted_buffer) // 2]
            
            # Opcional: aplicar suavizado adicional
            if len(lidar_buffer) == LIDAR_FILTER_WINDOW_SIZE:
                # Quitar valores extremos y promediar
                trimmed_buffer = sorted_buffer[1:-1]  # Quitar mínimo y máximo
                filtered_distance = np.mean(trimmed_buffer)
    
    return round(filtered_distance, 3)  # Redondear a 3 decimales

# =====================================================
# CONTROL DEL BUZZER (HILO SEPARADO)
# =====================================================
def buzzer_control_thread():
    """Hilo independiente que controla la velocidad del buzzer según la distancia."""
    global running, buzzer_distance, buzzer_muted
    
    while running:
        # Si el buzzer está silenciado, simplemente apagar y esperar
        if buzzer_muted:
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            time.sleep(0.1)
            continue
            
        # Obtener distancia actual (protegida por lock)
        with buzzer_lock:
            dist = buzzer_distance
        
        # Lógica de velocidad según distancia (solo si no está silenciado)
        if dist < 0:  # Error de lectura
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            time.sleep(0.5)
            
        elif dist <= 0.5:  # MUY CERCA (0 - 0.5 metros)
            # Pitido continuo (alerta máxima)
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            time.sleep(0.05)
            
        elif dist <= 1.0:  # CERCA (0.5 - 1.0 metros)
            # Pitidos MUY rápidos
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            time.sleep(0.05)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            time.sleep(0.05)
            
        elif dist <= ALERTA_UMBRAL:  # ALERTA (1.0 - 2.0 metros)
            # Pitidos MODERADOS
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            time.sleep(0.1)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            time.sleep(0.2)
            
        else:  # SEGURO (> 2.0 metros)
            # Silencio
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            time.sleep(0.3)

# =====================================================
# HTML (actualizado con botón de silenciar)
# =====================================================
def get_page():
    global last_lidar_distance, buzzer_muted

    # Determinar estado del buzzer para mostrar
    if buzzer_muted:
        buzzer_status = "🔇 SILENCIADO (manual)"
        buzzer_color = "#888888"
    elif last_lidar_distance < 0:
        buzzer_status = "❓ ERROR LIDAR"
        buzzer_color = "#FFA500"
    elif last_lidar_distance <= 0.5:
        buzzer_status = "🔴 ALERTA MÁXIMA (continuo)"
        buzzer_color = "#FF0000"
    elif last_lidar_distance <= 1.0:
        buzzer_status = "🟠 ALTA (rápido)"
        buzzer_color = "#FF6600"
    elif last_lidar_distance <= ALERTA_UMBRAL:
        buzzer_status = "🟡 MEDIA (moderado)"
        buzzer_color = "#FFFF00"
    else:
        buzzer_status = "🟢 SILENCIO (automático)"
        buzzer_color = "#00FF00"

    # Botón de grabación
    controles_grabacion = """
    <form action="/stop" method="post">
        <button class="btn stop">⏹ Detener grabación</button>
    </form>
    <div class="estado grabando">🔴 GRABANDO</div>
    """ if recording else """
    <form action="/start" method="post">
        <button class="btn start">▶ Iniciar grabación</button>
    </form>
    <div class="estado detenido">⚪ DETENIDO</div>
    """
    
    # Botón de silenciar/activar buzzer
    boton_buzzer = """
    <form action="/mute" method="post">
        <button class="btn mute">🔇 Silenciar Buzzer</button>
    </form>
    """ if not buzzer_muted else """
    <form action="/unmute" method="post">
        <button class="btn unmute">🔊 Activar Buzzer</button>
    </form>
    """

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>UTN - Ingeniería Automotriz</title>

<style>
body {{
 margin:0;
 font-family:Arial;
 background:url("/static/fondo_utn.JPG") no-repeat center center fixed;
 background-size:cover;
 color:white;
}}

.overlay {{
 background:rgba(0,0,0,0.65);
 min-height:100vh;
 padding:15px;
}}

.header {{
 display:flex;
 align-items:center;
 justify-content:center;
 gap:20px;
 border-bottom:3px solid #b30000;
 padding-bottom:10px;
}}

.header img {{ height:70px; }}

.header h1 {{ color:#ff2b2b; margin:0; }}
.header h2 {{ margin:0; font-size:18px; color:#ddd; }}

.stream {{
 display:flex;
 justify-content:center;
 margin-top:15px;
}}

.stream img {{
 width:95vw;
 max-width:1400px;
 border:3px solid #b30000;
}}

.controls {{
 text-align:center;
 margin-top:20px;
}}

.btn {{
 font-size:18px;
 padding:10px 20px;
 border:none;
 border-radius:6px;
 cursor:pointer;
 margin:5px;
}}

.btn.start {{ background:#1fa31f; color:white; }}
.btn.stop {{ background:#b30000; color:white; }}
.btn.mute {{ background:#ff9900; color:white; }}
.btn.unmute {{ background:#0066cc; color:white; }}

.estado {{ margin-top:10px; font-size:18px; }}
.estado.grabando {{ color:red; }}
.estado.detenido {{ color:#ccc; }}

.lidar {{
 margin-top:15px;
 font-size:22px;
 font-weight:bold;
 color:#00ffcc;
}}

.buzzer {{
 margin-top:10px;
 font-size:18px;
 font-weight:bold;
}}

.footer {{
 text-align:center;
 font-size:14px;
 color:#ccc;
 margin-top:15px;
}}

.control-group {{
 background:rgba(0,0,0,0.3);
 border-radius:10px;
 padding:15px;
 margin-top:15px;
}}

.filter-info {{
 font-size:14px;
 color:#aaa;
 margin-top:5px;
}}
</style>
</head>

<body>
<div class="overlay">

 <div class="header">
  <img src="/static/logo_utn.png">
  <div>
   <h1>Universidad Técnica del Norte</h1>
   <h2>Ingeniería Automotriz</h2>
  </div>
  <img src="/static/automotriz.png">
 </div>

 <div class="stream">
  <img src="/stream.mjpg">
 </div>

 <div class="controls">
  <div class="control-group">
   <h3>🎥 Control de Grabación</h3>
   {controles_grabacion}
  </div>
  
  <div class="control-group">
   <h3>🔊 Control de Buzzer</h3>
   {boton_buzzer}
   <div class="buzzer" style="color:{buzzer_color}">
    Estado: {buzzer_status}
   </div>
  </div>
  

 <div class="footer">
  Sistema de detección de obstáculos – Trabajo de Titulación<br>
  Resolución: {RESOLUTION[0]}x{RESOLUTION[1]} · FPS: {FPS}
 </div>

</div>
</body>
</html>
"""

# =====================================================
# STREAM BUFFER
# =====================================================
class StreamingOutput:
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

output = StreamingOutput()

# =====================================================
# GRABACIÓN
# =====================================================
def start_recording():
    global recording, frames_jpg, registros
    global current_video_path, current_excel_path

    if recording:
        return

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_video_path = f"{LOG_DIR}/video_{ts}.mp4"
    current_excel_path = f"{LOG_DIR}/registro_{ts}.xlsx"

    frames_jpg = []
    registros = []
    recording = True
    print("▶ Grabación iniciada")

def stop_recording():
    global recording

    if not recording:
        return

    recording = False
    time.sleep(0.3)

    if frames_jpg:
        first = cv2.imdecode(frames_jpg[0], cv2.IMREAD_COLOR)
        h, w, _ = first.shape

        writer = cv2.VideoWriter(
            current_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (w, h)
        )

        for jpg in frames_jpg:
            writer.write(cv2.imdecode(jpg, cv2.IMREAD_COLOR))

        writer.release()

    if registros:
        pd.DataFrame(registros).to_excel(current_excel_path, index=False)

    print("⏹ Grabación detenida")

# =====================================================
# CONTROL BUZZER (funciones para silenciar/activar)
# =====================================================
def mute_buzzer():
    global buzzer_muted
    buzzer_muted = True
    GPIO.output(BUZZER_PIN, GPIO.LOW)  # Asegurar que se apague inmediatamente
    print("🔇 Buzzer silenciado")

def unmute_buzzer():
    global buzzer_muted
    buzzer_muted = False
    print("🔊 Buzzer activado")

# =====================================================
# SERVIDOR HTTP (actualizado con rutas para buzzer)
# =====================================================
class StreamingHandler(server.BaseHTTPRequestHandler):

    def do_GET(self):

        if self.path in ('/', '/index.html'):
            content = get_page().encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)

        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header(
                'Content-Type',
                'multipart/x-mixed-replace; boundary=FRAME'
            )
            self.end_headers()
            try:
                while running:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame + b'\r\n')
            except:
                pass

        elif self.path.startswith("/static/"):
            try:
                filepath = self.path.lstrip("/")
                with open(filepath, "rb") as f:
                    data = f.read()

                self.send_response(200)
                if filepath.lower().endswith(".png"):
                    self.send_header("Content-Type", "image/png")
                elif filepath.lower().endswith((".jpg", ".jpeg")):
                    self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(data)
            except:
                self.send_error(404)

        else:
            self.send_error(404)

    def do_POST(self):
        if self.headers.get('Content-Length'):
            self.rfile.read(int(self.headers['Content-Length']))

        if self.path == "/start":
            start_recording()
        elif self.path == "/stop":
            stop_recording()
        elif self.path == "/mute":
            mute_buzzer()
        elif self.path == "/unmute":
            unmute_buzzer()

        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    daemon_threads = True

# =====================================================
# LIDAR
# =====================================================
bus = smbus2.SMBus(1)

def read_lidar_raw():
    """Lee el valor crudo del sensor LiDAR"""
    try:
        bus.write_byte_data(LIDAR_ADDR, 0x00, 0x04)
        time.sleep(0.02)
        h = bus.read_byte_data(LIDAR_ADDR, 0x0F)
        l = bus.read_byte_data(LIDAR_ADDR, 0x10)
        raw_distance = ((h << 8) | l) / 100.0
        return raw_distance
    except Exception as e:
        print(f"Error lectura LiDAR: {e}")
        return -1.0

def read_lidar():
    """Lee y filtra la distancia del LiDAR"""
    raw_distance = read_lidar_raw()
    if raw_distance < 0:
        return -1.0
    
    filtered_distance = apply_lidar_filter(raw_distance)
    return filtered_distance

# =====================================================
# LOOP CÁMARA (actualizado para mostrar estado de silencio)
# =====================================================
def camera_loop():
    global running, last_lidar_distance, buzzer_distance, buzzer_muted

    model = YOLO(MODEL_PATH, task="detect")

    picam2 = Picamera2()
    picam2.configure(
        picam2.create_video_configuration(
            main={"size": RESOLUTION, "format": "BGR888"},
            controls={"FrameRate": FPS}
        )
    )
    picam2.start()

    frame_id = 0
    last_results = None
    raw_distance_readings = []  # Para debugging

    while running:
        frame = picam2.capture_array("main")
        now = datetime.datetime.now()

        flecha_izq = flecha_der = False
        objetos = []

        if frame_id % YOLO_EVERY_N == 0:
            last_results = model(frame, verbose=False)

        if last_results:
            for det in last_results[0].boxes:
                if det.conf < CONF_TH:
                    continue
                cls = model.names[int(det.cls)]
                etiqueta = CLASES_YOLO.get(cls, cls)
                x1, y1, x2, y2 = det.xyxy.cpu().numpy().astype(int)[0]
                cx = (x1 + x2) // 2

                if cx < RESOLUTION[0] // 3:
                    flecha_izq = True
                elif cx > 2 * RESOLUTION[0] // 3:
                    flecha_der = True

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame, etiqueta,(x1,y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
                objetos.append(etiqueta)

        # Leer y filtrar distancia LiDAR
        dist = read_lidar()
        
        # Actualizar variables de distancia (global y para buzzer)
        last_lidar_distance = dist
        with buzzer_lock:
            buzzer_distance = dist
        
        # Mostrar distancia en el frame
        cv2.putText(frame, f"LIDAR: {dist:.3f} m",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,
                    (0,0,255) if dist <= ALERTA_UMBRAL else (0,255,0),2)
        
        # Mostrar estado del buzzer en el frame (incluyendo si está silenciado)
        if buzzer_muted:
            buzzer_text = "BUZZER: SILENCIADO (MANUAL)"
            text_color = (128, 128, 128)  # Gris
        elif dist <= 0.5:
            buzzer_text = "BUZZER: CONTINUO"
            text_color = (0, 0, 255)  # Rojo
        elif dist <= 1.0:
            buzzer_text = "BUZZER: RAPIDO"
            text_color = (0, 165, 255)  # Naranja
        elif dist <= ALERTA_UMBRAL:
            buzzer_text = "BUZZER: MODERADO"
            text_color = (0, 255, 255)  # Amarillo
        else:
            buzzer_text = "BUZZER: SILENCIO"
            text_color = (0, 255, 0)  # Verde
            
        cv2.putText(frame, buzzer_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Mostrar información del filtro

        if (frame_id // 5) % 2 == 0:
            if flecha_izq:
                cv2.arrowedLine(frame,(80,180),(20,180),(0,0,255),6,tipLength=0.5)
            if flecha_der:
                cv2.arrowedLine(frame,(400,180),(460,180),(0,0,255),6,tipLength=0.5)

        if recording:
            registros.append({
                "frame": frame_id,
                "hora": now.hour,
                "minuto": now.minute,
                "segundo": now.second,
                "milisegundo": int(now.microsecond / 1000),
                "objetos_detectados": ", ".join(objetos) if objetos else "Nada",
                "distancia_m": dist,
                "buzzer_muted": buzzer_muted
            })

        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])

        if recording:
            frames_jpg.append(jpg)

        output.write(jpg.tobytes())
        frame_id += 1

    try:
        picam2.stop()
    except Exception as e:
        print("Cámara ya detenida:", e)

# =====================================================
# SALIDA LIMPIA
# =====================================================
def shutdown(sig, frame):
    global running
    running = False
    stop_recording()
    time.sleep(0.5)  # Dar tiempo a que los hilos terminen
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    GPIO.cleanup()
    print("\n GPIO limpiados y buzzer apagado.")
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)

# =====================================================
# START
# =====================================================
# Iniciar hilo de la cámara
Thread(target=camera_loop, daemon=True).start()

# Iniciar hilo del buzzer
Thread(target=buzzer_control_thread, daemon=True).start()
print("✅ Buzzer activo en GPIO18. Control por distancia iniciado.")
print(f"🔧 Filtro LiDAR activo: ventana de {LIDAR_FILTER_WINDOW_SIZE} muestras")

# Iniciar servidor
server = StreamingServer(('', PORT), StreamingHandler)
print(f"📡 Streaming activo en http://IP_RASPBERRY:{PORT}")
server.serve_forever()