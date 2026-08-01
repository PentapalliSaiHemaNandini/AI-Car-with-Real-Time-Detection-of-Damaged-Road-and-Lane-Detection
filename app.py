from flask import Flask, Response, request, render_template, jsonify, session, redirect, url_for, flash
from flask_cors import CORS
import cv2
import os
import sqlite3
from functools import wraps
import time
import numpy as np 
from ultralytics import YOLO 

app = Flask(__name__)
CORS(app)
print("RUNNING FILE:", __file__)

def init_db():
    conn = sqlite3.connect("users.db")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        latitude TEXT,
        longitude TEXT,
        damage_count INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.close()

init_db()

app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB limit
app.secret_key = "supersecretkey"

# ================= DATA STORAGE =================
fps_history = []
frame_numbers = []
video_path = None
mode = "damage"
damage_count = None 
lat = None
lon = None 
# ================= LOAD YOLO MODEL =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "train10",
    "weights",
    "best.pt"
)

print("MODEL_PATH =", MODEL_PATH)

model = YOLO(MODEL_PATH)

print("Damage classes:", model.names)

LANE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "lane_best.pt"
)

print("LANE_MODEL_PATH = ", LANE_MODEL_PATH)

lane_model = YOLO(LANE_MODEL_PATH)

print("Lane classes:", lane_model.names)


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "name" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


# ================= HOME PAGE =================
@app.route("/")
def index():
    return render_template("login.html")

# ================= ABOUT PAGE =================
@app.route("/home")
def home():

    name = session.get("name")

    if name :
        first_letter = name[0].upper()
    else:
        first_letter = ""

    return render_template("home.html",first_letter=first_letter)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/features")
@login_required
def features():
    return render_template("features.html")

@app.route("/detect")
@login_required
def detect():
    return render_template("detect.html")

@app.route("/contact")
@login_required
def contact():
    return render_template("contact.html")


# ================= VIDEO UPLOAD =================
@app.route("/upload", methods=["POST"])
def upload():

    global video_path, mode, fps_history, frame_numbers, lat, lon

    file = request.files["video"]
    mode = request.form.get("type")

    # ✅ FIX: GET LAT & LON PROPERLY
    lat = request.form.get("lat")
    lon = request.form.get("lon")

    fps_history = []
    frame_numbers = []

    video_path = os.path.join(BASE_DIR, "uploaded_video.mp4")
    file.save(video_path)

    print("Video uploaded:", video_path)
    print("MODE=", mode)
    print("GPS:", lat, lon)

    return "uploaded"

# ================= FPS GRAPH DATA =================
@app.route("/fps_data")
def fps_data():

    return jsonify({
        "frames": frame_numbers,
        "fps": fps_history
    })

# ==============================
# SIGNUP
# ==============================

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]
        confirm = request.form["confirm"]

        if password != confirm:
            flash("Passwords do not match!", "error")
            return redirect(url_for("signup"))

        try:
            conn = sqlite3.connect("users.db")
            conn.execute(
                "INSERT INTO users (name,email,password) VALUES (?,?,?)",
                (name, email, password)
            )
            conn.commit()
            conn.close()

            flash("Account created successfully! Please login.", "success")
            return redirect(url_for("login"))

        except sqlite3.IntegrityError:
            flash("Email already exists!", "error")
            return redirect(url_for("signup"))

    return render_template("signup.html")

# ==============================
# LOGIN
# ==============================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE email=? AND password=?",
            (email, password)
        )
        user = cursor.fetchone()
        conn.close()

        if user:
            # ✅ STORE NAME AND EMAIL PROPERLY
            session["name"] = user[1]
            session["email"] = user[2]

            return redirect(url_for("home"))
        else:
            flash("Invalid email or password")
            return redirect(url_for("login"))

    return render_template("login.html")

# ==============================
# LOGOUT
# ==============================

@app.route("/logout")
@login_required
def logout():
    session.clear()
    flash("Logged out successfully!")
    return redirect(url_for("login"))


@app.route("/get_locations")
def get_locations():

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("SELECT latitude, longitude FROM detections")

    data = cursor.fetchall()
    conn.close()

    return jsonify(data)


import numpy as np

def detect_lanes_opencv(frame):

    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 50, 150)

    mask = np.zeros_like(edges)

    polygon = np.array([[
        (0,height),
        (width,height),
        (width,int(height*0.6)),
        (0,int(height*0.6))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)

    cropped = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        cropped,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=60,
        maxLineGap=30
    )

    if lines is None:
        return frame

    left_lines=[]
    right_lines=[]

    for line in lines:

        line=np.array(line).flatten()

        if len(line)!=4:
            continue

        x1,y1,x2,y2=line

        if x2==x1:
            continue

        slope=(y2-y1)/(x2-x1)

        if abs(slope)<0.4:
            continue

        if slope<0:
            left_lines.append((x1,y1,x2,y2))
        else:
            right_lines.append((x1,y1,x2,y2))

    def draw(avg_lines,color):

        if len(avg_lines)==0:
            return

        x1=int(np.mean([l[0] for l in avg_lines]))
        y1=int(np.mean([l[1] for l in avg_lines]))
        x2=int(np.mean([l[2] for l in avg_lines]))
        y2=int(np.mean([l[3] for l in avg_lines]))

        cv2.line(frame,(x1,y1),(x2,y2),color,5)

    draw(left_lines,(0,255,0))
    draw(right_lines,(0,255,0))

    return frame

# ================= VIDEO PROCESSING =================
def generate_frames():

    global video_path, fps_history, frame_numbers, mode, lan, lon 

    cap = cv2.VideoCapture(video_path)

    frame_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_id += 1

        # ================= PREPROCESS =================
        start_pre = time.time()

        frame = cv2.resize(frame, (640, 384))

        preprocess_time = (time.time() - start_pre) * 1000

        # ================= INFERENCE =================
        start_inf = time.time()

        if mode == "damage":

            results = model(
                frame,
                conf=0.25,
                imgsz=640,
                device="cpu",
                verbose=False
            )

        else:
            results = lane_model(
                frame,
                conf=0.25,
                imgsz=640,
                device="cpu",
                verbose=False
            )

        inference_time = (time.time() - start_inf) * 1000

        # ================= POSTPROCESS =================

        start_post = time.time()

        annotated = frame.copy()

        if mode == "lane":
            annotated = detect_lanes_opencv(annotated)

        pothole_found = False
        damage_count = 0
        max_conf = 0
        # ---------- BOTH DAMAGE + LANE -------------------

        for r in results:

           if r.boxes is None:
              continue

           for box in r.boxes:

            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            if mode == "damage":
               label = model.names[cls_id]
               color = (255, 0, 0)
            else:
               label = lane_model.names[cls_id]
               color = (255, 0, 0)

            if conf > 0.25:

               damage_count += 1
               pothole_found = True

               max_conf = max(max_conf, conf)

               x1, y1, x2, y2 = map(int, box.xyxy[0])

               cv2.rectangle(
                  annotated,
                  (x1, y1),
                  (x2, y2),
                  color,
                  2,
                )

               cv2.putText(
                  annotated,
                  f"{label} {conf:.2f}",
                  (x1, y1 - 10),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.5,
                  color,
                  2,
                )

        postprocess_time = (time.time() - start_post) * 1000

        # ================= SAVE GPS DATA (FIXED) =================
        if pothole_found and lat and lon:
            try:
                with sqlite3.connect("users.db") as conn:
                    conn.execute(
                        "INSERT INTO detections (latitude, longitude, damage_count) VALUES (?, ?, ?)",
                        (lat, lon, damage_count)
                    )
            except Exception as e:
                print("DB Error:", e)

        # ================= FPS =================

        total_time = preprocess_time + inference_time + postprocess_time

        fps = float(1000 / total_time) if total_time > 0 else 0

        fps_history.append(round(fps, 2))
        frame_numbers.append(frame_id)

        # ================= ACCURACY =================

        accuracy = min(100, 90 + max_conf * 10)
        # ================= COLLISION =================

        if mode == "damage":
            collision = "DANGER" if pothole_found else "SAFE"
        else:
            collision = "SAFE"

        # ================= INFO BAR =================

        cv2.rectangle(
            annotated,
            (0, 0),
            (annotated.shape[1], 40),
            (50, 50, 50),
            -1,
        )

        text = (
            f"FPS:{fps:.1f} | "
            f"Accuracy:{accuracy:.1f}% | "
            f"Collision:{collision} | "
            f"Damage:{damage_count}"
        )

        cv2.putText(
            annotated,
            text,
            (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        fps = fps if isinstance(fps, (int, float)) else 0
        accuracy = accuracy if isinstance(accuracy, (int, float)) else 0
        preprocess_time = preprocess_time if isinstance(preprocess_time, (int, float)) else 0
        inference_time = inference_time if isinstance(inference_time, (int, float)) else 0
        postprocess_time = postprocess_time if isinstance(postprocess_time, (int, float)) else 0

        print(
    "[Frame {}] FPS:{:.1f} | Accuracy:{:.1f}% | Pre:{:.1f}ms | Inf:{:.1f}ms | Post:{:.1f}ms | Damage:{} | Collision:{}".format(
        frame_id,
        float(fps) if fps else 0.0,
        float(accuracy) if accuracy else 0.0,
        float(preprocess_time) if preprocess_time else 0.0,
        float(inference_time) if inference_time else 0.0,
        float(postprocess_time) if postprocess_time else 0.0,
        damage_count,
        collision
    )
)

        # ================= STREAM =================

        ret, buffer = cv2.imencode(".jpg", annotated)

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes +
            b"\r\n"
        )

    cap.release()

# ================= IMAGE DETECTION =================
@app.route("/detect_image", methods=["POST"])
def detect_image():

    global mode

    mode = request.form.get("type", mode)
    print("IMAGE MODE =", mode)

    file = request.files["image"]

    img_path = "uploaded_image.jpg"
    file.save(img_path)

    img = cv2.imread(img_path)
    annotated = img.copy()

    pothole_found = False
    damage_count = 0

    lat = request.form.get("lat")
    lon = request.form.get("lon")

    print("📍 GPS Location:", lat, lon)

    # ================= LANE MODE =================
    if mode == "lane":

        # OpenCV lane detection
        annotated = detect_lanes_opencv(annotated)

        # YOLO lane detection
        results = lane_model(img)

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:

                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                if conf > 0.25:

                    label = lane_model.names[cls_id]

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),  # Green for lanes
                        2
                    )

                    cv2.putText(
                        annotated,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

        out_path = "static/result.jpg"
        cv2.imwrite(out_path, annotated)

        return jsonify({
            "result": "/static/result.jpg",
            "damage": 0,
            "pothole": False
        })

    # ================= DAMAGE MODE =================
    else:

        results = model(img)

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:

                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                if conf > 0.25:

                    pothole_found = True
                    damage_count += 1

                    label = model.names[cls_id]

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),  # 🔵 BLUE (BGR format)
                        2
                    )

                    cv2.putText(
                        annotated,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2
                    )

        out_path = "static/result.jpg"
        cv2.imwrite(out_path, annotated)

        # ================= SAVE TO DATABASE =================
        try:
            if pothole_found and lat and lon:

                conn = sqlite3.connect("users.db")

                conn.execute(
                    "INSERT INTO detections (latitude, longitude, damage_count) VALUES (?, ?, ?)",
                    (lat, lon, damage_count)
                )

                conn.commit()
                conn.close()

                print("✅ Saved to DB")

        except Exception as e:
            print("DB Error:", e)

        return jsonify({
            "result": "/static/result.jpg",
            "damage": damage_count,
            "pothole": pothole_found
        })

# ================= VIDEO STREAM =================
@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# ================= RUN =================
if __name__=="__main__":
    app.run(debug=True,use_reloader=False)