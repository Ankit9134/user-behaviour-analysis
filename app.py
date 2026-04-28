import os
import sys
import cv2
import numpy as np
import string
from collections import Counter
from flask import Flask, render_template, request, jsonify, Response
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import threading
import warnings
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

# ── NLTK Emotion Model ───────────────────────────────────────────────────────
try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    NLTK_READY = True
except Exception:
    NLTK_READY = False

# Load emotions.txt into dict once at startup
EMOTION_MAP = {}
_emotions_path = os.path.join(config.SENTIMENT_ANALYSIS_PATH, 'emotions.txt')
try:
    with open(_emotions_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Handle format:  'word': 'emotion',  OR  word: emotion
            line = line.strip().strip(',')
            if ':' not in line:
                continue
            # Remove surrounding quotes from both parts
            parts = line.split(':', 1)
            word = parts[0].strip().strip("' \"")
            emo  = parts[1].strip().strip("' \",")
            if word and emo:
                EMOTION_MAP[word] = emo
    print(f"Loaded {len(EMOTION_MAP)} emotion mappings")
except Exception as e:
    print(f"Could not load emotions.txt: {e}")

# Download required NLTK data silently
if NLTK_READY:
    import nltk
    for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


def detect_emotions_nltk(text):
    """Return Counter of emotions found in text using emotions.txt"""
    if not EMOTION_MAP:
        return {}
    try:
        cleaned = text.lower().translate(str.maketrans('', '', string.punctuation))
        # Simple split fallback if NLTK tokenizer fails
        if NLTK_READY:
            try:
                from nltk.corpus import stopwords as sw
                from nltk.stem import WordNetLemmatizer
                from nltk.tokenize import word_tokenize
                tokens = word_tokenize(cleaned)
                stop   = set(sw.words('english'))
                lemma  = WordNetLemmatizer()
                words  = [lemma.lemmatize(w) for w in tokens if w not in stop and w.isalpha()]
            except Exception:
                words = [w for w in cleaned.split() if w.isalpha()]
        else:
            words = [w for w in cleaned.split() if w.isalpha()]

        found = [EMOTION_MAP[w] for w in words if w in EMOTION_MAP]
        return dict(Counter(found))
    except Exception as e:
        print(f"NLTK emotion error: {e}")
        return {}

os.environ['TF_CPP_MIN_LOG_LEVEL'] = config.TF_CPP_MIN_LOG_LEVEL
warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)

# ── Model & Camera Globals ───────────────────────────────────────────────────
webcam        = None
is_running    = False
lock          = threading.Lock()
model         = None
face_cascade  = None
model_status  = "not_loaded"   # not_loaded | loading | ready | error
model_error   = ""

current_emotion     = "No face detected"
current_confidence  = 0.0
emotion_counts      = {e: 0 for e in ['angry','disgust','fear','happy','neutral','sad','surprise']}
LABELS = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'neutral', 5:'sad', 6:'surprise'}

# ── Model Loading ────────────────────────────────────────────────────────────
def load_model_async():
    """Load model in background thread so server starts instantly."""
    global model, face_cascade, model_status, model_error
    model_status = "loading"
    try:
        from tensorflow import keras
        model_path = os.path.join(config.FACE_EMOTION_PATH, config.FACE_EMOTION_MODEL_H5)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = keras.models.load_model(model_path)
        # Warm-up prediction to avoid first-frame lag
        model.predict(np.zeros((1, 48, 48, 1)), verbose=0)
        model_status = "ready"
    except Exception as e:
        model_error  = str(e)
        model_status = "error"

    # Always load face cascade
    haar = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar)


def extract_features(image):
    return np.array(image, dtype=np.float32).reshape(1, 48, 48, 1) / 255.0


# ── Frame Generator ──────────────────────────────────────────────────────────
def generate_frames():
    global current_emotion, current_confidence, emotion_counts

    while is_running:
        with lock:
            if webcam is None:
                break
            ok, frame = webcam.read()

        if not ok:
            time.sleep(0.05)
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Equalise histogram for better detection in varied lighting
        gray  = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(48, 48), flags=cv2.CASCADE_SCALE_IMAGE
        )

        detected = False
        for (x, y, w, h) in faces:
            if model_status != "ready":
                continue
            try:
                roi  = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
                pred = model.predict(extract_features(roi), verbose=0)[0]
                idx  = int(np.argmax(pred))
                conf = float(pred[idx])
                label = LABELS[idx]

                emotion_counts[label] += 1
                current_emotion    = label
                current_confidence = round(conf * 100, 1)
                detected = True

                hex_c = config.EMOTION_COLORS.get(label, '#FFFFFF')
                r, g, b = (int(hex_c[i:i+2], 16) for i in (1, 3, 5))
                bgr = (b, g, r)

                # Rounded rectangle overlay
                cv2.rectangle(frame, (x, y), (x+w, y+h), bgr, 2)
                # Label background pill
                text     = f"{label}  {conf*100:.0f}%"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(frame, (x, y-th-14), (x+tw+10, y), bgr, -1)
                cv2.putText(frame, text, (x+5, y-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
            except Exception as e:
                print(f"Prediction error: {e}")

        if not detected:
            current_emotion    = "No face detected"
            current_confidence = 0.0

        # Overlay model status if not ready
        if model_status == "loading":
            cv2.putText(frame, "Loading model...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,200,0), 2)
        elif model_status == "error":
            cv2.putText(frame, "Model error!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sentiment')
def sentiment_page():
    return render_template('sentiment.html')

@app.route('/emotion')
def emotion_page():
    return render_template('emotion.html')


# Sentiment Analysis API
@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = (data or {}).get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    scores   = SentimentIntensityAnalyzer().polarity_scores(text)
    compound = scores['compound']

    if compound >= 0.05:
        overall = 'Positive'
    elif compound <= -0.05:
        overall = 'Negative'
    else:
        overall = 'Neutral'

    # NLTK emotion detection from emotions.txt
    emotion_counts = detect_emotions_nltk(text)
    top_emotion    = max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'neutral'

    return jsonify({
        'positive':      round(scores['pos'] * 100, 1),
        'neutral':       round(scores['neu'] * 100, 1),
        'negative':      round(scores['neg'] * 100, 1),
        'compound':      round(compound, 4),
        'overall':       overall,
        'emotions':      emotion_counts,
        'top_emotion':   top_emotion
    })


# Model status API
@app.route('/api/model/status')
def model_status_api():
    return jsonify({
        'status': model_status,
        'error':  model_error
    })


@app.route('/api/camera/start', methods=['POST'])
def camera_start():
    global webcam, is_running

    # Detect cloud environment (Render / Railway / etc.)
    if os.environ.get("RENDER") or os.environ.get("PORT"):
        return jsonify({
            "status": "disabled",
            "message": "Camera not available on cloud server"
        })

    if is_running:
        return jsonify({'status': 'already running'})

    cam = cv2.VideoCapture(config.DEFAULT_CAMERA_INDEX)

    if not cam.isOpened():
        return jsonify({'error': 'Cannot open webcam'}), 500

    # Set resolution
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 30)

    with lock:
        webcam = cam

    is_running = True

    return jsonify({
        'status': 'started',
        'model': model_status
    })


@app.route('/api/camera/stop', methods=['POST'])
def camera_stop():
    global webcam, is_running, emotion_counts, current_emotion, current_confidence
    is_running = False
    time.sleep(0.1)
    with lock:
        if webcam:
            webcam.release()
            webcam = None
    emotion_counts     = {e: 0 for e in emotion_counts}
    current_emotion    = 'No face detected'
    current_confidence = 0.0
    return jsonify({'status': 'stopped'})


@app.route('/api/camera/feed')
def camera_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/camera/stats')
def camera_stats():
    total = sum(emotion_counts.values())
    return jsonify({
        'emotion':    current_emotion,
        'confidence': current_confidence,
        'counts':     emotion_counts,
        'total':      total,
        'running':    is_running,
        'model':      model_status
    })

@app.route("/health")
def health():
    return "ok", 200
# ── Startup ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Load model in background so Flask starts immediately
    t = threading.Thread(target=load_model_async, daemon=True)
    t.start()

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )
