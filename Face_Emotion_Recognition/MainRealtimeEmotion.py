import os
import warnings
import cv2
import numpy as np
import customtkinter as ctk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import time
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = config.TF_CPP_MIN_LOG_LEVEL
warnings.filterwarnings('ignore', category=UserWarning)

from tensorflow import keras
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class FaceEmotionRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.webcam = None
        self.is_running = False
        self.thread = None
        self.current_frame = None
        self.emotion_counts = {e: 0 for e in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']}
        self.emotion_colors = config.EMOTION_COLORS
        self.labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

        self._build_ui()
        haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(haar_file)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self._center_window()
        self._load_model()

    def _build_ui(self):
        self.root.title("Face Emotion Recognition")
        self.root.geometry(config.WINDOW_SIZE)

        # ── Header ──────────────────────────────────────────────
        header = ctk.CTkFrame(self.root, fg_color=config.ACCENT_COLOR, corner_radius=0, height=64)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header, text="😊  Face Emotion Recognition",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="white"
        ).pack(side="left", padx=24)

        ctk.CTkButton(
            header, text="Return to Main Menu", width=180,
            fg_color=config.DANGER_BTN_COLOR, hover_color="#c0392b",
            command=self.on_closing
        ).pack(side="right", padx=24, pady=12)

        # ── Body ────────────────────────────────────────────────
        body = ctk.CTkFrame(self.root, fg_color=config.BG_COLOR, corner_radius=0)
        body.pack(fill="both", expand=True, padx=20, pady=12)

        # Left: video feed
        left = ctk.CTkFrame(body, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True)

        video_card = ctk.CTkFrame(left, fg_color=config.CARD_BG, corner_radius=10)
        video_card.pack(fill="both", expand=True, pady=(0, 10))

        ctk.CTkLabel(video_card, text="Camera Feed",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=config.TEXT_COLOR).pack(anchor="w", padx=12, pady=(8, 4))

        self.video_canvas = ctk.CTkCanvas(video_card, bg="black", width=640, height=460,
                                          highlightthickness=0)
        self.video_canvas.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Controls
        ctrl = ctk.CTkFrame(left, fg_color="transparent")
        ctrl.pack(fill="x")

        self.start_btn = ctk.CTkButton(
            ctrl, text="▶  Start Camera", width=160,
            fg_color=config.SUCCESS_BTN_COLOR, hover_color="#1e8449",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.toggle_camera
        )
        self.start_btn.pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            ctrl, text="Reset Stats", width=120,
            fg_color=config.NEUTRAL_BTN_COLOR, hover_color="#636e72",
            command=self.reset_stats
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            ctrl, text="Return to Main Menu", width=180,
            fg_color=config.DANGER_BTN_COLOR, hover_color="#c0392b",
            command=self.on_closing
        ).pack(side="left")

        # Right: stats
        right = ctk.CTkFrame(body, fg_color="transparent", width=360)
        right.pack(side="right", fill="both", padx=(16, 0))
        right.pack_propagate(False)

        stats_card = ctk.CTkFrame(right, fg_color=config.CARD_BG, corner_radius=10)
        stats_card.pack(fill="both", expand=True, pady=(0, 10))

        ctk.CTkLabel(stats_card, text="Emotion Statistics",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=config.TEXT_COLOR).pack(anchor="w", padx=12, pady=(8, 4))

        self.fig = Figure(figsize=(3.6, 3.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.set_facecolor(config.BG_COLOR)
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=stats_card)
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._update_pie()

        emotion_card = ctk.CTkFrame(right, fg_color=config.CARD_BG, corner_radius=10)
        emotion_card.pack(fill="x")

        ctk.CTkLabel(emotion_card, text="Current Emotion",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=config.TEXT_COLOR).pack(anchor="w", padx=12, pady=(8, 2))

        self.emotion_label = ctk.CTkLabel(
            emotion_card, text="No face detected",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=config.TEXT_COLOR
        )
        self.emotion_label.pack(pady=(4, 12))

        # ── Status bar ──────────────────────────────────────────
        footer = ctk.CTkFrame(self.root, fg_color=config.CARD_BG, corner_radius=0, height=36)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        self.status_label = ctk.CTkLabel(
            footer, text="Ready. Press 'Start Camera' to begin.",
            font=ctk.CTkFont(size=11), text_color=config.TEXT_COLOR
        )
        self.status_label.pack(side="left", padx=14)

    def _center_window(self):
        self.root.update_idletasks()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        x = (self.root.winfo_screenwidth() - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _load_model(self):
        try:
            self.model = keras.models.load_model(config.FACE_EMOTION_MODEL_H5)
            self.status_label.configure(text="Model loaded successfully.")
        except Exception as e:
            self.status_label.configure(text=f"Error loading model: {e}")
            try:
                with open(config.FACE_EMOTION_MODEL_JSON) as f:
                    self.model = keras.models.model_from_json(f.read())
                self.model.load_weights(config.FACE_EMOTION_MODEL_H5)
                self.status_label.configure(text="Model loaded from JSON.")
            except Exception as e2:
                messagebox.showerror("Error", f"Cannot load model: {e2}")
                self.root.destroy()

    def _extract_features(self, image):
        return np.array(image).reshape(1, 48, 48, 1) / 255.0

    def toggle_camera(self):
        if self.is_running:
            self.is_running = False
            self.start_btn.configure(text="▶  Start Camera", fg_color=config.SUCCESS_BTN_COLOR)
            self.status_label.configure(text="Camera stopped.")
            if self.webcam:
                self.webcam.release()
                self.webcam = None
        else:
            self.webcam = cv2.VideoCapture(config.DEFAULT_CAMERA_INDEX)
            if not self.webcam.isOpened():
                messagebox.showerror("Error", "Could not open webcam.")
                return
            self.is_running = True
            self.start_btn.configure(text="⏹  Stop Camera", fg_color=config.DANGER_BTN_COLOR)
            self.status_label.configure(text="Camera started. Detecting emotions…")
            self.thread = threading.Thread(target=self._process_video, daemon=True)
            self.thread.start()

    def _process_video(self):
        while self.is_running:
            ok, frame = self.webcam.read()
            if not ok:
                self.status_label.configure(text="Error reading frame.")
                break
            self._process_frame(frame)
            time.sleep(0.01)

    def _process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        current = "No face detected"

        try:
            for (x, y, w, h) in faces:
                roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
                pred = self.model.predict(self._extract_features(roi))
                label = self.labels[np.argmax(pred)]
                self.emotion_counts[label] += 1
                current = label

                hex_c = self.emotion_colors.get(label, "#FFFFFF")
                r, g, b = (int(hex_c[i:i+2], 16) for i in (1, 3, 5))
                cv2.rectangle(rgb, (x, y), (x+w, y+h), (r, g, b), 2)
                cv2.putText(rgb, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (r, g, b), 2)

            color = self.emotion_colors.get(current, config.TEXT_COLOR)
            self.emotion_label.configure(text=current.capitalize(), text_color=color)

            total = sum(self.emotion_counts.values())
            if total % 10 == 0 and total > 0:
                self._update_pie()

        except Exception as e:
            print(f"Prediction error: {e}")

        self.current_frame = rgb.copy()
        cw = self.video_canvas.winfo_width()
        ch = self.video_canvas.winfo_height()

        if cw > 1 and ch > 1:
            ih, iw = rgb.shape[:2]
            scale = min(cw / iw, ch / ih)
            rgb = cv2.resize(rgb, (int(iw * scale), int(ih * scale)))

        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_canvas.create_image(cw // 2, ch // 2, image=imgtk, anchor="center")
        self.video_canvas.image = imgtk

    def _update_pie(self):
        self.ax.clear()
        total = sum(self.emotion_counts.values())
        if total > 0:
            data = {e: c for e, c in self.emotion_counts.items() if c > 0}
            self.ax.pie(
                data.values(),
                labels=[f"{e} ({c})" for e, c in data.items()],
                colors=[self.emotion_colors[e] for e in data],
                autopct='%1.1f%%', startangle=90, textprops={'color': config.TEXT_COLOR}
            )
            self.ax.axis('equal')
        else:
            self.ax.text(0.5, 0.5, 'No data yet', ha='center', va='center',
                         transform=self.ax.transAxes, color=config.TEXT_COLOR)
            self.ax.axis('off')

        self.fig.set_facecolor(config.BG_COLOR)
        self.chart_canvas.draw()

    def reset_stats(self):
        self.emotion_counts = {e: 0 for e in self.emotion_counts}
        self._update_pie()
        self.status_label.configure(text="Statistics reset.")

    def on_closing(self):
        self.is_running = False
        if self.webcam:
            self.webcam.release()
        self.root.destroy()
        sys.exit(config.RETURN_TO_MENU_CODE)


if __name__ == "__main__":
    root = ctk.CTk()
    app = FaceEmotionRecognitionApp(root)
    root.mainloop()
