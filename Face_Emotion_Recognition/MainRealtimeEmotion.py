import os
import warnings
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import time
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = config.TF_CPP_MIN_LOG_LEVEL
warnings.filterwarnings('ignore', category=UserWarning)

from tensorflow import keras  # Or from keras.models import load_model
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class HoverButton(tk.Button):
    """Button that changes appearance on hover"""
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.default_bg = self["bg"]
        self.default_fg = self["fg"]

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

        # Add 3D effect with border
        self.config(relief=tk.RAISED, borderwidth=2)

    def _on_enter(self, _):
        """Mouse entered the button"""
        r, g, b = self.master.winfo_rgb(self.default_bg)
        hover_bg = f'#{min(int(r/256) + 30, 255):02x}{min(int(g/256) + 30, 255):02x}{min(int(b/256) + 30, 255):02x}'
        self.config(bg=hover_bg, cursor="hand2")

    def _on_leave(self, _):
        """Mouse left the button"""
        self.config(bg=self.default_bg)

class FaceEmotionRecognitionApp:
    def __init__(self, root):
        self.root = root

        # Initialize variables
        self.webcam = None
        self.is_running = False
        self.thread = None
        self.current_frame = None
        self.emotion_counts = {emotion: 0 for emotion in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']}
        self.emotion_colors = config.EMOTION_COLORS

        # Emotion labels
        self.labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

        # Setup the UI
        self.setup_ui()

        # Load face cascade
        haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(haar_file)

        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Center window
        self.center_window()

        # Load the model (do this last as it might take time)
        self.load_model()

    def setup_ui(self):
        # Set window properties
        self.root.title("Face Emotion Recognition")
        self.root.geometry(config.WINDOW_SIZE)

        # Set color scheme from config
        self.bg_color = config.BG_COLOR
        self.text_color = config.TEXT_COLOR
        self.accent_color = config.ACCENT_COLOR

        self.root.configure(bg=self.bg_color)

        # Create header
        header_frame = tk.Frame(self.root, bg=self.accent_color, height=60)
        header_frame.pack(fill="x")

        title_label = tk.Label(
            header_frame,
            text="Face Emotion Recognition",
            font=("Helvetica", 20, "bold"),
            bg=self.accent_color,
            fg="white",
            pady=10
        )
        title_label.pack(side=tk.LEFT, padx=20)

        # Add close button to header
        close_button = HoverButton(
            header_frame,
            text="Return to Main Menu",
            font=("Helvetica", 10, "bold"),
            bg=config.DANGER_BTN_COLOR,
            fg="white",
            padx=10,
            pady=5,
            command=self.on_closing
        )
        close_button.pack(side=tk.RIGHT, padx=20, pady=10)

        # Create main content frame with two columns
        content_frame = tk.Frame(self.root, bg=self.bg_color)
        content_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Left column for video feed
        left_frame = tk.Frame(content_frame, bg=self.bg_color)
        left_frame.pack(side=tk.LEFT, fill="both", expand=True)

        # Video feed frame
        video_frame = tk.LabelFrame(
            left_frame,
            text="Camera Feed",
            font=("Helvetica", 12, "bold"),
            bg=self.bg_color,
            fg=self.text_color,
            padx=10,
            pady=10
        )
        video_frame.pack(fill="both", expand=True, pady=10)

        # Video canvas
        self.video_canvas = tk.Canvas(
            video_frame,
            bg="black",
            width=640,
            height=480
        )
        self.video_canvas.pack(fill="both", expand=True)

        # Control panel
        control_frame = tk.Frame(left_frame, bg=self.bg_color)
        control_frame.pack(fill="x", pady=10)

        # Start button
        self.start_button = HoverButton(
            control_frame,
            text="Start Camera",
            font=("Helvetica", 12, "bold"),
            bg=config.SUCCESS_BTN_COLOR,
            fg="white",
            padx=15,
            pady=5,
            command=self.toggle_camera
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        # Removed snapshot button

        # Reset button
        self.reset_button = HoverButton(
            control_frame,
            text="Reset Stats",
            font=("Helvetica", 12),
            bg=config.NEUTRAL_BTN_COLOR,
            fg="white",
            padx=15,
            pady=5,
            command=self.reset_stats
        )
        self.reset_button.pack(side=tk.LEFT, padx=5)

        # Return to Main Menu button
        self.return_button = HoverButton(
            control_frame,
            text="Return to Main Menu",
            font=("Helvetica", 12),
            bg=config.DANGER_BTN_COLOR,
            fg="white",
            padx=15,
            pady=5,
            command=self.on_closing
        )
        self.return_button.pack(side=tk.LEFT, padx=5)

        # Right column for stats and controls
        right_frame = tk.Frame(content_frame, bg=self.bg_color, width=400)
        right_frame.pack(side=tk.RIGHT, fill="both", padx=(20, 0))

        # Stats frame
        stats_frame = tk.LabelFrame(
            right_frame,
            text="Emotion Statistics",
            font=("Helvetica", 12, "bold"),
            bg=self.bg_color,
            fg=self.text_color,
            padx=10,
            pady=10
        )
        stats_frame.pack(fill="both", expand=True, pady=10)

        # Create matplotlib figure for the pie chart
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=stats_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize empty pie chart
        self.update_pie_chart()

        # Current emotion frame
        current_emotion_frame = tk.LabelFrame(
            right_frame,
            text="Current Emotion",
            font=("Helvetica", 12, "bold"),
            bg=self.bg_color,
            fg=self.text_color,
            padx=10,
            pady=10
        )
        current_emotion_frame.pack(fill="x", pady=10)

        self.current_emotion_label = tk.Label(
            current_emotion_frame,
            text="No face detected",
            font=("Helvetica", 16, "bold"),
            bg=self.bg_color,
            fg=self.text_color,
            pady=10
        )
        self.current_emotion_label.pack()

        # Status bar
        status_frame = tk.Frame(self.root, bg=config.CARD_BG, height=30)
        status_frame.pack(fill="x", side=tk.BOTTOM)

        self.status_label = tk.Label(
            status_frame,
            text="Ready. Press 'Start Camera' to begin.",
            font=("Helvetica", 10),
            bg=config.CARD_BG,
            fg=self.text_color,
            pady=5
        )
        self.status_label.pack(side=tk.LEFT, padx=10)

    def center_window(self):
        """Center the window on the screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def load_model(self):
        """Load the emotion recognition model"""
        try:
            model_path = config.FACE_EMOTION_MODEL_H5
            self.model = keras.models.load_model(model_path)
            self.status_label.config(text="Model loaded successfully from .h5 file.")
        except Exception as e:
            self.status_label.config(text=f"Error loading model from .h5: {e}")
            try:
                json_path = config.FACE_EMOTION_MODEL_JSON
                with open(json_path, "r") as json_file:
                    model_json = json_file.read()
                self.model = keras.models.model_from_json(model_json)
                self.model.load_weights(model_path)
                self.status_label.config(text="Model loaded successfully from JSON and weights.")
            except FileNotFoundError as fe:
                messagebox.showerror("Error", f"Error loading model files: {fe}")
                self.root.destroy()
            except Exception as e2:
                messagebox.showerror("Error", f"Error loading model architecture or weights: {e2}")
                self.root.destroy()

    def extract_features(self, image):
        """Extract features from the image for model prediction"""
        feature = np.array(image)
        feature = feature.reshape(1, 48, 48, 1)
        return feature / 255.0

    def toggle_camera(self):
        """Start or stop the camera feed"""
        if self.is_running:
            self.is_running = False
            self.start_button.config(text="Start Camera", bg=config.SUCCESS_BTN_COLOR)
            self.start_button.default_bg = config.SUCCESS_BTN_COLOR  # Update default_bg for HoverButton
            self.status_label.config(text="Camera stopped.")
            if self.webcam is not None:
                self.webcam.release()
                self.webcam = None
        else:
            self.webcam = cv2.VideoCapture(config.DEFAULT_CAMERA_INDEX)
            if not self.webcam.isOpened():
                messagebox.showerror("Error", "Could not open webcam.")
                return

            self.is_running = True
            self.start_button.config(text="Stop Camera", bg=config.DANGER_BTN_COLOR)
            self.start_button.default_bg = config.DANGER_BTN_COLOR  # Update default_bg for HoverButton
            self.status_label.config(text="Camera started. Detecting emotions...")

            # Start video processing in a separate thread
            self.thread = threading.Thread(target=self.process_video)
            self.thread.daemon = True
            self.thread.start()

    def process_video(self):
        """Process video frames in a separate thread"""
        while self.is_running:
            success, frame = self.webcam.read()
            if not success:
                self.status_label.config(text="Error: Could not read frame from webcam.")
                break

            # Process the frame
            self.process_frame(frame)

            # Small delay to reduce CPU usage
            time.sleep(0.01)

    def process_frame(self, frame):
        """Process a single frame for emotion detection"""
        # Convert to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        # Store current emotion for display
        current_emotion = "No face detected"

        try:
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = gray[y:y + h, x:x + w].copy()

                # Resize for the model
                resized_face = cv2.resize(face_roi, (48, 48))

                # Extract features
                img = self.extract_features(resized_face)

                # Predict emotion
                pred = self.model.predict(img)
                predicted_label_index = np.argmax(pred)
                prediction_label = self.labels[predicted_label_index]

                # Update emotion counts
                self.emotion_counts[prediction_label] += 1

                # Get color for the emotion
                emotion_color = self.emotion_colors.get(prediction_label, "#FFFFFF")

                # Convert hex color to BGR
                r, g, b = tuple(int(emotion_color[i:i+2], 16) for i in (1, 3, 5))
                color_bgr = (b, g, r)

                # Draw rectangle and label
                cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), color_bgr, 2)
                cv2.putText(frame_rgb, prediction_label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_bgr, 2)

                # Update current emotion
                current_emotion = prediction_label

            # Update the current emotion label
            self.current_emotion_label.config(text=current_emotion.capitalize())

            # If an emotion was detected, update its color
            if current_emotion in self.emotion_colors:
                self.current_emotion_label.config(fg=self.emotion_colors[current_emotion])
            else:
                self.current_emotion_label.config(fg=self.text_color)

            # Update the pie chart every 10 frames (to avoid excessive updates)
            if sum(self.emotion_counts.values()) % 10 == 0 and sum(self.emotion_counts.values()) > 0:
                self.update_pie_chart()

        except Exception as e:
            print(f"An error occurred during prediction: {e}")

        # Convert to PhotoImage for display
        h, w = frame_rgb.shape[:2]
        self.current_frame = frame_rgb.copy()  # Store for snapshot

        # Resize if needed to fit the canvas
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:  # Ensure canvas has been drawn
            # Calculate aspect ratio
            img_ratio = w / h
            canvas_ratio = canvas_width / canvas_height

            if img_ratio > canvas_ratio:
                # Image is wider than canvas
                new_width = canvas_width
                new_height = int(canvas_width / img_ratio)
            else:
                # Image is taller than canvas
                new_height = canvas_height
                new_width = int(canvas_height * img_ratio)

            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))

        # Convert to PhotoImage
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the canvas
        self.video_canvas.create_image(
            canvas_width // 2, canvas_height // 2,
            image=imgtk, anchor=tk.CENTER
        )
        self.video_canvas.image = imgtk  # Keep a reference

    def update_pie_chart(self):
        """Update the emotion statistics pie chart"""
        # Clear the previous chart
        self.ax.clear()

        # Get data for the pie chart
        labels = []
        sizes = []
        colors = []

        total = sum(self.emotion_counts.values())

        if total > 0:
            for emotion, count in self.emotion_counts.items():
                if count > 0:
                    labels.append(f"{emotion} ({count})")
                    sizes.append(count)
                    colors.append(self.emotion_colors[emotion])

            # Create pie chart
            self.ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                       shadow=True, startangle=90)
            self.ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        else:
            # If no data, show empty chart with message
            self.ax.text(0.5, 0.5, 'No data yet', horizontalalignment='center',
                        verticalalignment='center', transform=self.ax.transAxes)
            self.ax.axis('off')

        # Update the canvas
        self.fig.set_facecolor(self.bg_color)
        self.canvas.draw()

    # Take snapshot method removed

    def reset_stats(self):
        """Reset the emotion statistics"""
        self.emotion_counts = {emotion: 0 for emotion in self.emotion_counts}
        self.update_pie_chart()
        self.status_label.config(text="Statistics reset.")

    def on_closing(self, event=None):
        """Handle window closing"""
        self.is_running = False
        if self.webcam is not None:
            self.webcam.release()
        self.root.destroy()
        # Exit with a special code to signal return to main menu
        import sys
        sys.exit(config.RETURN_TO_MENU_CODE)

# Driver code
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceEmotionRecognitionApp(root)
    root.mainloop()