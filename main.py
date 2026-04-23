import os
import sys
import subprocess
import tkinter as tk
from tkinter import messagebox
import config  # Import our configuration module

def run_face_emotion_recognition():
    """Run the Face Emotion Recognition project"""
    try:
        # Save the current working directory
        original_dir = os.getcwd()

        # Change to the Face_Emotion_Recognition directory using config
        os.chdir(config.FACE_EMOTION_PATH)

        # Run the script directly using subprocess and capture the return code
        result = subprocess.run([sys.executable, "MainRealtimeEmotion.py"])

        # Return to the original directory
        os.chdir(original_dir)

        # Always restart the main menu when the Face Emotion Recognition app closes
        if result.returncode == config.RETURN_TO_MENU_CODE:
            start_main_menu()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to run Face Emotion Recognition: {str(e)}")
        # Make sure we return to the original directory even if there's an error
        if 'original_dir' in locals():
            os.chdir(original_dir)
        # Restart the main menu even after an error
        start_main_menu()

def run_sentiment_analysis():
    """Run the Sentiment Analysis project"""
    try:
        # Save the current working directory
        original_dir = os.getcwd()

        # Change to the Sentiment_Analysis directory using config
        os.chdir(config.SENTIMENT_ANALYSIS_PATH)

        # Run the script directly using subprocess and capture the return code
        result = subprocess.run([sys.executable, "analysis.py"])

        # Return to the original directory
        os.chdir(original_dir)

        # Always restart the main menu when the Sentiment Analysis app closes
        if result.returncode == config.RETURN_TO_MENU_CODE:
            start_main_menu()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to run Sentiment Analysis: {str(e)}")
        # Make sure we return to the original directory even if there's an error
        if 'original_dir' in locals():
            os.chdir(original_dir)
        # Restart the main menu even after an error
        start_main_menu()

def start_main_menu():
    """Start the main menu GUI"""
    root = tk.Tk()
    app = ProjectSelector(root)
    root.mainloop()

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

class ProjectSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Project Hub")
        self.root.geometry(config.WINDOW_SIZE)

        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

        # Use color scheme from config
        bg_color = config.BG_COLOR
        text_color = config.TEXT_COLOR
        accent_color = config.ACCENT_COLOR

        self.root.configure(bg=bg_color)

        # Create a header frame
        header_frame = tk.Frame(root, bg=accent_color, height=60)
        header_frame.pack(fill="x")

        # App title in header
        title_label = tk.Label(
            header_frame,
            text="AI Project Hub",
            font=("Helvetica", 20, "bold"),
            bg=accent_color,
            fg="white",
            pady=10
        )
        title_label.pack(side=tk.LEFT, padx=20)

        # Add close button to header
        close_button = HoverButton(
            header_frame,
            text="Exit Application",
            font=("Helvetica", 10, "bold"),
            bg=config.DANGER_BTN_COLOR,
            fg="white",
            padx=10,
            pady=5,
            command=self.root.destroy
        )
        close_button.pack(side=tk.RIGHT, padx=20, pady=10)

        # Main content frame
        content_frame = tk.Frame(root, bg=bg_color)
        content_frame.pack(expand=True, fill="both", padx=20, pady=10)

        # Welcome message
        welcome_label = tk.Label(
            content_frame,
            text="Welcome to the AI Project Hub",
            font=("Helvetica", 16, "bold"),
            bg=bg_color,
            fg=text_color,
            pady=10
        )
        welcome_label.pack()

        # Description
        description = tk.Label(
            content_frame,
            text="Select a project to explore AI capabilities:",
            font=("Helvetica", 12),
            bg=bg_color,
            fg=text_color,
            pady=5
        )
        description.pack()

        # Project cards frame using grid layout
        projects_frame = tk.Frame(content_frame, bg=bg_color, pady=20)
        projects_frame.pack(fill="both", expand=True)

        # Configure grid columns
        projects_frame.columnconfigure(0, weight=1)
        projects_frame.columnconfigure(1, weight=1)

        # Face Emotion Recognition card
        face_card = tk.Frame(projects_frame, bg=config.CARD_BG, padx=15, pady=15, relief=tk.RAISED, borderwidth=1)
        face_card.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        face_title = tk.Label(
            face_card,
            text="Face Emotion Recognition",
            font=("Helvetica", 14, "bold"),
            bg="#34495e",
            fg=text_color,
            pady=5
        )
        face_title.pack()

        face_desc = tk.Label(
            face_card,
            text="Detect emotions in real-time using\nyour webcam and AI technology.",
            font=("Helvetica", 10),
            bg="#34495e",
            fg=text_color,
            pady=5,
            justify=tk.CENTER
        )
        face_desc.pack()

        face_button = HoverButton(
            face_card,
            text="Launch",
            font=("Helvetica", 12, "bold"),
            width=15,
            bg=config.SUCCESS_BTN_COLOR,  # Green
            fg="white",
            command=self.run_face_emotion,
            pady=8
        )
        face_button.pack(pady=10)

        # Sentiment Analysis card
        sentiment_card = tk.Frame(projects_frame, bg=config.CARD_BG, padx=15, pady=15, relief=tk.RAISED, borderwidth=1)
        sentiment_card.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        sentiment_title = tk.Label(
            sentiment_card,
            text="Sentiment Analysis",
            font=("Helvetica", 14, "bold"),
            bg="#34495e",
            fg=text_color,
            pady=5
        )
        sentiment_title.pack()

        sentiment_desc = tk.Label(
            sentiment_card,
            text="Analyze the sentiment and emotions\nin text using natural language processing.",
            font=("Helvetica", 10),
            bg="#34495e",
            fg=text_color,
            pady=5,
            justify=tk.CENTER
        )
        sentiment_desc.pack()

        sentiment_button = HoverButton(
            sentiment_card,
            text="Launch",
            font=("Helvetica", 12, "bold"),
            width=15,
            bg=config.PRIMARY_BTN_COLOR,  # Blue
            fg="white",
            command=self.run_sentiment_analysis,
            pady=8
        )
        sentiment_button.pack(pady=10)

        # Footer frame
        footer_frame = tk.Frame(root, bg=config.CARD_BG, height=30)
        footer_frame.pack(fill="x", side="bottom")

        # Status label
        status_label = tk.Label(
            footer_frame,
            text="Ready to launch a project",
            font=("Helvetica", 10),
            bg=config.CARD_BG,
            fg=text_color,
            pady=5
        )
        status_label.pack(side=tk.LEFT, padx=10)

        # Center the window on the screen
        self.center_window()

    def center_window(self):
        """Center the window on the screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def run_face_emotion(self):
        self.root.destroy()  # Close the selector window
        run_face_emotion_recognition()

    def run_sentiment_analysis(self):
        self.root.destroy()  # Close the selector window
        run_sentiment_analysis()

if __name__ == "__main__":
    start_main_menu()
