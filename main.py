import os
import sys
import subprocess
import customtkinter as ctk
from tkinter import messagebox
import config

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def run_face_emotion_recognition():
    original_dir = os.getcwd()
    try:
        os.chdir(config.FACE_EMOTION_PATH)
        result = subprocess.run([sys.executable, "MainRealtimeEmotion.py"])
        os.chdir(original_dir)
        if result.returncode == config.RETURN_TO_MENU_CODE:
            start_main_menu()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run Face Emotion Recognition: {str(e)}")
        os.chdir(original_dir)
        start_main_menu()


def run_sentiment_analysis():
    original_dir = os.getcwd()
    try:
        os.chdir(config.SENTIMENT_ANALYSIS_PATH)
        result = subprocess.run([sys.executable, "analysis.py"])
        os.chdir(original_dir)
        if result.returncode == config.RETURN_TO_MENU_CODE:
            start_main_menu()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run Sentiment Analysis: {str(e)}")
        os.chdir(original_dir)
        start_main_menu()


def start_main_menu():
    root = ctk.CTk()
    app = ProjectSelector(root)
    root.mainloop()


class ProjectSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Project Hub")
        self.root.geometry(config.WINDOW_SIZE)
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        self._build_ui()
        self._center_window()

    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────
        header = ctk.CTkFrame(self.root, fg_color=config.ACCENT_COLOR, corner_radius=0, height=64)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header, text="🤖  AI Project Hub",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="white"
        ).pack(side="left", padx=24)

        ctk.CTkButton(
            header, text="Exit", width=100,
            fg_color=config.DANGER_BTN_COLOR,
            hover_color="#c0392b",
            command=self.root.destroy
        ).pack(side="right", padx=24, pady=12)

        # ── Body ────────────────────────────────────────────────
        body = ctk.CTkFrame(self.root, fg_color=config.BG_COLOR, corner_radius=0)
        body.pack(fill="both", expand=True, padx=30, pady=20)

        ctk.CTkLabel(
            body, text="Welcome to the AI Project Hub",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=config.TEXT_COLOR
        ).pack(pady=(10, 4))

        ctk.CTkLabel(
            body, text="Select a project to explore AI capabilities:",
            font=ctk.CTkFont(size=13),
            text_color=config.TEXT_COLOR
        ).pack(pady=(0, 20))

        cards_frame = ctk.CTkFrame(body, fg_color="transparent")
        cards_frame.pack(fill="both", expand=True)
        cards_frame.columnconfigure((0, 1), weight=1)

        self._make_card(
            cards_frame, col=0,
            title="😊  Face Emotion Recognition",
            desc="Detect emotions in real-time using\nyour webcam and AI technology.",
            btn_text="Launch",
            btn_color=config.SUCCESS_BTN_COLOR,
            btn_hover="#1e8449",
            command=self._run_face
        )
        self._make_card(
            cards_frame, col=1,
            title="💬  Sentiment Analysis",
            desc="Analyze the sentiment and emotions\nin text using natural language processing.",
            btn_text="Launch",
            btn_color=config.PRIMARY_BTN_COLOR,
            btn_hover="#1a6fa8",
            command=self._run_sentiment
        )

        # ── Footer ──────────────────────────────────────────────
        footer = ctk.CTkFrame(self.root, fg_color=config.CARD_BG, corner_radius=0, height=36)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        ctk.CTkLabel(
            footer, text="Ready to launch a project",
            font=ctk.CTkFont(size=11),
            text_color=config.TEXT_COLOR
        ).pack(side="left", padx=14)

    def _make_card(self, parent, col, title, desc, btn_text, btn_color, btn_hover, command):
        card = ctk.CTkFrame(parent, fg_color=config.CARD_BG, corner_radius=12)
        card.grid(row=0, column=col, padx=14, pady=10, sticky="nsew")

        ctk.CTkLabel(
            card, text=title,
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=config.TEXT_COLOR
        ).pack(pady=(20, 8))

        ctk.CTkLabel(
            card, text=desc,
            font=ctk.CTkFont(size=12),
            text_color=config.TEXT_COLOR,
            justify="center"
        ).pack(pady=(0, 16))

        ctk.CTkButton(
            card, text=btn_text, width=160,
            fg_color=btn_color, hover_color=btn_hover,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=command
        ).pack(pady=(0, 20))

    def _center_window(self):
        self.root.update_idletasks()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        x = (self.root.winfo_screenwidth() - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _run_face(self):
        self.root.destroy()
        run_face_emotion_recognition()

    def _run_sentiment(self):
        self.root.destroy()
        run_sentiment_analysis()


if __name__ == "__main__":
    start_main_menu()
