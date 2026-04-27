from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import customtkinter as ctk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class SentimentAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Analysis")
        self.root.geometry(config.WINDOW_SIZE)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.pos_color = config.SENTIMENT_COLORS['positive']
        self.neu_color = config.SENTIMENT_COLORS['neutral']
        self.neg_color = config.SENTIMENT_COLORS['negative']

        self._build_ui()
        self._center_window()

    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────
        header = ctk.CTkFrame(self.root, fg_color=config.ACCENT_COLOR, corner_radius=0, height=64)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header, text="💬  Sentiment Analysis",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="white"
        ).pack(side="left", padx=24)

        ctk.CTkButton(
            header, text="Return to Main Menu", width=160,
            fg_color=config.DANGER_BTN_COLOR, hover_color="#c0392b",
            command=self.on_closing
        ).pack(side="right", padx=24, pady=12)

        # ── Body ────────────────────────────────────────────────
        body = ctk.CTkFrame(self.root, fg_color=config.BG_COLOR, corner_radius=0)
        body.pack(fill="both", expand=True, padx=24, pady=16)

        # Input section
        input_box = ctk.CTkFrame(body, fg_color=config.CARD_BG, corner_radius=10)
        input_box.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(
            input_box, text="Text Input",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=config.TEXT_COLOR
        ).pack(anchor="w", padx=14, pady=(10, 4))

        self.text_area = ctk.CTkTextbox(
            input_box, height=140,
            font=ctk.CTkFont(size=12),
            fg_color="#2c3e50", text_color=config.TEXT_COLOR,
            border_color=config.ACCENT_COLOR, border_width=1
        )
        self.text_area.pack(fill="x", padx=14, pady=(0, 12))

        # Buttons
        btn_row = ctk.CTkFrame(body, fg_color="transparent")
        btn_row.pack(fill="x", pady=(0, 12))

        ctk.CTkButton(
            btn_row, text="Analyze Sentiment", width=180,
            fg_color=config.PRIMARY_BTN_COLOR, hover_color="#1a6fa8",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.analyze_sentiment
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            btn_row, text="Clear", width=100,
            fg_color=config.NEUTRAL_BTN_COLOR, hover_color="#636e72",
            command=self.clear_all
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            btn_row, text="Return to Main Menu", width=180,
            fg_color=config.DANGER_BTN_COLOR, hover_color="#c0392b",
            command=self.on_closing
        ).pack(side="left")

        # Results section
        results_box = ctk.CTkFrame(body, fg_color=config.CARD_BG, corner_radius=10)
        results_box.pack(fill="both", expand=True)

        ctk.CTkLabel(
            results_box, text="Analysis Results",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=config.TEXT_COLOR
        ).pack(anchor="w", padx=14, pady=(10, 6))

        results_inner = ctk.CTkFrame(results_box, fg_color="transparent")
        results_inner.pack(fill="both", expand=True, padx=14, pady=(0, 12))

        # Left: score labels
        left = ctk.CTkFrame(results_inner, fg_color="transparent")
        left.pack(side="left", fill="y", padx=(0, 20))

        # Overall
        overall_row = ctk.CTkFrame(left, fg_color="transparent")
        overall_row.pack(fill="x", pady=6)
        ctk.CTkLabel(overall_row, text="Overall Sentiment:",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=config.TEXT_COLOR).pack(side="left")
        self.overall_label = ctk.CTkLabel(
            overall_row, text="—",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=config.TEXT_COLOR, width=100
        )
        self.overall_label.pack(side="left", padx=10)

        # Score rows
        self.score_labels = {}
        for name, color in [("Positive", self.pos_color),
                             ("Neutral", self.neu_color),
                             ("Negative", self.neg_color)]:
            row = ctk.CTkFrame(left, fg_color="transparent")
            row.pack(fill="x", pady=4)
            ctk.CTkLabel(row, text=f"{name}:",
                         font=ctk.CTkFont(size=12),
                         text_color=config.TEXT_COLOR, width=80,
                         anchor="w").pack(side="left")
            lbl = ctk.CTkLabel(row, text="—",
                               font=ctk.CTkFont(size=12),
                               text_color=color, width=80)
            lbl.pack(side="left", padx=8)
            self.score_labels[name.lower()] = lbl

        # Right: chart
        self.viz_frame = ctk.CTkFrame(results_inner, fg_color="transparent")
        self.viz_frame.pack(side="right", fill="both", expand=True)
        self._draw_chart()

        # ── Footer ──────────────────────────────────────────────
        footer = ctk.CTkFrame(self.root, fg_color=config.CARD_BG, corner_radius=0, height=36)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        ctk.CTkLabel(footer, text="Ready to analyze text sentiment",
                     font=ctk.CTkFont(size=11),
                     text_color=config.TEXT_COLOR).pack(side="left", padx=14)

    def _draw_chart(self, pos=0, neu=0, neg=0):
        for w in self.viz_frame.winfo_children():
            w.destroy()

        fig, ax = plt.subplots(figsize=(4, 2.8), dpi=100)
        fig.patch.set_facecolor(config.BG_COLOR)
        ax.set_facecolor(config.BG_COLOR)

        bars = ax.barh(
            ['Positive', 'Neutral', 'Negative'],
            [pos, neu, neg],
            color=[self.pos_color, self.neu_color, self.neg_color],
            height=0.5
        )
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 1, bar.get_y() + bar.get_height() / 2,
                    f'{w:.1f}%', va='center', fontsize=9,
                    color=config.TEXT_COLOR)

        ax.set_xlim(0, 110)
        ax.set_title('Sentiment Distribution', color=config.TEXT_COLOR, fontsize=11)
        ax.tick_params(colors=config.TEXT_COLOR)
        ax.xaxis.label.set_color(config.TEXT_COLOR)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#555')

        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    def analyze_sentiment(self):
        text = self.text_area.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showinfo("Input Required", "Please enter some text to analyze.")
            return

        scores = SentimentIntensityAnalyzer().polarity_scores(text)
        pos, neu, neg = scores['pos'] * 100, scores['neu'] * 100, scores['neg'] * 100
        compound = scores['compound']

        self.score_labels['positive'].configure(text=f"{pos:.1f}%", text_color=self.pos_color)
        self.score_labels['neutral'].configure(text=f"{neu:.1f}%", text_color=self.neu_color)
        self.score_labels['negative'].configure(text=f"{neg:.1f}%", text_color=self.neg_color)

        if compound >= 0.05:
            self.overall_label.configure(text="Positive ✅", text_color=self.pos_color)
        elif compound <= -0.05:
            self.overall_label.configure(text="Negative ❌", text_color=self.neg_color)
        else:
            self.overall_label.configure(text="Neutral ➖", text_color=self.neu_color)

        self._draw_chart(pos, neu, neg)

    def clear_all(self):
        self.text_area.delete("1.0", "end")
        self.overall_label.configure(text="—", text_color=config.TEXT_COLOR)
        for lbl in self.score_labels.values():
            lbl.configure(text="—", text_color=config.TEXT_COLOR)
        self._draw_chart()

    def _center_window(self):
        self.root.update_idletasks()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        x = (self.root.winfo_screenwidth() - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def on_closing(self):
        self.root.destroy()
        sys.exit(config.RETURN_TO_MENU_CODE)


if __name__ == "__main__":
    root = ctk.CTk()
    app = SentimentAnalysisApp(root)
    root.mainloop()
