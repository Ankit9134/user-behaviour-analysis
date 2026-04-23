
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

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

class SentimentAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.setup_ui()

        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        # Set window properties
        self.root.title("Sentiment Analysis")
        self.root.geometry(config.WINDOW_SIZE)

        # Set color scheme from config
        self.bg_color = config.BG_COLOR
        self.text_color = config.TEXT_COLOR
        self.accent_color = config.ACCENT_COLOR
        self.positive_color = config.SENTIMENT_COLORS['positive']
        self.negative_color = config.SENTIMENT_COLORS['negative']
        self.neutral_color = config.SENTIMENT_COLORS['neutral']

        self.root.configure(bg=self.bg_color)

        # Create header
        header_frame = tk.Frame(self.root, bg=self.accent_color, height=60)
        header_frame.pack(fill="x")

        title_label = tk.Label(
            header_frame,
            text="Sentiment Analysis",
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

        # Create main content frame
        content_frame = tk.Frame(self.root, bg=self.bg_color)
        content_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Input section
        input_frame = tk.LabelFrame(
            content_frame,
            text="Text Input",
            font=("Helvetica", 12, "bold"),
            bg=self.bg_color,
            fg=self.text_color,
            padx=10,
            pady=10
        )
        input_frame.pack(fill="x", pady=10)

        # Text area with scrollbar
        self.text_area = scrolledtext.ScrolledText(
            input_frame,
            wrap=tk.WORD,
            width=80,
            height=8,
            font=("Helvetica", 11),
            bg=config.CARD_BG,
            fg=self.text_color,
            insertbackground=self.text_color  # Cursor color
        )
        self.text_area.pack(fill="both", expand=True, padx=5, pady=5)

        # Buttons frame
        button_frame = tk.Frame(content_frame, bg=self.bg_color)
        button_frame.pack(fill="x", pady=10)

        # Analyze button
        self.analyze_button = HoverButton(
            button_frame,
            text="Analyze Sentiment",
            font=("Helvetica", 12, "bold"),
            bg=config.PRIMARY_BTN_COLOR,
            fg="white",
            padx=15,
            pady=5,
            command=self.analyze_sentiment
        )
        self.analyze_button.pack(side=tk.LEFT, padx=5)

        # Clear button
        self.clear_button = HoverButton(
            button_frame,
            text="Clear",
            font=("Helvetica", 12),
            bg=config.NEUTRAL_BTN_COLOR,
            fg="white",
            padx=15,
            pady=5,
            command=self.clear_all
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)

        # Return to Main Menu button
        self.return_button = HoverButton(
            button_frame,
            text="Return to Main Menu",
            font=("Helvetica", 12),
            bg=config.DANGER_BTN_COLOR,
            fg="white",
            padx=15,
            pady=5,
            command=self.on_closing
        )
        self.return_button.pack(side=tk.LEFT, padx=5)

        # Results section
        results_frame = tk.LabelFrame(
            content_frame,
            text="Analysis Results",
            font=("Helvetica", 12, "bold"),
            bg=self.bg_color,
            fg=self.text_color,
            padx=10,
            pady=10
        )
        results_frame.pack(fill="both", expand=True, pady=10)

        # Split results into text and visualization
        results_container = tk.Frame(results_frame, bg=self.bg_color)
        results_container.pack(fill="both", expand=True)

        # Text results on the left
        text_results_frame = tk.Frame(results_container, bg=self.bg_color)
        text_results_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=10)

        # Overall sentiment
        overall_frame = tk.Frame(text_results_frame, bg=self.bg_color)
        overall_frame.pack(fill="x", pady=10)

        tk.Label(
            overall_frame,
            text="Overall Sentiment:",
            font=("Helvetica", 12, "bold"),
            bg=self.bg_color,
            fg=self.text_color
        ).pack(side=tk.LEFT)

        self.overall_result = tk.Label(
            overall_frame,
            text="",
            font=("Helvetica", 12, "bold"),
            width=10,
            bg=config.CARD_BG,
            fg=self.text_color
        )
        self.overall_result.pack(side=tk.LEFT, padx=10)

        # Detailed scores
        scores_frame = tk.Frame(text_results_frame, bg=self.bg_color)
        scores_frame.pack(fill="x", pady=5)

        # Create a frame for each sentiment score
        sentiments = [("Positive:", "positive_score"),
                      ("Neutral:", "neutral_score"),
                      ("Negative:", "negative_score")]

        for i, (label_text, attr_name) in enumerate(sentiments):
            frame = tk.Frame(scores_frame, bg=self.bg_color)
            frame.pack(fill="x", pady=5)

            tk.Label(
                frame,
                text=label_text,
                font=("Helvetica", 11),
                width=10,
                anchor="w",
                bg=self.bg_color,
                fg=self.text_color
            ).pack(side=tk.LEFT)

            result_label = tk.Label(
                frame,
                text="",
                font=("Helvetica", 11),
                width=15,
                bg=config.CARD_BG,
                fg=self.text_color
            )
            result_label.pack(side=tk.LEFT, padx=10)

            setattr(self, attr_name, result_label)

        # Visualization on the right
        self.viz_frame = tk.Frame(results_container, bg=self.bg_color)
        self.viz_frame.pack(side=tk.RIGHT, fill="both", expand=True)

        # Create initial empty chart
        self.create_chart()

        # Footer with status information
        footer_frame = tk.Frame(self.root, bg=config.CARD_BG, height=30)
        footer_frame.pack(fill="x", side="bottom")

        status_label = tk.Label(
            footer_frame,
            text="Ready to analyze text sentiment",
            font=("Helvetica", 10),
            bg=config.CARD_BG,
            fg=self.text_color,
            pady=5
        )
        status_label.pack(side=tk.LEFT, padx=10)

        # Center window on screen
        self.center_window()

    def center_window(self):
        """Center the window on the screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def create_chart(self, pos=0, neu=0, neg=0):
        """Create or update the sentiment visualization chart"""
        # Clear previous chart if it exists
        for widget in self.viz_frame.winfo_children():
            widget.destroy()

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)

        # Data
        categories = ['Positive', 'Neutral', 'Negative']
        values = [pos, neu, neg]
        colors = [self.positive_color, self.neutral_color, self.negative_color]

        # Create horizontal bar chart
        bars = ax.barh(categories, values, color=colors)

        # Add percentage labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 1
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                    va='center', fontsize=9)

        # Customize chart
        ax.set_title('Sentiment Distribution', fontsize=12)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Percentage (%)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Set background color
        fig.patch.set_facecolor(self.bg_color)
        ax.set_facecolor(self.bg_color)

        # Set text color
        ax.title.set_color(self.text_color)
        ax.xaxis.label.set_color(self.text_color)
        ax.yaxis.label.set_color(self.text_color)
        ax.tick_params(colors=self.text_color)

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def analyze_sentiment(self):
        """Analyze the sentiment of the input text"""
        # Get text from the text area
        text = self.text_area.get("1.0", "end-1c").strip()

        if not text:
            messagebox.showinfo("Input Required", "Please enter some text to analyze.")
            return

        # Create analyzer and get sentiment scores
        analyzer = SentimentIntensityAnalyzer()
        sentiment_dict = analyzer.polarity_scores(text)

        # Extract scores
        neg = sentiment_dict['neg'] * 100
        neu = sentiment_dict['neu'] * 100
        pos = sentiment_dict['pos'] * 100
        compound = sentiment_dict['compound']

        # Update score labels
        self.positive_score.config(text=f"{pos:.1f}%")
        self.neutral_score.config(text=f"{neu:.1f}%")
        self.negative_score.config(text=f"{neg:.1f}%")

        # Determine overall sentiment
        if compound >= 0.05:
            overall = "Positive"
            self.overall_result.config(text=overall, fg=self.positive_color)
        elif compound <= -0.05:
            overall = "Negative"
            self.overall_result.config(text=overall, fg=self.negative_color)
        else:
            overall = "Neutral"
            self.overall_result.config(text=overall, fg=self.neutral_color)

        # Update chart
        self.create_chart(pos, neu, neg)

    def clear_all(self):
        """Clear all input and results"""
        self.text_area.delete("1.0", tk.END)
        self.overall_result.config(text="")
        self.positive_score.config(text="")
        self.neutral_score.config(text="")
        self.negative_score.config(text="")
        self.create_chart()  # Reset chart

    def on_closing(self, event=None):
        """Handle window closing"""
        self.root.destroy()
        # Exit with a special code to signal return to main menu
        import sys
        sys.exit(config.RETURN_TO_MENU_CODE)

# Driver Code
if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentAnalysisApp(root)
    root.mainloop()

