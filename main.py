#!/usr/bin/env python3

import os
import sys
import subprocess
import tkinter as tk
from tkinter import messagebox, filedialog, ttk

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


class WildfireDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wildfire Detection Suite")
        self.root.geometry("600x500")
        self.root.resizable(True, True)

        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')

        self.setup_ui()

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(main_frame, text="Wildfire Detection Suite",
                                font=('Arial', 18, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Detection Options Frame
        detection_frame = ttk.LabelFrame(
            main_frame, text="Detection Options", padding="10")
        detection_frame.grid(row=1, column=0, columnspan=2,
                             sticky=(tk.W, tk.E), pady=(0, 15))

        # Basic Detection Button
        basic_btn = ttk.Button(detection_frame, text="Start Basic Detection",
                               command=self.start_basic_detection, width=25)
        basic_btn.grid(row=0, column=0, padx=5, pady=5)

        # Advanced Detection Button
        advanced_btn = ttk.Button(detection_frame, text="Start Advanced Detection",
                                  command=self.start_advanced_detection, width=25)
        advanced_btn.grid(row=0, column=1, padx=5, pady=5)

        # Color Fire Detection Button (NEW - Working Method!)
        color_fire_btn = ttk.Button(detection_frame, text="Real-time Fire Detection",
                                    command=self.start_color_fire_detection, width=25)
        color_fire_btn.grid(row=1, column=0, padx=5, pady=5)

        # Batch Test Button
        batch_test_btn = ttk.Button(detection_frame, text="Test Fire Detection",
                                    command=self.run_batch_test, width=25)
        batch_test_btn.grid(row=1, column=1, padx=5, pady=5)

        # Training Options Frame
        training_frame = ttk.LabelFrame(
            main_frame, text="Model Training", padding="10")
        training_frame.grid(row=2, column=0, columnspan=2,
                            sticky=(tk.W, tk.E), pady=(0, 15))

        # Train Model Button
        train_btn = ttk.Button(training_frame, text="Train Custom Model",
                               command=self.start_training, width=25)
        train_btn.grid(row=0, column=0, padx=5, pady=5)

        # GPU Training Button
        gpu_train_btn = ttk.Button(training_frame, text="GPU Training",
                                   command=self.start_gpu_training, width=25)
        gpu_train_btn.grid(row=0, column=1, padx=5, pady=5)

        # Test Model Button
        test_btn = ttk.Button(training_frame, text="Test Model",
                              command=self.test_model, width=25)
        test_btn.grid(row=1, column=0, padx=5, pady=5)

        # Optimized Training Button
        opt_train_btn = ttk.Button(training_frame, text="Optimized Training",
                                   command=self.start_optimized_training, width=25)
        opt_train_btn.grid(row=1, column=1, padx=5, pady=5)

        # Configuration Frame
        config_frame = ttk.LabelFrame(
            main_frame, text="Configuration", padding="10")
        config_frame.grid(row=3, column=0, columnspan=2,
                          sticky=(tk.W, tk.E), pady=(0, 15))

        # Model Selection
        ttk.Label(config_frame, text="Model:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.model_var = tk.StringVar(value="yolov8n.pt")
        model_combo = ttk.Combobox(
            config_frame, textvariable=self.model_var, width=30)
        model_combo['values'] = ('yolov8n.pt', 'yolo11n.pt', 'Custom Model...')
        model_combo.grid(row=0, column=1, padx=5, pady=2)

        # Confidence Threshold
        ttk.Label(config_frame, text="Confidence:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 5))
        self.conf_var = tk.DoubleVar(value=0.35)
        conf_scale = ttk.Scale(config_frame, from_=0.1, to=0.9, variable=self.conf_var,
                               orient=tk.HORIZONTAL, length=200)
        conf_scale.grid(row=1, column=1, padx=5, pady=2)

        # Confidence Label
        self.conf_label = ttk.Label(config_frame, text="0.35")
        self.conf_label.grid(row=1, column=2, padx=(5, 0))
        conf_scale.configure(command=self.update_conf_label)

        # Source Selection
        ttk.Label(config_frame, text="Source:").grid(
            row=2, column=0, sticky=tk.W, padx=(0, 5))
        self.source_var = tk.StringVar(value="0")
        source_frame = ttk.Frame(config_frame)
        source_frame.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Radiobutton(source_frame, text="Webcam", variable=self.source_var,
                        value="0").grid(row=0, column=0, padx=(0, 10))
        ttk.Radiobutton(source_frame, text="File", variable=self.source_var,
                        value="file").grid(row=0, column=1)

        # Info Text
        info_frame = ttk.LabelFrame(
            main_frame, text="Information", padding="10")
        info_frame.grid(row=4, column=0, columnspan=2,
                        sticky=(tk.W, tk.E), pady=(0, 15))

        info_text = tk.Text(info_frame, height=8, width=60, wrap=tk.WORD)
        info_text.grid(row=0, column=0, sticky=(tk.W, tk.E))

        scrollbar = ttk.Scrollbar(
            info_frame, orient=tk.VERTICAL, command=info_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        info_text.configure(yscrollcommand=scrollbar.set)

        info_text.insert(tk.END, """Welcome to Wildfire Detection Suite!

Features:
• Basic Detection: Real-time wildfire detection with alerts
• Advanced Detection: Multi-frame confirmation and tracking
• Model Training: Train custom models on your dataset
• GPU Support: Accelerated training with CUDA
• Custom Models: Use your trained wildfire detection models

Instructions:
1. Select your detection model and confidence threshold
2. Choose camera (0) or video file as source  
3. Click on desired detection or training option
4. Press 'q' to quit detection windows

Note: Make sure your camera is connected for webcam detection.
For file input, you'll be prompted to select a video file.
""")

        info_text.configure(state=tk.DISABLED)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

    def update_conf_label(self, value):
        self.conf_label.config(text=f"{float(value):.2f}")

    def get_source(self):
        if self.source_var.get() == "file":
            file_path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                    ("All files", "*.*")
                ]
            )
            return file_path if file_path else "0"
        return self.source_var.get()

    def get_model_path(self):
        model = self.model_var.get()
        if model == "Custom Model...":
            file_path = filedialog.askopenfilename(
                title="Select Custom Model",
                filetypes=[
                    ("Model files", "*.pt *.onnx"),
                    ("All files", "*.*")
                ]
            )
            return file_path if file_path else "yolov8n.pt"
        return model

    def run_script(self, script_name, args=None):
        try:
            script_path = os.path.join(project_root, "scripts", script_name)
            cmd = [sys.executable, script_path]

            if args:
                cmd.extend(args)

            # Run in a new terminal window
            if sys.platform.startswith('win'):
                subprocess.Popen(['cmd', '/c', 'start', 'cmd', '/k'] + cmd,
                                 cwd=project_root, shell=True)
            else:
                subprocess.Popen(['gnome-terminal', '--', 'bash', '-c',
                                  ' '.join(cmd) + '; read'], cwd=project_root)

        except Exception as e:
            messagebox.showerror(
                "Error", f"Failed to run {script_name}:\n{str(e)}")

    def start_basic_detection(self):
        source = self.get_source()
        if source == "0" or os.path.exists(source):
            model_path = self.get_model_path()
            conf = self.conf_var.get()

            args = [
                "--model", model_path,
                "--source", source,
                "--conf", str(conf)
            ]

            self.run_script("wildfire_detection.py", args)
        else:
            messagebox.showwarning(
                "Invalid Source", "Please select a valid video file or use webcam.")

    def start_advanced_detection(self):
        source = self.get_source()
        if source == "0" or os.path.exists(source):
            model_path = self.get_model_path()
            conf = self.conf_var.get()

            args = [
                "--model", model_path,
                "--source", source,
                "--conf", str(conf)
            ]

            self.run_script("wildfire_advanced.py", args)
        else:
            messagebox.showwarning(
                "Invalid Source", "Please select a valid video file or use webcam.")

    def start_training(self):
        self.run_script("train_wildfire_model.py")

    def start_gpu_training(self):
        self.run_script("train_wildfire_gpu.py")

    def start_optimized_training(self):
        self.run_script("train_wildfire_optimized.py")

    def test_model(self):
        self.run_script("test_wildfire_model.py")

    def start_color_fire_detection(self):
        """Start real-time color-based fire detection (actually works!)"""
        sensitivity = "medium"  # Could add GUI control for this
        args = ["--sensitivity", sensitivity, "--camera", "0"]
        self.run_script("realtime_fire_detection.py", args)

    def run_batch_test(self):
        """Run batch test on fire detection accuracy"""
        self.run_script("test_color_detection.py")


def main():
    # Check if running from executable
    if getattr(sys, 'frozen', False):
        # Running from PyInstaller bundle
        os.chdir(sys._MEIPASS)

    root = tk.Tk()
    app = WildfireDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
