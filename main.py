import sys
import os
import datetime
import whisper
import srt
from moviepy.editor import VideoFileClip
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QComboBox, 
                            QFileDialog, QProgressBar, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

class TranscriptionWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, file_path, model_size, task, output_format, language):
        super().__init__()
        self.file_path = file_path
        self.model_size = model_size
        self.task = task
        self.output_format = output_format
        self.language = language if language else None

    def run(self):
        try:
            # Check if file is video
            if self.is_video_file(self.file_path):
                self.progress.emit("Extracting audio from video...")
                audio_path = self.extract_audio_from_video(self.file_path)
            else:
                audio_path = self.file_path

            # Load model
            self.progress.emit(f"Loading {self.model_size} model...")
            model = whisper.load_model(self.model_size)

            # Transcribe
            self.progress.emit("Transcribing audio...")
            result = model.transcribe(audio_path, task=self.task, language=self.language)

            # Generate output
            if self.output_format == 'srt':
                output_path = self.generate_srt(result)
            else:
                output_path = self.generate_txt(result)

            self.finished.emit(output_path)

        except Exception as e:
            self.error.emit(str(e))

    def is_video_file(self, file_path):
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        ext = os.path.splitext(file_path)[-1].lower()
        return ext in video_extensions

    def extract_audio_from_video(self, video_path):
        audio_path = video_path.rsplit('.', 1)[0] + '.mp3'
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec='mp3')
        return audio_path

    def generate_srt(self, result):
        subtitles = []
        for segment in result['segments']:
            start = datetime.timedelta(seconds=segment['start'])
            end = datetime.timedelta(seconds=segment['end'])
            text = segment['text']
            subtitle = srt.Subtitle(index=len(subtitles)+1, start=start, end=end, content=text)
            subtitles.append(subtitle)

        srt_content = srt.compose(subtitles)
        srt_file = self.file_path.rsplit('.', 1)[0] + '.srt'
        with open(srt_file, "w", encoding='utf-8') as file:
            file.write(srt_content)
        return srt_file

    def generate_txt(self, result):
        transcription_text = " ".join([segment['text'] for segment in result['segments']])
        text_file = self.file_path.rsplit('.', 1)[0] + '_transcription.txt'
        with open(text_file, "w", encoding='utf-8') as file:
            file.write(transcription_text)
        return text_file

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper Subtitle Generator")
        self.setMinimumSize(600, 400)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_button = QPushButton("Select File")
        self.file_button.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_button)
        layout.addLayout(file_layout)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model Size:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(['tiny', 'base', 'small', 'medium', 'large'])
        self.model_combo.setCurrentText('medium')
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
        # Task selection
        task_layout = QHBoxLayout()
        task_layout.addWidget(QLabel("Task:"))
        self.task_combo = QComboBox()
        self.task_combo.addItems(['transcribe', 'translate'])
        task_layout.addWidget(self.task_combo)
        layout.addLayout(task_layout)
        
        # Output format selection
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Output Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(['srt', 'txt'])
        format_layout.addWidget(self.format_combo)
        layout.addLayout(format_layout)
        
        # Language selection
        language_layout = QHBoxLayout()
        language_layout.addWidget(QLabel("Language:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(['Auto-detect', 'en', 'fr', 'es', 'de', 'it', 'pt', 'nl', 'tr', 'pl', 'ru', 'ja', 'ko', 'zh'])
        language_layout.addWidget(self.language_combo)
        layout.addLayout(language_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.progress_bar)
        
        # Generate button
        self.generate_button = QPushButton("Generate Subtitles")
        self.generate_button.clicked.connect(self.generate_subtitles)
        layout.addWidget(self.generate_button)
        
        self.selected_file = None
        self.worker = None

    def select_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio/Video File",
            "",
            "Media Files (*.mp4 *.avi *.mov *.mkv *.mp3 *.wav *.m4a);;All Files (*.*)"
        )
        if file_name:
            self.selected_file = file_name
            self.file_label.setText(os.path.basename(file_name))

    def generate_subtitles(self):
        if not self.selected_file:
            QMessageBox.warning(self, "Warning", "Please select a file first!")
            return

        self.generate_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Initializing...")

        # Get selected options
        model_size = self.model_combo.currentText()
        task = self.task_combo.currentText()
        output_format = self.format_combo.currentText()
        language = None if self.language_combo.currentText() == 'Auto-detect' else self.language_combo.currentText()

        # Create and start worker thread
        self.worker = TranscriptionWorker(
            self.selected_file,
            model_size,
            task,
            output_format,
            language
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.process_finished)
        self.worker.error.connect(self.process_error)
        self.worker.start()

    def update_progress(self, message):
        self.progress_bar.setFormat(message)

    def process_finished(self, output_path):
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Complete!")
        self.generate_button.setEnabled(True)
        QMessageBox.information(
            self,
            "Success",
            f"Subtitles generated successfully!\nSaved to: {output_path}"
        )

    def process_error(self, error_message):
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Error occurred!")
        self.generate_button.setEnabled(True)
        QMessageBox.critical(
            self,
            "Error",
            f"An error occurred:\n{error_message}"
        )

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 