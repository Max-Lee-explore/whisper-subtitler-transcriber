import os
import tempfile
import whisper
import srt
import datetime
import gradio as gr
import yt_dlp
import torch
import re
import shutil
import platform
from pathlib import Path
from typing import Optional, Tuple, Dict
import hashlib

# Constants
MAX_FILE_SIZE_MB = 500  # Maximum file size in MB
SUPPORTED_FORMATS = ['.mp3', '.wav', '.m4a', '.mp4', '.avi', '.mov', '.mkv']
CACHE_DIR = Path(__file__).parent / "cache"  # Changed from home directory to project directory

# Create necessary directories
PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

def get_available_devices() -> Tuple[list, str]:
    """Get available devices based on the system."""
    if torch.backends.mps.is_available():
        # For Apple Silicon Macs
        return ['mps', 'cpu'], 'mps'  # Default to MPS for better performance
    elif torch.cuda.is_available():
        # NVIDIA GPU
        return ['cuda', 'cpu'], 'cuda'
    else:
        # CPU only
        return ['cpu'], 'cpu'

# Get available devices
AVAILABLE_DEVICES, DEFAULT_DEVICE = get_available_devices()

class ModelCache:
    """Singleton class to manage model caching"""
    _instance = None
    _models: Dict[str, whisper.Whisper] = {}
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_model(self, model_size: str, device: str, model_path: Path) -> whisper.Whisper:
        cache_key = f"{model_size}_{device}_{model_path}"
        if cache_key not in self._models:
            self._models[cache_key] = whisper.load_model(
                model_size, 
                device=device, 
                download_root=str(model_path.parent)
            )
        return self._models[cache_key]

def validate_video_url(url: str) -> bool:
    """Validate if the URL is a supported video platform URL."""
    # Basic URL validation
    url_pattern = r'^https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)$'
    if not re.match(url_pattern, url):
        return False
    
    # Try to extract info without downloading to validate URL
    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            ydl.extract_info(url, download=False)
        return True
    except:
        return False

def get_file_hash(file_path: str) -> str:
    """Generate a hash for the file to use as cache key."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def get_model_path(model_size: str, use_project_dir: bool = False) -> Path:
    """Get the path where the model will be stored."""
    if use_project_dir:
        return MODELS_DIR / f"{model_size}.pt"
    else:
        cache_dir = Path.home() / ".cache" / "whisper"
        return cache_dir / f"{model_size}.pt"

def is_model_downloaded(model_size: str, use_project_dir: bool = False) -> bool:
    """Check if the model is already downloaded."""
    return get_model_path(model_size, use_project_dir).exists()

def get_model_size_mb(model_size: str) -> int:
    """Get the approximate size of the model in MB."""
    sizes = {
        'tiny': 75,
        'base': 142,
        'small': 466,
        'medium': 1500,
        'large': 3000
    }
    return sizes.get(model_size, 0)

def download_video_audio(url: str, progress_callback=None) -> Tuple[str, str]:
    """Download audio from video URL and return the file path and title."""
    if not validate_video_url(url):
        raise ValueError("Invalid or unsupported video URL")
    
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio/best',  # Prefer m4a audio, fall back to other audio formats
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
        'progress_hooks': [progress_callback] if progress_callback else None,
        'noplaylist': True,  # Don't download playlists
        'quiet': True,  # Reduce output noise
        'no_warnings': True,  # Reduce output noise
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return os.path.join(tempfile.gettempdir(), f"{info['id']}.mp3"), info.get('title', 'Unknown Title')

def validate_input_file(file_path: str) -> None:
    """Validate input file size and format."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File size exceeds maximum limit of {MAX_FILE_SIZE_MB}MB")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_FORMATS)}")

def process_media(
    input_file: Optional[gr.File],
    video_url: str,
    model_size: str,
    device: str,
    task: str,
    output_format: str,
    prompt: str,
    language: str,
    use_project_dir: bool,
    progress=gr.Progress()
) -> Tuple[str, str, str]:
    """Process media file and generate transcription/subtitles."""
    temp_files = []
    try:
        # Print system information
        print(f"PyTorch version: {torch.__version__}")
        print(f"Selected device: {device}")
        print(f"Model size: {model_size}")
        if device == 'mps':
            print(f"MPS available: {torch.backends.mps.is_available()}")
            print(f"MPS built: {torch.backends.mps.is_built()}")

        # Handle video URL if provided
        if video_url:
            if not video_url.strip():
                return "Please provide a video URL", None, "Error: No video URL provided"
            
            def video_progress(d):
                if d['status'] == 'downloading':
                    progress(0, desc=f"Downloading video audio: {d.get('_percent_str', '0%')}")
            
            progress(0, desc="Starting video download...")
            audio_path, video_title = download_video_audio(video_url, video_progress)
            temp_files.append(audio_path)
        else:
            if input_file is None:
                return "Please provide either a file or video URL", None, "Error: No input file provided"
            audio_path = input_file.name
            validate_input_file(audio_path)

        # Check cache
        file_hash = get_file_hash(audio_path)
        cache_file = CACHE_DIR / f"{file_hash}_{model_size}_{task}_{language}_{output_format}"
        if cache_file.exists():
            progress(0.9, desc="Loading from cache...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                output = f.read()
            return output, str(cache_file), "Success: Loaded from cache!"

        # Check if model needs to be downloaded
        if not is_model_downloaded(model_size, use_project_dir):
            size_mb = get_model_size_mb(model_size)
            model_path = get_model_path(model_size, use_project_dir)
            progress(0.1, desc=f"Downloading {model_size} model ({size_mb}MB)... This may take a few minutes.")
            
            # Load model
            model = ModelCache.get_instance().get_model(
                model_size,
                device,
                get_model_path(model_size, use_project_dir)
            )
            progress(0.3, desc="Model download completed")
        else:
            progress(0.1, desc="Loading model...")
            # Try to load model with selected device, fall back to CPU if MPS fails
            try:
                print(f"Attempting to load model with device: {device}")
                # For tiny model on MPS, we'll try to use it directly
                if model_size == 'tiny' and device == 'mps':
                    model = ModelCache.get_instance().get_model(
                        model_size,
                        device,
                        get_model_path(model_size, use_project_dir)
                    )
                else:
                    # For other models or CPU, use the selected device
                    model = ModelCache.get_instance().get_model(
                        model_size,
                        device,
                        get_model_path(model_size, use_project_dir)
                    )
            except Exception as e:
                print(f"Error loading model with {device}: {str(e)}")
                if device == 'mps':
                    print("Falling back to CPU...")
                    progress(0.1, desc="MPS device failed, falling back to CPU...")
                    model = ModelCache.get_instance().get_model(
                        model_size,
                        'cpu',
                        get_model_path(model_size, use_project_dir)
                    )
                else:
                    raise e

        progress(0.3, desc="Model loaded successfully")

        # Transcribe with prompt
        progress(0.4, desc="Transcribing audio...")
        result = model.transcribe(
            audio_path,
            task=task,
            language=language if language != 'Auto-detect' else None,
            initial_prompt=prompt if prompt else None
        )
        progress(0.8, desc="Transcription completed")

        # Generate output
        progress(0.9, desc="Generating output file...")
        if output_format == 'srt':
            subtitles = []
            for segment in result['segments']:
                start = datetime.timedelta(seconds=segment['start'])
                end = datetime.timedelta(seconds=segment['end'])
                text = segment['text']
                subtitle = srt.Subtitle(index=len(subtitles)+1, start=start, end=end, content=text)
                subtitles.append(subtitle)

            output = srt.compose(subtitles)
            output_file = "output.srt"
        else:
            output = " ".join([segment['text'] for segment in result['segments']])
            output_file = "output.txt"

        # Save to cache
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(output)

        # Save to output file
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(output)

        progress(1.0, desc="Task completed successfully!")
        return output, output_file, "Success: Processing completed successfully!"

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return "", None, error_msg
    
    finally:
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass

def update_device_warning(device: str, model_size: str) -> gr.update:
    """Update warning message based on device and model size selection."""
    if device == 'mps':
        if model_size == 'tiny':
            return gr.update(value="ℹ️ Using Apple Silicon GPU (MPS). This should provide good performance.", visible=True)
        else:
            return gr.update(value="⚠️ MPS (Apple Silicon GPU) selected: Some operations may fall back to CPU for stability. Consider using 'tiny' model for best MPS performance.", visible=True)
    elif device == 'cpu':
        if model_size in ['medium', 'large']:
            return gr.update(value="⚠️ CPU selected: Processing will be slow. Consider using 'tiny' or 'base' model for faster results.", visible=True)
        else:
            return gr.update(value="ℹ️ CPU selected: Processing will be slower than GPU, but 'tiny' and 'base' models should work reasonably well.", visible=True)
    return gr.update(visible=False)

def update_model_info(model_size: str, use_project_dir: bool) -> gr.update:
    """Update model information message."""
    if not is_model_downloaded(model_size, use_project_dir):
        size_mb = get_model_size_mb(model_size)
        model_path = get_model_path(model_size, use_project_dir)
        return gr.update(
            value=f"⚠️ First-time setup: The {model_size} model ({size_mb}MB) will be downloaded to {model_path.parent}. This may take a few minutes depending on your internet connection.",
            visible=True
        )
    return gr.update(visible=False)

def preview_output(output: str, output_format: str) -> str:
    """Generate a preview of the output."""
    if not output:
        return "No output to preview"
    
    if output_format == 'srt':
        # Show first 3 subtitles
        subtitles = output.split('\n\n')[:3]
        return '\n\n'.join(subtitles) + '\n\n...'
    else:
        # Show first 200 characters
        return output[:200] + '...' if len(output) > 200 else output

def cleanup_models(use_project_dir: bool) -> str:
    """Remove downloaded Whisper models."""
    try:
        if use_project_dir:
            # Clean project directory
            for model_file in MODELS_DIR.glob("*.pt"):
                model_file.unlink()
            return f"Models removed from project directory: {MODELS_DIR}"
        else:
            # Clean default cache directory
            cache_dir = Path.home() / ".cache" / "whisper"
            for model_file in cache_dir.glob("*.pt"):
                model_file.unlink()
            return f"Models removed from cache directory: {cache_dir}"
    except Exception as e:
        return f"Error removing models: {str(e)}"

# Custom CSS for fixed dropdown
custom_css = """
.dropdown-menu {
    max-height: 200px !important;
    overflow-y: auto !important;
}
/* Make accordion headers bold only */
.gr-accordion .gr-accordion-header,
.gr-accordion .gr-panel .gr-panel-header,
.gr-accordion .gr-panel-header,
.gr-panel .gr-panel-header,
button[role="button"][aria-expanded] {
    font-weight: 700 !important;
}
/* Center the project title and subtitle */
#header {
    text-align: center !important;
    margin-bottom: 0.5em !important;
}
/* Download Output File area*/
#download-file-box .gr-file {
    min-height: 20px !important;
    height: 20px !important;
    padding: 0.25em 0.5em !important;
}
/* Single white line above footer */
#footer-hr {
    border: none;
    border-top: 2px solid #fff;
    margin: 2em 0 0.5em 0;
    width: 100%;
}
#footer-author {
    text-align: center;
    color: #fff;
    font-size: 1.1em;
    margin-bottom: 0.2em;
}
"""

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("""
    # 🎥 Whisper Subtitle Generator
    <span style='font-size: 1.1em;'>Generate subtitles or transcriptions from your media files or video URLs using OpenAI's Whisper model.<br>Supports various video platforms including YouTube, Vimeo, Dailymotion, and more.</span>
    """, elem_id="header")

    with gr.Row():
        # Left Column: Input & Settings
        with gr.Column(scale=1, min_width=400):
            gr.Markdown("## Input", elem_id="input-header")
            gr.Markdown("**Supported file types:** mp3, wav, m4a, mp4, avi, mov, mkv")
            input_file = gr.File(
                label="Upload Media File",
                file_types=["audio", "video"]
            )
            video_url = gr.Textbox(
                label="Or paste video URL",
                placeholder="https://www.youtube.com/watch?v=... or any supported video platform URL",
                info="Supports YouTube, Vimeo, Dailymotion, Twitch, and many other platforms"
            )
            with gr.Accordion("Advanced Settings", open=False):
                gr.Markdown("### Settings", elem_id="settings-header")
                with gr.Row():
                    model_size = gr.Dropdown(
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        value='tiny' if DEFAULT_DEVICE == 'cpu' else 'medium',
                        label="Model Size",
                        info="Larger models are more accurate but require more memory and processing power"
                    )
                    device = gr.Dropdown(
                        choices=AVAILABLE_DEVICES,
                        value=DEFAULT_DEVICE,
                        label="Device",
                        info="Select the device to use for processing. MPS is for Apple Silicon GPUs."
                    )
                use_project_dir = gr.Checkbox(
                    label="Save models in project folder",
                    value=False,
                    info="If unchecked, models will be saved in the default cache directory (~/.cache/whisper)"
                )
                cleanup_btn = gr.Button("Remove Downloaded Models", variant="secondary")
                cleanup_output = gr.Textbox(label="Cleanup Status", interactive=False)
                device_warning = gr.Markdown(
                    value="",
                    visible=False,
                    elem_classes=["warning-message"]
                )
                model_info = gr.Markdown(
                    value="",
                    visible=False,
                    elem_classes=["info-message"]
                )
                task = gr.Dropdown(
                    choices=['transcribe', 'translate'],
                    value='transcribe',
                    label="Task"
                )
                output_format = gr.Dropdown(
                    choices=['srt', 'txt'],
                    value='srt',
                    label="Output Format"
                )
                language = gr.Dropdown(
                    choices=['Auto-detect', 'en', 'fr', 'es', 'de', 'it', 'pt', 'nl', 'tr', 'pl', 'ru', 'ja', 'ko', 'zh'],
                    value='Auto-detect',
                    label="Language",
                    elem_classes=["dropdown-menu"]
                )
                prompt = gr.Textbox(
                    label="Whisper Prompt",
                    placeholder="Enter a prompt to guide the transcription (optional)",
                    lines=3,
                    info="Use this to guide the model's transcription style or spelling. See documentation for details."
                )
            # Sticky Generate button
            with gr.Row():
                process_btn = gr.Button("Generate", variant="primary", elem_id="generate-btn")
            # Help, Tips & Model Guide (now under Generate button, open by default)
            with gr.Accordion("Help, Tips & Model Guide", open=False):
                gr.Markdown("""
                ## 📝 Tips for Using Prompts
                Prompts can help guide the Whisper model's transcription style. Here are some examples:
                - For consistent spelling: "Names: John Smith, Mary Johnson"
                - For style guidance: "This is a formal business meeting transcript."
                - For technical terms: "Technical terms: API, CPU, GPU, RAM"
                The prompt should be relevant to your content and not exceed 224 tokens.
                ## 💻 Model Size Guide
                - **tiny**: Fastest, least accurate, ~1GB memory
                  - CPU: ~1-2x real-time
                  - GPU: ~10-20x real-time
                - **base**: Good balance of speed and accuracy, ~1GB memory
                  - CPU: ~2-4x real-time
                  - GPU: ~20-40x real-time
                - **small**: Better accuracy, ~2GB memory
                  - CPU: ~4-8x real-time
                  - GPU: ~40-80x real-time
                - **medium**: High accuracy, ~5GB memory
                  - CPU: ~8-16x real-time
                  - GPU: ~80-160x real-time
                - **large**: Best accuracy, ~10GB memory
                  - CPU: ~16-32x real-time
                  - GPU: ~160-320x real-time
                Processing times are approximate and depend on your hardware. For example, a 1-minute audio file might take:
                - 30 seconds to process with tiny model on CPU
                - 3 seconds to process with tiny model on GPU
                - 2 minutes to process with large model on CPU
                - 12 seconds to process with large model on GPU
                Choose a smaller model if you're experiencing memory issues or want faster processing.
                """)

        # Right Column: Output & Help
        with gr.Column(scale=1, min_width=400):
            gr.Markdown("## Output", elem_id="output-header")
            status_message = gr.Markdown(
                value="Ready to process",
                elem_classes=["status-message"]
            )
            preview = gr.Textbox(
                label="Output Preview",
                lines=5,
                interactive=False
            )
            output_text = gr.Textbox(label="Full Output", lines=8)
            output_file = gr.File(label="Download Output File", elem_id="download-file-box")

    # Logic for interactivity
    device.change(
        fn=update_device_warning,
        inputs=[device, model_size],
        outputs=[device_warning]
    )
    model_size.change(
        fn=update_device_warning,
        inputs=[device, model_size],
        outputs=[device_warning]
    )
    model_size.change(
        fn=update_model_info,
        inputs=[model_size, use_project_dir],
        outputs=[model_info]
    )
    use_project_dir.change(
        fn=update_model_info,
        inputs=[model_size, use_project_dir],
        outputs=[model_info]
    )
    cleanup_btn.click(
        fn=cleanup_models,
        inputs=[use_project_dir],
        outputs=[cleanup_output]
    )
    process_btn.click(
        fn=process_media,
        inputs=[input_file, video_url, model_size, device, task, output_format, prompt, language, use_project_dir],
        outputs=[output_text, output_file, status_message]
    ).then(
        fn=preview_output,
        inputs=[output_text, output_format],
        outputs=[preview]
    )

    # Add single white line and author above footer
    gr.Markdown("<div id='footer-author' style='text-align:center;'>Created by Max Lee</div>")
    gr.HTML("<hr id='footer-hr'>")

if __name__ == "__main__":
    demo.launch(allowed_paths=[str(CACHE_DIR), str(MODELS_DIR)]) 