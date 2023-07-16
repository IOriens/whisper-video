## Video to SRT Subtitle Converter

This Python script converts video files to SRT subtitle files using the Faster-Whisper model for speech recognition. The script reads video files from an input folder, converts them to MP3 format using ffmpeg, transcribes the audio using the Faster-Whisper model, and generates an SRT subtitle file for each video file in the output folder.

### Requirements

To run this script, you will need:

- Python 3.7 or higher
- The dependencies listed in `requirements.txt`
- ffmpeg

To install the required Python libraries, you can use pip:

```
pip install -r requirements.txt
```

To install ffmpeg, you can download a pre-built binary from the official website:

- [FFmpeg Downloads ↗](https://www.ffmpeg.org/download.html)

### Usage

1. Configure the input and output folder paths in the `config.ini` file.
2. Run the script using the following command:

```
python main.py
```

The script will iterate over all video files in the input folder, convert them to MP3 format using ffmpeg, transcribe the audio using the Faster-Whisper model, and generate an SRT subtitle file for each video file in the output folder.

### Configuration

The `config.ini` file contains two sections:

- `PATHS`: This section contains the input and output folder paths.
- `MODEL`: This section contains the model configuration parameters.

By default, the script uses the `large-v2` model size and runs on a CUDA GPU with FP16 compute type. You can change these parameters in the `config.ini` file.

### License

This script is released under the MIT License.