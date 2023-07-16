import os
import subprocess
import configparser
from faster_whisper import WhisperModel
import datetime
import srt

def read_config(config_file):
    """
    Reads the configuration file and returns the input and output folder paths, model size,
    device, compute type, and generate_transcript option as a tuple.
    """
    config = configparser.ConfigParser()
    config.read(config_file)

    input_folder = config.get('PATHS', 'input_folder')
    output_folder = config.get('PATHS', 'output_folder')
    model_size = config.get('MODEL', 'model_size')
    device = config.get('MODEL', 'device')
    compute_type = config.get('MODEL', 'compute_type')
    generate_transcript = config.getboolean('OPTIONS', 'generate_transcript')

    return input_folder, output_folder, model_size, device, compute_type, generate_transcript

def convert_to_mp3(video_file, audio_file):
    """
    Converts the input video file to MP3 format using ffmpeg and saves the resulting audio file to
    the specified output file path. Returns True if the conversion is successful, False otherwise.
    """
    if os.path.exists(audio_file):
        print(f"Skipping {video_file} - audio file already exists")
        return True

    cmd = ["ffmpeg", "-i", video_file, "-vn", "-acodec", "libmp3lame", "-ab", "192k", "-ac", "2", "-loglevel", "quiet", audio_file]
    print(f"Processing: {video_file}")
    print(f"Output audio path: {audio_file}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"Conversion of {video_file} failed")
        return False

    print(f"Conversion of {video_file} success")
    return True

def generate_subtitles(audio_file, subtitle_file, transcript_file=None):
    """
    Transcribes the specified audio file using the Faster-Whisper model, generates an SRT subtitle file
    at the specified output file path, and optionally generates a transcript file. Returns True if the
    subtitle generation is successful, False otherwise.
    """
    if os.path.exists(subtitle_file):
        print(f"Skipping {audio_file} - subtitles file already exists")
        return True

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(audio_file, vad_filter=True)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    subtitles = []

    for i, segment in enumerate(segments):
        start_time = datetime.timedelta(milliseconds=segment.start * 1000)
        end_time = datetime.timedelta(milliseconds=segment.end * 1000)
        text = segment.text.strip()

        if text:
            # Create a subtitle object for the segment
            subtitle = srt.Subtitle(index=i+1, start=start_time, end=end_time, content=text)
            subtitles.append(subtitle)

    # Write the subtitles to the SRT file
    with open(subtitle_file, "w", encoding="utf-8") as f:
        f.write(srt.compose(subtitles))

    print(f"Generation of {subtitle_file} successful")

    # Write the transcript file if requested
    if transcript_file:
        with open(transcript_file, "w", encoding="utf-8") as f:
            for segment in segments:
                f.write(segment.text + " ")

        print(f"Generation of {transcript_file} successful")

    return True

if __name__ == "__main__":
    config_file = 'config.ini'
    input_folder, output_folder, model_size, device, compute_type, generate_transcript = read_config(config_file)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith((".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv")):
            video_file = os.path.join(input_folder, file_name)
            audio_file = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".mp3")
            subtitle_file = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".srt")
            transcript_file = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".txt") if generate_transcript else None

            if convert_to_mp3(video_file, audio_file):
                generate_subtitles(audio_file, subtitle_file, transcript_file)

    print("All conversions complete")