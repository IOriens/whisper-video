import os
import subprocess
import configparser
from faster_whisper import WhisperModel
import datetime
import srt

# Read the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Get the input and output folder paths from the configuration file
input_folder = config.get('PATHS', 'input_folder')
output_folder = config.get('PATHS', 'output_folder')
model_size = config.get('MODEL', 'model_size')
device = config.get('MODEL', 'device')
compute_type = config.get('MODEL', 'compute_type')
generate_transcript = config.getboolean('OPTIONS', 'generate_transcript')

# Ensure that the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize the Faster-Whisper model
model = WhisperModel(model_size, device=device, compute_type=compute_type)

# Iterate over all video files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith((".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv")):
        # Construct the input and output file paths
        input_file = os.path.join(input_folder, file_name)
        output_file = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".mp3")

        # Check if the output file already exists
        if os.path.exists(output_file):
            print(f"Skipping {file_name} - audio file already exists")
        else:
            # Convert the video file to MP3 format using ffmpeg, and disable progress output
            cmd = ["ffmpeg", "-i", input_file, "-vn", "-acodec", "libmp3lame", "-ab", "192k", "-ac", "2", "-loglevel", "quiet", output_file]
            print(f"Processing file: {input_file}")
            print(f"Output file path: {output_file}")
            result = subprocess.run(cmd)

            # Check the result of the ffmpeg command
            if result.returncode != 0:
                print(f"Conversion of {file_name} failed")
                continue
            print(f"Conversion of {file_name} success")

        # Transcribe the MP3 audio file using the Faster-Whisper model and generate an SRT subtitle file
        audio_file = output_file
        subtitles_file = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".srt")
        transcript_file = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".txt")

        # Check if the subtitles file already exists
        if os.path.exists(subtitles_file):
            print(f"Skipping {file_name} - subtitles file already exists")
        else:
            print(f"Generating file: {subtitles_file}")
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
            with open(subtitles_file, "w", encoding="utf-8") as f:
                f.write(srt.compose(subtitles))

            print(f"Generation of {subtitles_file} successful")

            # Write the transcript file if requested
            if generate_transcript:
                with open(transcript_file, "w", encoding="utf-8") as f:
                    for segment in segments:
                        f.write(segment.text + " ")

                print(f"Generation of {transcript_file} successful")

print("All conversions complete")