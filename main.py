import os
import subprocess
import configparser
from faster_whisper import WhisperModel
import srt

# 读取配置文件
config = configparser.ConfigParser()
config.read('config.ini')

# 从配置文件中获取输入文件夹和输出文件夹的地址
input_folder = config.get('PATHS', 'input_folder')
output_folder = config.get('PATHS', 'output_folder')

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 初始化 Faster-Whisper 模型
model_size = "large-v2"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

# 遍历输入文件夹内的所有视频文件
for file_name in os.listdir(input_folder):
    if file_name.endswith((".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv")):
        # 构造输入文件名和输出文件名
        input_file = os.path.join(input_folder, file_name)
        output_file = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".mp3")

        # 执行 ffmpeg 命令进行转换，并禁用进度输出
        cmd = ["ffmpeg", "-i", input_file, "-vn", "-acodec", "libmp3lame", "-ab", "192k", "-ac", "2", "-loglevel", "quiet", output_file]
        print('Processing file: ' + input_file)
        print('Output file path: ' + output_file)
        result = subprocess.run(cmd)

        # 检查 ffmpeg 执行结果
        if result.returncode != 0:
            print(f"Conversion of {file_name} failed")
            continue
        print(f"Conversion of {file_name} success")

        # 将 MP3 音频文件转换为字幕文件
        audio_file = output_file
        subtitles_file = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".srt")

        segments, info = model.transcribe(audio_file, beam_size=5)

        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        subtitles = []

        for i, segment in enumerate(segments):
            start_time = int(segment.start * 1000)
            end_time = int(segment.end * 1000)
            text = segment.text.strip()

            if text:
                subtitle = srt.Subtitle(index=i+1, start=srt.millisecond_to_timedelta(start_time), end=srt.millisecond_to_timedelta(end_time), content=text)
                subtitles.append(subtitle)

        with open(subtitles_file, "w", encoding="utf-8") as f:
            f.write(srt.compose(subtitles))

        print(f"Conversion of {file_name} successful")