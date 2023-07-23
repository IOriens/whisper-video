import os
import subprocess
import configparser
from faster_whisper import WhisperModel
import datetime
import srt
import shutil
import logging
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate


def read_config(config_file):
    """
    Reads the configuration file and returns the input and output folder paths, model size,
    device, compute type, and generate_transcript option as a tuple.
    """
    config = configparser.ConfigParser()
    config.read(config_file)

    input_folder = config.get("PATHS", "input_folder")
    output_folder = config.get("PATHS", "output_folder")
    model_size = config.get("MODEL", "model_size")
    device = config.get("MODEL", "device")
    compute_type = config.get("MODEL", "compute_type")
    task = config.get("MODEL", "task")
    generate_transcript = config.getboolean("OPTIONS", "generate_transcript")
    generate_summary = config.getboolean("OPTIONS", "generate_summary")
    summary_language = config.get("OPTIONS", "summary_language")
    openai_api_key = config.get("OPENAI", "API_KEY")
    openai_api_base = config.get("OPENAI", "API_BASE")
    openai_model = config.get("OPENAI", "model")
    text_chunk_size = config.getint("OPTIONS", "text_chunk_size")
    max_chunk = config.getint("OPTIONS", "max_chunk")

    return (
        input_folder,
        output_folder,
        model_size,
        device,
        compute_type,
        task,
        generate_transcript,
        generate_summary,
        summary_language,
        openai_api_key,
        openai_api_base,
        openai_model,
        text_chunk_size,
        max_chunk
    )


def convert_to_mp3(video_file, audio_file):
    """
    Converts the input video file to MP3 format using ffmpeg and saves the resulting audio file to
    the specified output file path. Returns True if the conversion is successful, False otherwise.
    """
    if os.path.exists(audio_file):
        print(f"Skipping {video_file} - audio file already exists")
        return True
    
    # if file is mp3, just use shutil to copy it
    if video_file.endswith(".mp3"):
        print(f"Copying {video_file} to {audio_file}")
        shutil.copy2(video_file, audio_file)
        return True

    cmd = [
        "ffmpeg",
        "-i",
        video_file,
        "-vn",
        "-acodec",
        "libmp3lame",
        "-ab",
        "192k",
        "-ac",
        "2",
        "-loglevel",
        "quiet",
        audio_file,
    ]
    print(f"Processing: {video_file}")
    print(f"Output audio path: {audio_file}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"Conversion of {video_file} failed")
        return False

    print(f"Conversion of {video_file} success")
    return True


def generate_subtitles(model, task, audio_file, subtitle_file, transcript_file=None):
    """
    Transcribes the specified audio file using the Faster-Whisper model, generates an SRT subtitle file
    at the specified output file path, and optionally generates a transcript file. Returns True if the
    subtitle generation is successful, False otherwise.
    """
    if os.path.exists(subtitle_file):
        print(f"Skipping {audio_file} - subtitles file already exists")
        return True

    segments, info = model.transcribe(audio_file, vad_filter=True, task=task)
    print(
        "Detected language '%s' with probability %f"
        % (info.language, info.language_probability)
    )

    subtitles = []
    texts = []

    for i, segment in enumerate(segments):
        start_time = datetime.timedelta(milliseconds=segment.start * 1000)
        end_time = datetime.timedelta(milliseconds=segment.end * 1000)
        text = segment.text.strip()
        texts.append(text)

        if text:
            # Create a subtitle object for the segment
            subtitle = srt.Subtitle(
                index=i + 1, start=start_time, end=end_time, content=text
            )
            subtitles.append(subtitle)

    # Write the subtitles to the SRT file
    with open(subtitle_file, "w", encoding="utf-8") as f:
        f.write(srt.compose(subtitles))

    print(f"Generation of {subtitle_file} successful")

    # Write the transcript file if requested
    print("transcript_file   " + transcript_file)
    if transcript_file:
        with open(transcript_file, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text + " ")
        print(f"Generation of {transcript_file} successful")

    return True


def summarize(
    transcript_file,
    summary_file,
    summary_language,
    openai_api_key,
    openai_api_base,
    openai_model,
    text_chunk_size,
    max_chunk
):
    # Instantiate the LLM modelI
    if os.path.exists(summary_file):
        print(f"Skipping {summary_file} - summary file already exists")
        return True
    print("Strat summaring " + transcript_file)

    llm = ChatOpenAI(
        model=openai_model,
        temperature=0.7,
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
    )
    with open(transcript_file, "r") as file:
        txt = file.read()
    # print(txt)
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=text_chunk_size,
    )
    texts = text_splitter.split_text(txt)
    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    # Text summarization
    # Write a concise summary of the following
    # CONCISE SUMMARY IN {language}:"""

    if len(docs) == 1:
        stuff_prompt_template = """Please use Markdown syntax to help me summarize the key information and important content. Your response should summarize the main information and important content in the original text in a clear manner, using appropriate headings, markers, and formats to facilitate readability and understanding.Please note that your response should retain the relevant details in the original text while presenting them in a concise and clear manner. You can freely choose the content to highlight and use appropriate Markdown markers to emphasize it. Now summary following content in {language}:

        {text}

        """
        stuff_prompt = PromptTemplate(
            template=stuff_prompt_template,
            input_variables=[
                "text",
            ],
            partial_variables={"language": summary_language},
        )

        chain = load_summarize_chain(
            llm,
            chain_type="stuff",
            prompt=stuff_prompt
        )
        response = chain.run(docs)
        print(response)
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(response)
        return True

    if(len(docs) > max_chunk):
        print('The doc is too long, you should use gpt4-4k or calude to summarize it')
        return True
    map_prompt_template = """Write a concise summary of the following:


    {text}


    SUMMARY IN {language}:"""

    map_prompt = PromptTemplate(
        template=map_prompt_template,
        input_variables=[
            "text",
        ],
        partial_variables={"language": summary_language},
    )

    combine_prompt_template = """Write a concise summary of the following:


    {text}


    CONCISE SUMMARY IN {language}:"""

    combine_prompt = PromptTemplate(
        template=combine_prompt_template,
        input_variables=[
            "text",
        ],
        partial_variables={"language": summary_language},
    )

    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        return_intermediate_steps=True,
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
    )

    response = chain({"input_documents": docs}, return_only_outputs=True)
    print(response)
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Summary_text:\n")
        f.write(response["output_text"])
        f.write("\n\nSection Contents:\n\n")
        for idx, section in enumerate(response["intermediate_steps"]):
            f.write(f"{idx + 1}.{section}\n")
    return True


if __name__ == "__main__":
    config_file = "config.ini"
    (
        input_folder,
        output_folder,
        model_size,
        device,
        compute_type,
        task,
        generate_transcript,
        generate_summary,
        summary_language,
        openai_api_key,
        openai_api_base,
        openai_model,
        text_chunk_size,
        max_chunk
    ) = read_config(config_file)

    logging.basicConfig()
    logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        # if file is video or audio
        if file_name.endswith((".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv", ".mp3")):
            video_file = os.path.join(input_folder, file_name)
            audio_file = os.path.join(
                output_folder, os.path.splitext(file_name)[0] + ".mp3"
            )
            subtitle_file = os.path.join(
                output_folder, os.path.splitext(file_name)[0] + ".srt"
            )
            transcript_file = (
                os.path.join(output_folder, os.path.splitext(file_name)[0] + ".txt")
                if generate_transcript
                else None
            )
            summary_file = (
                os.path.join(
                    output_folder, os.path.splitext(file_name)[0] + ".md"
                )
                if generate_summary
                else None
            )

            if convert_to_mp3(video_file, audio_file):
                generate_subtitles(
                    model,
                    task,
                    audio_file,
                    subtitle_file,
                    transcript_file,
                )
            try:
                if generate_summary:
                    summarize(
                        transcript_file,
                        summary_file,
                        summary_language,
                        openai_api_key,
                        openai_api_base,
                        openai_model,
                        text_chunk_size,
                        max_chunk
                    )
            except Exception as e:
                print(e)
                continue

    print("All conversions complete")
