import copy
from typing import Annotated as A
import ast
import os
import re
import json
import logging
from api import call_openai_model_with_tools_ali as call_openai_model


def convert_seconds_to_hhmmss(seconds):
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def convert_hhmmss_to_seconds(hhmmss):
    if not isinstance(hhmmss, str):
        return hhmmss # Already in seconds
    hhmmss = hhmmss.split('.')[0]
    parts = hhmmss.split(":")
    if len(parts) < 2:
        raise ValueError("Invalid time format. Expected HH:MM:SS.")
    elif len(parts) == 2:
        parts = ["00"] + parts
    hours, minutes, seconds = map(int, parts)
    return hours * 3600 + minutes * 60 + seconds



class SubtitleAgent:
    def __init__(self,question: str, subtitle_path: str, data_name: str, logger, **kwargs):
        self.subtitle_path = subtitle_path
        self.data_name = data_name
        self.logger = logger
        self.question = question

    def run(self) -> str:
        """
        Analyzes video subtitles based on given instructions and question context.
        """
        start_time = 0
        if not os.path.exists(self.subtitle_path):
            return "No subtitle file found."

        with open(self.subtitle_path, 'r', encoding='utf-8') as f:
            subtitles = json.load(f)

        formatted_subtitles = []
        for subtitle in subtitles:
            start_seconds = convert_hhmmss_to_seconds(subtitle["start"])
            end_seconds = convert_hhmmss_to_seconds(subtitle["end"])

            if start_seconds >= start_time:
                relative_start = start_seconds - start_time
                relative_end = end_seconds - start_time
                
                start_time_str = convert_seconds_to_hhmmss(relative_start)
                end_time_str = convert_seconds_to_hhmmss(relative_end)
                
                text = subtitle.get('line', subtitle.get('text', ''))
                formatted_subtitles.append(f"{start_time_str}-{end_time_str}: {text}")
        
        full_subtitle = " ".join(formatted_subtitles)
        prompt = '''
        Your task is to analyze the video subtitles based on the user's question.

        Based on the following information:
        The original video understanding question:  {question}
        The full video subtitles for analysis: {subtitles}

        Your Analysis Task:
        1. Question-elevant Analysis: Extract subtitle segments directly related to the question from the original subtitles.
        2. Entity and Sentiment Identification: Use the subtitle information to identify key entities mentioned and their associated sentiment.
        3. General Content Summary: Provide a brief, high-level summary of the overall topic covered in the subtitle content.

        Please respond strictly in the following JSON format:
        {{
          "relevant_subtitle_info": "A multi-line string containing the most relevant subtitle segments. Format each entry as:\n[HH:MM:SS - HH:MM:SS]: Actual subtitle text.\nFor example:\n[00:15:32 - 00:15:35]: ...\n[00:18:05 - 00:18:09]: ...",
          "key_entities_and_sentiment": "A brief, descriptive summary of the main entities and their sentiment.",
          "overall_topic": "A one-sentence summary of the main topic discussed in the video, based only on the subtitles."
        }}

        Please return only the JSON object.
        '''

        prompt=prompt.format(question=self.question, subtitles=full_subtitle)
        messages = [
            {
                "role": "system",
                "content": 'You are a specialized Subtitle Analysis agent.'
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = call_openai_model(
            messages=messages,
            temperature=0.0,
            api_key=config.ALi_API_KEY,
        )

        return response["content"]