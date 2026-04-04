import json
import config
from api import call_openai_model_with_tools_ali as call_openai_model


class CoreAgent:
    def __init__(self, question: str,data_name: str, video_duration: float, logger):
        self.question = question
        self.video_duration = video_duration
        self.logger = logger
        self.data_name = data_name
        if data_name == "lv_bench":
            from promp_manager.lv_manager import build_core_prompt, build_system_prompt
        elif data_name == "video_mme":
            from promp_manager.videomme_manager import build_core_prompt, build_system_prompt
        elif data_name == "mlvu":
            from promp_manager.mlvu_manager import build_core_prompt, build_system_prompt
        elif data_name == "longvideo":
            from promp_manager.longvideo_manager import build_core_prompt, build_system_prompt

        self.build_core_prompt = build_core_prompt
        self.build_system_prompt = build_system_prompt

    def run(self, history: list[str]) -> str:
        """
        Makes a decision on which agent to call next based on JSON output.
        """
        prompt = self.build_core_prompt(self.question, history, self.video_duration)

        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = call_openai_model(
            messages=messages,
            temperature=0.0,
        )
        
        return response["content"]

    

    def _get_system_prompt(self) -> str:

        return self.build_system_prompt()