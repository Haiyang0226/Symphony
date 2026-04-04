import copy
from typing import Annotated as A
import ast
import config
from tools.perception_tools import retrieve_tool, retrieve_and_ans_tool
from tools.localize_tools import localize_tool
from tools.func_call_shema import as_json_schema
from tools.func_call_shema import doc as D
from api import call_openai_model_with_tools_ali as call_openai_model_with_tools

from utils import fix_and_parse_json


class StopException(Exception):
    """
    Stop Execution by raising this exception (Signal that the task is Finished).
    """
def finish(answer: A[str, D("For Type 0, return the complete positioning result directly.")]) -> None:
    raise StopException(answer)


class LocalizeAgent:
    def __init__(self, video_duration, frame_path, question, data_name, logger):
        self.tools = [localize_tool, retrieve_tool, finish]
        self.name_to_function_map = {tool.__name__: tool for tool in self.tools}
        self.function_schemas = [
            {"function": as_json_schema(func), "type": "function"}
            for func in self.name_to_function_map.values()
        ]
        
        self.video_duration_sec = video_duration
        self.video_duration_str = convert_seconds_to_hhmmss(video_duration)
        self.frame_path = frame_path
        self.question = question
        self.logger = logger
        self.data_name = data_name

        if data_name == "lv_bench":
            from promp_manager.lv_manager import S_prompt_localizeagent as S_prompt
            from promp_manager.lv_manager import prompt_localizeagent as prompt
        elif data_name == "video_mme":
            from promp_manager.videomme_manager import S_prompt_localizeagent as S_prompt
            from promp_manager.videomme_manager import prompt_localizeagent as prompt
        elif data_name == "mlvu":
            from promp_manager.mlvu_manager import S_prompt_localizeagent as S_prompt
            from promp_manager.mlvu_manager import prompt_localizeagent as prompt
        elif data_name == "longvideo":
            from promp_manager.longvideo_manager import S_prompt_localizeagent as S_prompt
            from promp_manager.longvideo_manager import prompt_localizeagent as prompt

        self.S_prompt = S_prompt
        self.prompt = prompt

    def _construct_messages(self):
        messages = [
            {"role": "system", "content": self.S_prompt},
            {"role": "user", "content": self.prompt},
        ]
        messages[-1]['content'] = messages[-1]['content'].replace("VIDEO_LENGTH", str(self.video_duration_str))
        messages[-1]["content"] = messages[-1]["content"].replace("QUESTION_PLACEHOLDER", self.question)

        return messages


    def _exec_tool(self, tool_call):
        name = tool_call.function.name
        if name not in self.name_to_function_map:
            return f"Invalid function name: {name!r}"

        args_dict = fix_and_parse_json(tool_call.function.arguments, self.logger)

        if args_dict is None:
            return f"Error decoding arguments for tool {name}: {tool_call.function.arguments}"

        self.logger.info(f"Calling function `{name}` with args: {args_dict}")

        # Inject frame_path if needed by the tool
        if "frame_path" in self.name_to_function_map[name].__code__.co_varnames:
             args_dict["frame_path"] = self.frame_path
        
        # Inject video_duration if needed by the tool
        if "video_duration" in self.name_to_function_map[name].__code__.co_varnames:
             args_dict["video_duration"] = self.video_duration_sec

        try:
            result = self.name_to_function_map[name](**args_dict)
            self.logger.info(f"Tool `{name}` executed successfully.")
            return result if result is not None else "Tool executed."
        except StopException as exc:
            self.logger.info(f"Finish task with message: '{exc!s}'")
            raise
        except Exception as exc:
            self.logger.error(f"Error executing tool {name}: {exc}")
            return f"Error: {exc}"

    def run(self) -> str:
        """
        Run a single decision step to choose and execute a tool.
        If localize_tool is chosen, summarize its long output.
        """
        msgs = self._construct_messages()

        for times_i in range(6):
            response = call_openai_model_with_tools(
                msgs,
                endpoints=config.AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST,
                model_name=config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME,
                tools=self.function_schemas,
                temperature=0.0,
                api_key=config.OPENAI_API_KEY,
            )
            if response.get("tool_calls") == None:
                self.logger.info(f"tool call None!!! try again -- {response}")
                continue
            else:
                break

        if response is None:
            return "Failed to get response from LLM."

        self.logger.info(f"LocalizeAgent LLM Response: {response}")

        if not response.get("tool_calls"):
            return response.get("content", "No action taken.")

        tool_call = response["tool_calls"][0]
        result = self._exec_tool(tool_call)
        return str(result)


def convert_seconds_to_hhmmss(seconds):
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


