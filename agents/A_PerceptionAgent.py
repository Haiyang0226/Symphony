import config
from tools.perception_tools import frame_inspect_tool, interval_summary_tool, frame_associate_tool, associate
from tools.func_call_shema import as_json_schema

from api import call_openai_model_with_tools_ali as call_openai_model_with_tools


from utils import fix_and_parse_json

class PerceptionAgent:
    def __init__(self, frame_path: str, data_name: str, logger, max_iterations: int = 6):
        # associate
        self.tools = [frame_inspect_tool, interval_summary_tool, frame_associate_tool]
        self.name_to_function_map = {tool.__name__: tool for tool in self.tools}
        self.function_schemas = [
            {"function": as_json_schema(func), "type": "function"}
            for func in self.name_to_function_map.values()
        ]
        self.max_iterations = max_iterations
        self.frame_path = frame_path
        self.logger = logger
        self.data_name = data_name

        if data_name == "lv_bench":
            from promp_manager.lv_manager import S_prompt_perceptionagent as S_prompt
            from promp_manager.lv_manager import prompt_without_q_perceptionagent as prompt_without_q
        elif data_name == "video_mme":
            from promp_manager.videomme_manager import S_prompt_perceptionagent as S_prompt
            from promp_manager.videomme_manager import prompt_without_q_perceptionagent as prompt_without_q
        elif data_name == "mlvu":
            from promp_manager.mlvu_manager import S_prompt_perceptionagent as S_prompt
            from promp_manager.mlvu_manager import prompt_without_q_perceptionagent as prompt_without_q
        elif data_name == "longvideo":
            from promp_manager.longvideo_manager import S_prompt_perceptionagent as S_prompt
            from promp_manager.longvideo_manager import prompt_without_q_perceptionagent as prompt_without_q
        
        self.S_prompt = S_prompt
        self.prompt_without_q = prompt_without_q

    def _construct_messages(self, instruct: str, question: str, time: str):
        self.question = question
        messages = [
            {
                "role": "system",
                "content": self.S_prompt
            },
            {
                "role": "user",
                "content": self.prompt_without_q
            },
        ]
        
        messages[-1]['content'] = messages[-1]['content'].replace("Instruct_PLACEHOLDER", str(instruct))
        
        if "QUESTION_PLACEHOLDER" in self.prompt_without_q:
            messages[-1]["content"] = messages[-1]["content"].replace("QUESTION_PLACEHOLDER", question)
            messages[-1]['content'] = messages[-1]['content'].replace("VIDEO_LENGTH", str(time))

        return messages

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #
    def _append_tool_msg(self, tool_call_id, name, content, msgs):
        m1 = {
            "tool_call_id": tool_call_id,
            "role": "tool",
            "name": name,
            "content": content,
        }
        msgs.append(m1)


    def _parse_tool_args(self, raw_args: str) -> dict | None:
        """
        Parses the raw tool arguments string into a dictionary using the utility function.
        """
        return fix_and_parse_json(raw_args, self.logger)


    def _exec_tool(self, tool_call, msgs):
        name = tool_call.function.name
        if name not in self.name_to_function_map:
            self._append_tool_msg(tool_call.id, name, f"Invalid function name: {name!r}", msgs)
            return

        # Parse arguments using the helper method with LLM fallback
        args_dict = self._parse_tool_args(tool_call.function.arguments)

        if args_dict is None:
            error_message = f"Could not obtain valid arguments for tool call '{name}' after attempting to fix. Original raw args: '{tool_call.function.arguments}'"
            self.logger.error(error_message)
            self._append_tool_msg(tool_call.id, name, error_message, msgs)
            return

        # Inject context-dependent arguments
        tool_func = self.name_to_function_map[name]
        if 'frame_path' in tool_func.__code__.co_varnames:
            args_dict["frame_path"] = self.frame_path

        # Execute the tool
        self.logger.info(f"Calling function `{name}` with args: {args_dict}")
        try:
            result = tool_func(**args_dict)
            self.logger.info(f"Tool '{name}' executed successfully.")
            self.logger.info(f"Tool '{name}' result: {result}")
            self._append_tool_msg(tool_call.id, name, str(result), msgs)
        except Exception as exc:
            error_message = f"Error executing tool '{name}': {exc}"
            self.logger.error(error_message)
            self._append_tool_msg(tool_call.id, name, error_message, msgs)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    
    def run(self, instruct: str, question: str, video_duration: float) -> list[dict]:
        """
        Run the ReAct-style loop with OpenAI Function Calling.
        """
        time_str = convert_seconds_to_hhmmss(video_duration)
        msgs = self._construct_messages(instruct, question, time_str)

        for i in range(self.max_iterations):
            # Force a final `finish` on the last iteration to avoid hanging
            if i == self.max_iterations - 1:
                msgs.append(
                    {
                        "role": "user",
                        "content": "强制返回答案：请总结对话内容并对Instruct进行回答,以[answer]开头",
                    }
                )
            for tt in range(6):
                response = call_openai_model_with_tools(
                    msgs,
                    endpoints=config.AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST,
                    model_name=config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME,
                    tools=self.function_schemas,
                    temperature=0.0,
                    api_key=config.ALi_API_KEY,
                )
                if response is None:
                    return None
                if '[answer]' not in response["content"]:
                    if response.get("tool_calls") == None:
                        self.logger.info(f"tool call None!!! try again -- {response}")
                        continue
                    else:
                        break
                else:
                    break

            self.logger.info("PerceptionAgent Call R1")
            self.logger.info(f"Response: {response}")


            response.setdefault("role", "assistant")
            msgs.append(response)

            # Execute any requested tool calls
            if '[answer]' not in msgs[-1]["content"]:
                
                if response.get("tool_calls", []) == None:
                    # return json.dumps(msgs[2:], indent=2, ensure_ascii=False)
                    return msgs[-1]["content"]

                for tool_call in response.get("tool_calls", []):
                    self._exec_tool(tool_call, msgs)
            else:
                return msgs[-1]["content"]



def convert_seconds_to_hhmmss(seconds):
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


if __name__ == "__main__":
    main()
