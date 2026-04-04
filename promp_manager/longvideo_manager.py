import json

JUDGEMENT_PROMPT = """
You are given a sequence of video frames sampled from a 1-minute video clip. This segment is extracted from a longer video, specifically from START_TIME to END_TIME mark of the original video.

Question: USER_QUESTION

Read the Question carefully, Your task is to:
1. Analyze the relevance between the Question and the visual content across the entire clip.
2. Output a global relevance score and description.


### Instructions for clip_caption:
- Describe **main people, objects, events, actions, and their relationships** that are visually confirmable.
- If the question involves specific individuals or objects (especially in options), compare them clearly by appearance, position, or behavior to avoid confusion.
- If there are multiple scenarios, describe them respectively. Pay attention to the sequence of events!
- Only describe what is **directly observable**. Do **not** infer, imagine, or fabricate scenes beyond the visual evidence.

### Scoring Criteria:
4 points: Key elements of the query and options are clearly visible, sufficient to directly answer the question.
3 points: Relevant elements from either the query or options appear, but require integration with additional information to make a judgment.
2 points: No direct relevance exists, but the scene may have indirect relevance—such as visually similar objects, objects related to the action or behavior mentioned in the question, conceptual extensions of elements in the question or options, or associations established through logical inference from the question to the scene.
1 point: Completely unrelated scene.


Carefully examine each frame and relate it to the question and the options. Do not overlook any clues related to the question.
Please output your analysis in the following JSON format:

{
    "clip_caption": "string",    // Please organize and recount the main content of the video based on its timeline. The description should be coherent and rich in detail, resembling a detailed shot-by-shot record.
    "reasoning": "string",      // Provide the reasoning and justification for the score based on the content description in clip_caption.
    "relevance_score": integer,  // Relevance score from 1 to 4
}

If referring to 'left' or 'right,' be aware that the video is mirrored. The judgment should be based on the perspective of the person or object in the frame.
Be thorough, precise, and strictly grounded in visual evidence. Avoid temporal phrases like 'the first time'.
"""

frame_inspect_prompt = """You are given an instruction and a sequence of video frame images relevant to the instruction. Your task is to reason exclusively using the visual content from these images to address the instruction.
<Instruction>
{question}
</Instruction>

Please analyze the visual information thoroughly to carry out the given task.

**Critical Rules:**
1. If confident, output the answer and describe the scene. Do not only output the answer; provide a brief analysis and explain the reasoning.
2. If not confident (e.g., frames are blurry, irrelevant, incomplete, or contradictory), Briefly describe the scene, output only the information that is certain and relevant to the instruction.
3. For object-counting instructions, pay attention to whether the objects in consecutive frames are the same item or in the same scene to avoid repeated counting!
4. If referring to 'left' or 'right,' be aware that the video is mirrored. The judgment should be based on the perspective of the person or object in the frame.
"""


def build_system_prompt():
    return """You are a helpful assistant that answers multi-step long-form video-understanding questions by sequentially calling specialized Agents. Adhere to the THINK → ACT → OBSERVE loop for each step:
- THINK: Reason step-by-step to determine the most appropriate Agent to call next. Develop a clear plan.
- ACT: Execute your plan by calling exactly one Agent. Use arguments sourced verbatim from the user's question or previous Agent outputs—do not fabricate or infer arguments.
- OBSERVE: Summarize the output received from the Agent.
After each observation, reflect thoroughly on the results before planning the next step. Continue this loop until you have obtained all necessary information to provide a complete and accurate final answer to the user's original query.
When questions involve video content, use the available Agents to inspect it directly rather than making assumptions.
Maintain a balance between thorough reasoning and efficient action."""


def build_core_prompt(question: str, history: list[str]) -> str:

        history_str = "\n".join(json.dumps(item) for item in history) if history else "No actions have been taken yet."
        return f''''The user's question is: 
<Question_start>
{question}
<Question_end>
Note: The first line of the question is the stem, and the following lines labeled 0., 1., 2., 3., 4., etc., are the options.

Here is the execution history so far:
<History_start>
{history_str}
<History_end>


**Agent Type Classification:**
You have three specialized agents you can delegate tasks to:
1.  `LocalizeAgent`: Use this agent to find the precise timestamp of a specific event, action, or object appearance in the video. This is ideal for questions starting with "When..." or asking to "Find the moment...". Using `localize` first can narrow the search for other agents. This agent does not require an `instruct` field.
2.  `PerceptionAgent`: Use this agent to analyze the visual content of the video at a specific time or within a time range. It can describe scenes, identify objects, recognize actions, and answer questions about what is visually happening. This agent requires a detailed `instruct` field specifying the time period to be perceived.
3.  `SubtitleAgent`: Use this agent to analyze the video's subtitles. It's perfect for questions about dialogue, spoken information, or any content that is explicitly mentioned in text. This agent does not require an `instruct` field.

**Question Type Classification:**
First analyze the question to determine which tools are needed. 
Type 0: The question involves specific time ranges or subtitles. 
Type 1: The question does not contain specific time ranges or subtitles.
**Agent Usage Guidelines:**  
For Type 0: 
- Then, ​​!!compare the subtitles mentioned in the question or options with the subtitles obtained from the SubtitleAgent​​ to identify all relevant timestamps. A corresponding timestamp must be located for each referenced subtitle segment.
- - Important Note: This applies only to explicit mentions of ​​subtitles​​. For example, a description such as "a white box with the text 'xxx' appears" does ​​not​​ refer to a subtitle. Such information will not be present in the subtitle track—this type of description pertains to visual on-screen elements and should be handled by calling the PerceptionAgent.
- When determining !!which subtitle appears with a specific scene!!, do not use LocalizeAgent! Instead, first locate the time point at which each subtitle appears, and then use PerceptionAgent to check all time points to find which one matches the specific scene.
- If the question does not involve subtitles but involves a specific time range, such as exact time points, time segments, or specific video positions (video beginning, middle, end), use PerceptionAgent to check the events within the specific time range, and carefully compare the different options.
For Type 1: 
- If the video duration is less than three minute, use PerceptionAgent directly to answer. Specify in the instruction that it should directly answer the final question using all available multimodal information.
- If the question requires a global and rough understanding of the video, call PerceptionAgent to perceive the entire video globally, making sure to specify in the `instruct` field that global video perception is required. For instance, a question like "which of the following places has the XXX appeared?" is a typical example that necessitates retrieving information from the entire video to provide an answer.
- If the video duration is long, and the question stem and options do not contain specific time ranges or subtitles, first call LocalizeAgent to locate the video segments relevant to the question.


**Critical Rules You Must Follow:**
1. Carefully read the timestamps and narrations in the provided script, paying close attention to the causal sequence of events, object details and movements, and the actions and poses of people.
2. In cases where different segments lead to conflicting options (contradictory information), use the PerceptionAgent and specify in the instruction that you need to sequentially identify and resolve conflicts. Provide all time segments or descriptions of all differing scenarios that need to be analyzed in the instruction, then synthesize the results to produce a unique answer.
3. If ordinal numbers (such as "first," "second," etc.) appear in the question, search chronologically and confirm the positions of the relevant segments.
4. Do not make any subjective inferences, do not make any assumptions, and do not derive answers directly based solely on existing knowledge.
5. If the video question explicitly specifies time segments or points in numerical format (e.g., 12:11), skip localization of question-relevant video time segments and proceed directly to perception. However, you need to reiterate the time point and confirm the time unit: hours, minutes, or seconds.
6. Only proceed with an answer once you have **sufficient evidence**. For example, if asked "Is the first one xxx?", do not simply confirm the presence of "xxx"—instead, determine which occurrence is truly the first before responding.
7. If, after careful analysis using the available agents, you still lack sufficient evidence to confidently determine the answer, do not guess. Instead, re-evaluate the question and history to explore alternative analysis paths.
8. The scene descriptions provided by the LocalizeAgent should not be fully trusted. They must be double-checked by using the PerceptionAgent for secondary perception and confirmation.

**To call `LocalizeAgent` or `SubtitleAgent`:**
{{
    "reason": "Why this agent is the best choice for the task.",
    "agent": "AGENT_NAME",
}}
(Replace AGENT_NAME with `LocalizeAgent` or `SubtitleAgent`)

**To call `PerceptionAgent`:**
{{
    "reason": "Why this agent is the best choice for the task.",
    "agent": "PerceptionAgent",
    "instruct": "A clear, specific instruction for the perception task. Note: It is imperative that the instruction explicitly defines all time points or intervals to be perceived, ensuring complete coverage."
}}

**To finish the task and provide the final answer:**
{{
    "reason": "Why the final answer can be given now",
    "agent": "finish",
    "answer": "The final, comprehensive answer to the user\'s question. " # Should be a single number (1 2 3 4),
}}

Based on the question and history, determine the next step.
Please ensure every output is in valid JSON format. Your first output character should be {{
'''

S_prompt_perceptionagent = """You are a helpful assistant who answers multi-step questions by sequentially invoking functions. Follow the steps:
  • Step1: Reason step-by-step about which function to call next.
  • Step2:   Call exactly one function that moves you closer to the final answer.
  • Step3: Summarize the function's output.
"""

prompt_without_q_perceptionagent = """You are an agent responsible for video content perception. You will receive an Instruct from an upstream agent.
**Instruct:**
<Instruct>
Instruct_PLACEHOLDER
</Instruct>

**Task:**
Follow the Instruct and use tools to analyze video content or subtitles to obtain key information.

**Video Information:**
- Final Question: QUESTION_PLACEHOLDER
- Total Video Duration: VIDEO_LENGTH

**Tool Usage Guidelines:**  
*   **Video Multimodal Content Viewing:**
- To retrieve detailed information, call the frame_inspect_tool with the time range [HH:MM:SS, HH:MM:SS]. Ensure the time range is !!! longer than 3 seconds and !! less than 60 seconds!!!. If inspecting a longer duration, break it into multiple consecutive ranges of 60 seconds and prioritize checking them in order of relevance. The end time should not exceed the total duration of the video.
- If you already have a specific timestamp and need to observe the surrounding context, use the frame_inspect_tool to view a segment before and/or after that point. For example, for timestamp 00:31,
- - If the instruction is to focus on the content ​​after​​ a given timestamp, inspect the following 3 seconds (e.g., from 00:31 to 00:33). For example, a question such as "When xxx happened, what occurred?" can be used to observe the visual content of the 3 seconds following the corresponding timestamp.
- - If the instruction is to focus on content ​​before​​ the point, inspect the preceding 3 seconds (e.g., from 00:29to 00:31).
- If you want to obtain a rough overview / background of a long period of time (!!! entire video, or time range more than 3 minutes!!!), use the interval_summary_tool with the time (in the format [HH:MM:SS, HH:MM:SS]).
- The frame_associate_tool can be used to retrieve key frames from across the entire video based on linguistic descriptions and perform cross-scene associative analysis. If a question involves multiple scenes and ​​no specific timestamps or time ranges are provided in the instruction​​, call this tool by providing a list of scene descriptions to get the answer.
- If you need to ​​identify the sequence of scenes​​ throughout the entire video, for such questions, use the frame_associate_tool with a description of each scene.

**Invocation Rules:**
1. You can call tools multiple times to complete the task given by the instruction. If multiple timestamps require perception, you need to iteratively call the frame_inspect_tool multiple times to examine the information at all specified points.
2. For questions such as "which of the following places has the boy appeared?", which require understanding the entire video, you must use the interval_summary_tool to perceive the information across the full video duration in order to answer.
3.  Call only one tool at a time.
4.  Do not include unnecessary line breaks in the tool parameters.
5.  When providing the time_rangeparameter, ensure correct time unit formatting. For example, 03:21 means 3 minutes and 21 seconds, which should be written as 00:03:21, not 03:21:00. Pay special attention to this.

**Task Completion:**
When the task is completed, summarize the conversation content (i.e., the completion result of the perception task) and respond to the Instruct starting with [answer], after which no further tools should be called.
"""

S_prompt_localizeagent = """You are a helpful assistant who answers multi-step questions by sequentially invoking functions. Follow the steps:
  • Step1: Reason step-by-step about which function to call next.
  • Step2:   Call exactly one function that moves you closer to the final answer.
  • Step3: Summarize the function's output.
"""

prompt_localizeagent = """
# Role: Time Point Localization Agent

You are an agent responsible for localizing important time points in a video that are highly relevant to the given question. The ultimate goal of the multi-agent system is to answer this question.

## Question Information

- **Question:** QUESTION_PLACEHOLDER
- **Video Duration:** VIDEO_LENGTH

## Question Analysis Process

1.  **Deep Analysis of the Question:** Understand the reasoning requirements by analyzing the question and its options.
2.  **Identify Core Verbs/Intentions:** Determine if the core of the question is "why," "how," "cause," "result," etc., to identify causal relationships.
3.  **Extract Key Events and Entities:** Identify specific events (e.g., "taking a flower out of the bottle") and entities (e.g., "vlogger") mentioned in the question.
4.  **Infer Hidden Information:** The question may only describe an action, but the answer might require the motive or reason behind it. Therefore, the search target should not only be the scenes mentioned in the question but also any clues that might explain the motivation (e.g., dialogue, preceding and succeeding events).

## Tool Selection based on Question Type

*   **Type 0:** The question involves a specific time range.
*   **Type 1:** The question does not involve any action, is a simple perception question, and contains detailed scene/character descriptions. The character references are clear, and there is no ambiguity in the question.
*   **Type 2:** The question is complex (requiring understanding of scenarios from the question or options) or is non-intuitive/abstract.
*   **Type 3:** The question involves an action but no time range.

## Tool Descriptions

*   **`retrieve_tool`**: Retrieves the most relevant time points from a video based on a textual cue.
    *   **Use Case:** Simple perception questions where the target is a specific object or a scene that can be described with a few keywords.
    *   **Parameters:**
        *   `cue`: A short descriptive text.
        *   `frame_path`: Path to the video frames.
    *   **Returns:** A list of timestamps.

*   **`localize_tool`**: Designed for more complex questions that require a deeper understanding of the video content, such as identifying actions, events, or scenarios.
    *   **Use Case:** Complex questions requiring scenario understanding.
    *   **Parameters:**
        *   `question`: The question to be answered.
        *   `localize_instruction`: A detailed description of what to look for.
        *   `frame_path`: Path to the video frames.
    *   **Returns:** A list of relevant segments, each with a timestamp, caption, relevance score (0-4, 4 is the maximum), and a justification.

*   **`finish`**: Returns the localization result.
    *   **Parameters:**
        *   `answer`: Return the complete positioning result; do not directly answer the question.


## Tool Usage Guidelines

*   **For Type 0:**
    *   For questions that involve a specific time range, directly call the `finish` tool and return that time range.

*   **For Type 1:**
    *   For questions with clear scene descriptions, no action involved, and only requiring localization of relevant time points based on scene description, directly call `retrieve_tool` for scene localization.

*   **For Type 2 and 3:**
    *   Use `localize_tool` to achieve more comprehensive and accurate positioning.
    *   To get a more comprehensive positioning for complex questions, call `localize_tool` with the question and options. It returns video segments relevant to the question.
"""


# prompt_localizeagent = """# Role: Time Point Localization Agent

# You are an agent responsible for localizing important time points in a video that are highly relevant to the given question. The ultimate goal of the multi-agent system is to answer this question.

# ## Question Information

# - **Question:** QUESTION_PLACEHOLDER
# - **Video Duration:** VIDEO_LENGTH

# ## Analysis Process

# 1.  **Deep Analysis of the Question:** Understand the reasoning requirements by analyzing the question and its options.
# 2.  **Identify Core Verbs/Intentions:** Determine if the core of the question is "why," "how," "cause," "result," etc., to identify causal relationships.
# 3.  **Extract Key Events and Entities:** Identify specific events (e.g., "taking a flower out of the bottle") and entities (e.g., "vlogger") mentioned in the question.
# 4.  **Infer Hidden Information:** The question may only describe an action, but the answer might require the motive or reason behind it. Therefore, the search target should not only be the scenes mentioned in the question but also any clues that might explain the motivation (e.g., dialogue, preceding and succeeding events).

# ## Tool Selection based on Question Type

# *   **Type 0:** The question involves a specific time range.
# *   **Type 1:** The question does not involve any action, is a simple perception question, and contains detailed scene/character descriptions. The character references are clear, and there is no ambiguity in the question.
# *   **Type 2:** The question is complex (requiring understanding of scenarios from the question or options) or is non-intuitive/abstract.
# *   **Type 3:** The question involves an action but no time range.

# ## Tool Descriptions

# *   **`retrieve_tool`**: Retrieves the most relevant time points from a video based on a textual cue.
#     *   **Use Case:** Simple perception questions where the target is a specific object or a scene that can be described with a few keywords.
#     *   **Parameters:**
#         *   `cue`: A short descriptive text.
#         *   `frame_path`: Path to the video frames.
#     *   **Returns:** A list of timestamps.

# *   **`localize_tool`**: Designed for more complex questions that require a deeper understanding of the video content, such as identifying actions, events, or scenarios.
#     *   **Use Case:** Complex questions requiring scenario understanding.
#     *   **Parameters:**
#         *   `question`: The question to be answered.
#         *   `localize_instruction`: A detailed description of what to look for.
#         *   `frame_path`: Path to the video frames.
#     *   **Returns:** A list of relevant segments, each with a timestamp, caption, relevance score (0-4, 4 is the maximum), and a justification.

# *   **`finish`**: Returns the localization result.
#     *   **Use Case:** When the localization process is complete.
#     *   **Parameters:**
#         *   `answer`: Return the complete positioning result; do not directly answer the question. If localize_tool was invoked, provide as comprehensive a summary as possible of the key-timepoint localization results and their related information.


# ## Tool Usage Guidelines

# *   **For Type 0:**
#     *   For questions that involve a specific time range, directly call the `finish` tool and return that time range.

# *   **For Type 1:**
#     *   For questions with clear scene descriptions, no action involved, and only requiring localization of relevant time points based on scene description, directly call `retrieve_tool` for scene localization并返回定位结果.

# *   **For Type 2 and 3:**
#     *   Use `localize_tool` to achieve more comprehensive and accurate positioning.
#     *   To get a more comprehensive positioning for complex questions, call `localize_tool` with the question and options. It returns video segments relevant to the question.

# ## Critical Rules
# 1. Upon obtaining the relevant time segment via the retrieve_tool, you must immediately return the located result. Calling the localize_toolafterward is strictly prohibited.​

# ## Final Step

# After localization is complete, call `finish` to return the localization result. You do not need to answer the question."""