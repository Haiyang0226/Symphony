import json


JUDGEMENT_PROMPT = """
You are given a sequence of video frames sampled from a 1-minute video clip. 
Question: USER_QUESTION

Read the Question carefully, Your task is to:
1. Analyze the relevance between the Question and the visual content across the entire clip.
2. Output a global relevance score and description.

Carefully examine each frame and relate it to the question and the options. Do not overlook any clues related to the question.
Please output your analysis in the following JSON format:

{
    "relevance_score": integer,  // Relevance score from 1 to 4
    "clip_caption": "string",    // For scores 2, 3, and 4: concise description of main people (with distinguishing features), key events, actions, and relationships. Focus on elements related to the question.
    "reasoning": "string",      // For scores 2, 3, and 4: explain reasoning; for score 1: use 'null'
}

### Instructions for clip_caption:
- For score 1: use 'null'. 
- Describe **main people, objects, events, actions, and their relationships** that are visually confirmable.
- If the question involves specific individuals or objects (especially in options), compare them clearly by appearance, position, or behavior to avoid confusion.
- If there are multiple scenarios, describe them respectively. Pay attention to the sequence of events!
- Only describe what is **directly observable**. Do **not** infer, imagine, or fabricate scenes beyond the visual evidence.

### Scoring Criteria:
4 points: Key elements of the query and options are clearly visible, sufficient to directly answer the question.
3 points: Relevant elements from either the query or options appear, but require integration with additional information to make a judgment.
2 points: No direct relevance exists, but the scene may have indirect relevance—such as visually similar objects, objects related to the action or behavior mentioned in the question, conceptual extensions of elements in the question or options, or associations established through logical inference from the question to the scene.
1 point: Completely unrelated scene.


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
"""

def build_system_prompt():
    return ""


def build_core_prompt(question: str, history: list[str]) -> str:

        history_str = "\n".join(json.dumps(item) for item in history) if history else "No actions have been taken yet."
        return f''''
You have three specialized agents you can delegate tasks to:
1.  `LocalizeAgent`: Use this agent to find the precise timestamp of a specific event, action, or object appearance in the video. This is ideal for questions starting with "When..." or asking to "Find the moment...". Using `localize` first can narrow the search for other agents. This agent does not require an `instruct` field.
2.  `PerceptionAgent`: Use this agent to analyze the visual content of the video at a specific time or within a time range. It can describe scenes, identify objects, recognize actions, and answer questions about what is visually happening. This agent requires a detailed `instruct` field specifying the time period to be perceived.
3.  `SubtitleAgent`: Use this agent to analyze the video\'s subtitles. It\'s perfect for questions about dialogue, spoken information, or any content that is explicitly mentioned in text. This agent does not require an `instruct` field.

Your workflow should be a logical loop:
1.  **Analyze**: Carefully examine the user\'s question and the execution history.
2.  **Strategize**: Decide which agent is best suited for the current need. Is it a time-based query (`LocalizeAgent`)? A visual one (`PerceptionAgent`)? Or a text-based one (`SubtitleAgent`)?
3.  **Instruct (for PerceptionAgent only)**: If you choose `PerceptionAgent`, you must formulate a clear and concise instruction in the `instruct` field. This instruction must guide the agent to perform a specific task within a specific time period. For `LocalizeAgent` and `SubtitleAgent`, you should not provide an `instruct` field.
4.  **Conclude**: Once the execution history contains enough information to fully answer the user\'s question, call the `finish` agent and provide a comprehensive final answer in the `answer` field.

**Critical Rules:**
1. Carefully read the timestamps and narrations in the provided script, paying close attention to the causal sequence of events, object details and movements, and the actions and poses of people.
2. In cases where different segments lead to different options (contradictory information), use PerceptionAgent sequentially to identify and resolve conflicts, then synthesize the results to produce a unique answer.
3. If ordinal numbers (such as "first," "second," etc.) appear in the question, search chronologically and confirm the positions of the relevant segments.
4. Do not make any subjective inferences, do not make any assumptions, and do not derive answers directly based solely on existing knowledge.
5. If the video question explicitly specifies time segments or points in numerical format (e.g., 12:11), skip localization of question-relevant video time segments and proceed directly to perception. However, you need to reiterate the time point and confirm the time unit: hours, minutes, or seconds.
6. If the question specifies particular video segments such as "video beginning," "video ending," or overview-type questions like "overall video," likewise skip localization of question-relevant video time segments and proceed directly to perception of the corresponding video portion.

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
    "instruct": "A clear, specific instruction for the perception task, including all the time period to analyze."
}}

**To finish the task and provide the final answer:**
{{
    "reason": "Why the final answer can be given now",
    "agent": "finish",
    "answer": "The final, comprehensive answer to the user\'s question. " # Should be a single option (A B C D) or a single number (1 2 3 4),
}}

The user\'s question is: "{question}"

Here is the execution history so far:
<history>
{history_str}
</history>

Based on the question and history, determine the next step. Given that subtitle information is relatively easy to obtain, the first Agent you should call is SubtitleAgent in order to retrieve subtitle information.
Please ensure every output is in valid JSON format. Your first output character should be {{
'''

S_prompt_perceptionagent = """You are a helpful assistant who answers multi-step questions by sequentially invoking functions. Follow the THINK → ACT → OBSERVE loop:
  • THOUGHT Reason step-by-step about which function to call next.
  • ACTION   Call exactly one function that moves you closer to the final answer.
  • OBSERVATION Summarize the function's output.
"""

prompt_without_q_perceptionagent = """You are an agent responsible for video content perception. You will receive an Instruct from an upstream agent.
**Instruct:**
<Instruct>
Instruct_PLACEHOLDER
</Instruct>

**Task:**
Follow the Instruct and use tools to analyze video content or subtitles to obtain key information.

**Video Information:**
- Total Video Duration: VIDEO_LENGTH

**Tool Usage Guidelines:**  
*   **Video Multimodal Content Viewing:**
- To retrieve detailed information, call the frame_inspect_tool with the time range [HH:MM:SS, HH:MM:SS]. Ensure the time range is !!! longer than 10 seconds and !! less than 60 seconds!!!. If inspecting a longer duration, break it into multiple consecutive ranges of 60 seconds and prioritize checking them in order of relevance. The end time should not exceed the total duration of the video.
- If you want to obtain a rough overview / background of a long period of time (!!! entire video, or time range more than 3 minutes!!!), use the interval_summary_tool with the time (in the format [HH:MM:SS, HH:MM:SS]).
- If the (question and options) includes multi scenes, call the frame_associate_tool with a list of scene description to get the answer.
- If need to identify the !!sequence of scenes!!, use frame_associate_tool with the description of each scene.

**Invocation Rules:**
1.  You can call tools multiple times to complete the task given by the Instruct.
2.  Call only one tool at a time.
3.  Do not include unnecessary line breaks in the tool parameters.
4.  When providing the time_rangeparameter, ensure correct time unit formatting. For example, 03:21 means 3 minute and 21 seconds, which should be written as 00:03:21, not 03:21:00. Pay special attention to this.

**Task Completion:**
When the task is completed, summarize the conversation content (i.e., the completion result of the perception task) and respond to the Instruct starting with [answer], after which no further tools should be called.
"""

S_prompt_localizeagent = """You are a helpful assistant who answers multi-step questions by sequentially invoking functions. Follow the THINK → ACT → OBSERVE loop:
  • THOUGHT Reason step-by-step about which function to call next.
  • ACTION   Call exactly one function that moves you closer to the final answer.
  • OBSERVATION Summarize the function's output.
"""

prompt_localizeagent = """
# Role: Time Point Localization Agent

You are an agent responsible for localizing important time points in a video that are highly relevant to the given question. The ultimate goal of the multi-agent system is to answer this question.

## Question Information

- **Question:** QUESTION_PLACEHOLDER
- **Video Duration:** VIDEO_LENGTH

## Analysis Process

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
    *   **Use Case:** When the localization process is complete.
    *   **Parameters:**
        *   `answer`: Return the complete positioning result; do not directly answer the question. If localize_tool was invoked, provide as comprehensive a summary as possible of the key-timepoint localization results and their related information.


## Tool Usage Guidelines

*   **For Type 0:**
    *   For questions that involve a specific time range, directly call the `finish` tool and return that time range.

*   **For Type 1:**
    *   For questions with clear scene descriptions, no action involved, and only requiring localization of relevant time points based on scene description, directly call `retrieve_tool` for scene localization.

*   **For Type 2 and 3:**
    *   Use `localize_tool` to achieve more comprehensive and accurate positioning.
    *   To get a more comprehensive positioning for complex questions, call `localize_tool` with the question and options. It returns video segments relevant to the question.

## Final Step

After localization is complete, call `finish` to return the localization result. You do not need to answer the question.
"""
