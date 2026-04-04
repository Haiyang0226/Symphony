import copy
import base64
from openai import OpenAI
import time
from mimetypes import guess_type
import cv2


def local_image_to_data_url(image_path,down_sample_frame=True):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # Default MIME type if none is found

    # Read the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from path: {image_path}")

    # Get image dimensions
    height, width = image.shape[:2]

    # Check if any dimension exceeds 2000 pixels
    if down_sample_frame and (width > 1000 or height > 1000):
        image = cv2.resize(image, (width // 2, height // 2), interpolation=cv2.INTER_AREA)

    elif width > 2000 or height > 2000:
        # Perform 2x2 down-sampling (reduce to half size)
        image = cv2.resize(image, (width // 2, height // 2), interpolation=cv2.INTER_AREA)

    # Encode the image to JPEG in memory (optional: compress as JPEG)
    # You can change .jpg to .png if you prefer lossless compression
    success, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not success:
        raise ValueError("Could not encode image to JPEG format.")

    # Convert to base64
    base64_encoded_data = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_encoded_data}"


def call_seed_vl_with_tools_huoshan(
        messages,
        model_name="doubao-seed-1-6-vision-250815",
        api_key="",  # your API
        tools: list = [],
        image_paths: list = [],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        tool_choice: str = "auto",
        return_json: bool = False,

) -> dict:

    stream = False
    base_url = "https://ark.cn-beijing.volces.com/api/v3"
    client = OpenAI(api_key=api_key, base_url=base_url)

    processed_messages = copy.deepcopy(messages)

    if image_paths and len(messages) == 2:
        # data URL
        image_data_list = [local_image_to_data_url(img_path) for img_path in image_paths]

        if processed_messages and processed_messages[-1]["role"] == "user":
            last_content = processed_messages[-1]["content"]

            if not isinstance(last_content, list):
                last_content = [{"type": "text", "text": last_content}]

            for image_data in image_data_list:
                last_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_data}
                })

            processed_messages[-1]["content"] = last_content
        else:
            content_list = []
            for image_data in image_data_list:
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": image_data}
                })
            processed_messages.append({"role": "user", "content": content_list})

    elif image_paths and len(messages) == 3:
        image_data_list_context_image = [local_image_to_data_url(img_path) for img_path in image_paths[1]]
        image_data_list_this_clip = [local_image_to_data_url(img_path) for img_path in image_paths[0]]

        if processed_messages and processed_messages[1]["role"] == "user":
            middle_content = processed_messages[1]["content"]

            if not isinstance(middle_content, list):
                middle_content = [{"type": "text", "text": middle_content}]

            for image_data in image_data_list_context_image:
                middle_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_data}
                })

            last_content = processed_messages[-1]["content"]
            if not isinstance(last_content, list):
                last_content = [{"type": "text", "text": last_content}]

            for image_data in image_data_list_this_clip:
                last_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_data}
                })
            merged_content = middle_content + last_content

            processed_messages[1]["content"] = merged_content
            processed_messages.pop()

    payload = {
        "model": model_name,
        "messages": processed_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream
    }

    if return_json:
        payload["response_format"] = {"type": "json_object"}

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice


    max_retries = 5
    retry_delay = 10
    for attempt in range(max_retries):
        try:

            response = client.chat.completions.create(**payload)
            message = response.choices[0].message


            if hasattr(message, 'tool_calls') and message.tool_calls:
                return {
                    "role": message.role,
                    "content": message.content,
                    "tool_calls": message.tool_calls
                }
            else:
                return {
                    "role": message.role,
                    "content": message.content or "",
                    "tool_calls": None
                }

        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"API failed，tried {max_retries} tims: {str(e)}")

            print(f" {attempt + 1} failed, retry in {retry_delay} seconds... error: {str(e)}")
            time.sleep(retry_delay)
            retry_delay *= 2


def call_openai_model_with_tools_ali(
    messages,
    endpoints=None,
    model_name="deepseek-r1",
    api_key = "",  # API
    tools: list = [],
    image_paths: list = [],
    max_tokens: int = 4096,
    temperature: float = 0.0,
    tool_choice: str = "required",
    return_json: bool = False,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
) -> dict:

    stream = False
    client = OpenAI(api_key=api_key, base_url=base_url)
    processed_messages = copy.deepcopy(messages)
    print(model_name)


    if image_paths and len(messages) == 2:
        image_data_list = [local_image_to_data_url(img_path) for img_path in image_paths]
        
        if processed_messages and processed_messages[-1]["role"] == "user":
            last_content = processed_messages[-1]["content"]
            
            if not isinstance(last_content, list):
                last_content = [{"type": "text", "text": last_content}]
                
            for image_data in image_data_list:
                last_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_data}
                })
            
            processed_messages[-1]["content"] = last_content
        else:
            content_list = []
            for image_data in image_data_list:
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": image_data}
                })
            processed_messages.append({"role": "user", "content": content_list})

    elif image_paths and len(messages) == 3:
        image_data_list_context_image = [local_image_to_data_url(img_path) for img_path in image_paths[1]]
        image_data_list_this_clip = [local_image_to_data_url(img_path) for img_path in image_paths[0]]
        if processed_messages and processed_messages[1]["role"] == "user":
            middle_content = processed_messages[1]["content"]
            
            if not isinstance(middle_content, list):
                middle_content = [{"type": "text", "text": middle_content}]
                
            for image_data in image_data_list_context_image:
                middle_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_data}
                })

            
            last_content = processed_messages[-1]["content"]
            if not isinstance(last_content, list):
                last_content = [{"type": "text", "text": last_content}]
                
            for image_data in image_data_list_this_clip:
                last_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_data}
                })
            merged_content = middle_content + last_content

            processed_messages[1]["content"] = merged_content
            processed_messages.pop()


    payload = {
        "model": model_name,
        "messages": processed_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream
    }
    if return_json:
        payload["response_format"] = {"type": "json_object"}
    
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
    

    max_retries = 5
    retry_delay = 60

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**payload)
            message = response.choices[0].message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                return {
                    "role": message.role,
                    "content": message.content,
                    "tool_calls": message.tool_calls
                }
            else:
                return {
                    "role": message.role,
                    "content": message.content or "",
                    "tool_calls": None
                }
                
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"API failed, tried{max_retries} times: {str(e)}")
            
            print(f"failed {attempt + 1} times, retry in {retry_delay} seconds... error: {str(e)}")
            time.sleep(retry_delay)
            retry_delay *= 2
