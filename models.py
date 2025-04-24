#!/usr/bin/python3

from abc import ABC, abstractmethod
from os import environ
import base64
import numpy as np
import cv2

class VQA(ABC):
  def encode_img(self, image):
    if type(image) is str:
      # image's url is given
      return image
    elif type(image) is np.ndarray:
      success, encoded_image = cv2.imencode('.png', image)
      assert success, "failed to encode numpy to png image!"
      png_bytes = encoded_image.tobytes()
      png_b64 = base64.b64encode(png_bytes).decode('utf-8')
      return f"data:image/png;base64,{png_b64}"
    else:
      raise RuntimeError('image can only be given in url or np.ndarray format!')
  @abstractmethod
  def inference(self, question, image, system = None):
    pass

class Qwen25VL7B_dashscope(VQA):
  def __init__(self, api_key):
    from openai import OpenAI
    self.client = OpenAI(
      api_key = api_key,
      base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
  def inference(self, question, image, system_message = None):
    messages = list()
    if system_message is not None:
      messages.append({'role': 'system', 'content': system_message})
    messages.append({'role': 'user', 'content': [
      {'type': 'text', 'text': question},
      {'type': 'image_url', "image_url": {
        'url': self.encode_img(image)
      }}
    ]})
    response = self.client.chat.completions.create(
      model = 'qwen2.5-vl-7b-instruct',
      messages = messages,
    )
    return response.choices[0].message.content

class Qwen25VL7B_vllm(VQA):
  def __init__(self, vllm_host = "http://192.168.80.21:8000/v1"):
    from openai import OpenAI
    self.client = OpenAI(
      api_key = 'token-abc123',
      base_url = vllm_host
    )
  def inference(self, question, image, system_message = None):
    messages = list()
    if system_message is not None:
      messages.append({'role': 'system', 'content': system_message})
    messages.append({'role': 'user', 'content': [
      {'type': 'text', 'text': question},
      {'type': 'image_url', "image_url": {
        'url': self.encode_img(image)
      }}
    ]})
    response = self.client.chat.completions.create(
      model = 'Qwen/Qwen2.5-VL-7B-Instruct',
      messages = messages,
      extra_body={
        "repetition_penalty": 1.1,
        "temperature": 0.1,
        "top_p": 0.9,
        "top_k": 40,
        },
    )
    return response.choices[0].message.content

class Qwen25VL7B_tgi(VQA):
  def __init__(self, tgi_host = "http://192.168.80.21:8080"):
    from huggingface_hub import InferenceClient
    self.client = InferenceClient(tgi_host)
  def inference(self, question, image, system_message = None):
    messages = list()
    if system_message is not None:
      messages.append({'role': 'system', 'content': system_message})
    messages.append({'role': 'user', 'content': [
      {'type': 'text', 'text': question},
      {'type': 'image_url', "image_url": {
        'url': self.encode_img(image)
      }}
    ]})
    response = self.client.chat_completion(
      messages = messages
    )
    return response.choices[0].message.content

class Qwen25VL7B_transformers(VQA):
  def __init__(self, huggingface_api_key, device = 'cuda'):
    assert device in {'cuda', 'cpu'}
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_api_key
    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      'Qwen/Qwen2.5-VL-7B-Instruct', torch_dtype = "auto", device_map = "auto", low_cpu_mem_usage = True if device == 'cpu' else None
    ).to(device)
    self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
  def inference(self, question, image, system_message = None):
    from qwen_vl_utils import process_vision_info
    messages = list()
    if system_message is not None:
      messages.append({'role': 'system', 'content': system_message})
    messages.append({'role': 'user', 'content': [
      {'type': 'text', 'text': question},
      {'type': 'image', "image": self.encode_img(image)}
    ]})
    text = self.processor.apply_chat_template(
      messages, tokenize = False, add_generation_prompt = True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = self.processor(
      text = [text],
      images = image_inputs,
      videos = video_inputs,
      padding = True,
      return_tensors = "pt"
    )
    inputs = inputs.to(next(self.model.parameters())[0].device)
    generated_ids = self.model.generate(**inputs)
    generated_ids_trimmed = [
      out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = self.processor.batch_decode(
      generated_ids_trimmed, skip_special_tokens = True, clean_up_tokenization_spaces = False
    )
    return output_text[0]
