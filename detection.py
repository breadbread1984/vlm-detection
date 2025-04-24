#!/usr/bin/python3

from typing import List
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from models import Qwen25VL7B_dashscope
from configs import dashscope_api_key

class Target(BaseModel):
  x: int = Field(description = 'the upper left x coordinate of a target bouding box.')
  y: int = Field(description = 'the upper left y coordinate of a target bounding box.')
  width: int = Field(description = 'the width of a target bounding box.')
  height: int = Field(description = 'the height of a target bounding box.')

class DetRes(BaseModel):
  targets: List[Target] = Field(description = 'a list of Target object which represents all detected targets.')

class Detection(object):
  def __init__(self,):
    self.qwen = Qwen25VL7B_dashscope(dashscope_api_key)
    self.parser = JsonOutputParser(pydantic_object = DetRes)
    self.instruction = self.parser.get_format_instructions()
  def detect(self, image, object_type = 'people'):
    prompt = f"""please detect objects of type {object_type} and output in the following format:

{self.instruction}"""
    result = self.qwen.inference(prompt, image)
    outputs = self.parser.parse(result)
    return outputs

if __name__ == "__main__":
  import cv2
  img = cv2.imread('test.jpg')
  det = Detection()
  results = det.detect(img)
  import pdb; pdb.set_trace()

