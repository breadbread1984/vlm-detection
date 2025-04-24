#!/usr/bin/python3

from typing import List
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from models import Qwen25VL7B_dashscope
from configs import dashscope_api_key

class BoundingBox(BaseModel):
  bbox_2d: List[int] = Field(description = '[x1, y1, x2, y2]'),
  label: str = Field(description = 'type of object')

class DetRes(BaseModel):
  targets: List[BoundingBox] = Field(description = 'a list of target bounding boxes.')

class Detection(object):
  def __init__(self,):
    self.qwen = Qwen25VL7B_dashscope(dashscope_api_key)
    self.parser = JsonOutputParser(pydantic_object = DetRes)
    self.instruction = self.parser.get_format_instructions()
  def detect(self, image, target_type = None):
    prompt = f"""Otline the position of each {target_type} object and output all the coordinates in JSON format."""
    result = self.qwen.inference(prompt, image)
    outputs = self.parser.parse(result)
    return outputs

if __name__ == "__main__":
  import cv2
  img = cv2.imread('test.jpg')
  det = Detection()
  results = det.detect(img)
  print(results)
  for result in results:
    x1, y1, x2, y2 = result
    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
  cv2.imwrite('output.png', img)

