# using iamge encoding and passing it to llm for the output
# to run this file we should add -m flag while running python3 -m exp.helperFunc.image.test from main visio_dev_project
import os
import sys
from logger_setup import logger
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# fix : getting the firector name and adding it python'search path 
dir_name =os.path.dirname(os.path.abspath(__file__))
# add dirname is the Python search path 
if dir_name not in sys.path:
    sys.path.append(dir_name)

from imageConvertor import encode_image


def llm_call(img_path):
    logger.info(f"Absolute image path located: {img_path}")
    imgb64=encode_image(img_path)
    logger.info(f"base 64 string : {imgb64[:10]}")
    
    llm =ChatOpenAI(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        openai_api_key="local-3090",
        openai_api_base="http://localhost:8000/v1"
    )

    vision_prompt = ChatPromptTemplate.from_messages([
        ("user", [
            {
                "type": "text", 
                "text": "Analyze this segmentation error. Failure Mode: {failure_mode}. Recipe: {rag_recipe}"
            },
            {
                "type": "image_url", 
                "image_url": {"url": "data:image/jpeg;base64,{image_data}"}
            }
        ])
    ])

    # The Pipe (|) sequence
    # 1. Takes the input dictionary
    # 2. Formats the prompt
    # 3. Sends it to the 3090 (vLLM)
    # 4. Cleans the output into a simple string
    chain = vision_prompt | llm | StrOutputParser()

    # 5. EXECUTE: Invoke the chain with required inputs
    logger.info("Invoking vision chain...")
    output = chain.invoke({
        "failure_mode": "Image Segmentation Boundary Leak",
        "rag_recipe": "Use Focal Loss or Atrous Spatial Pyramid Pooling to improve edge detection.",
        "image_data": imgb64
    })


def main():
    img_path=os.path.join(dir_name,"image.jpeg")
    llm_call(img_path)
    test_pipeline()
if __name__ =="__main__":
    main()

