import os
import sys
from logger_setup import logger
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Fix: Ensure local project imports work when running with -m
dir_name = os.path.dirname(os.path.abspath(__file__))
if dir_name not in sys.path:
    sys.path.append(dir_name)

def llm_text_call(user_query):
    logger.info(f"Processing text query: {user_query}")
    
    # Connect to your existing local 3090 vLLM server
    llm = ChatOpenAI(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        openai_api_key="local-3090",
        openai_api_base="http://localhost:8000/v1",
        temperature=0.7,
        max_tokens=512
    )

    # Standard Text-only Prompt
    text_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a VisionDev AI expert specializing in Image Segmentation and model optimization."),
        ("user", "{query}")
    ])

    # Chain definition
    chain = text_prompt | llm | StrOutputParser()

    logger.info("Invoking text generation chain...")
    output = chain.invoke({"query": user_query})
    
    print("-" * 30)
    logger.info(f"🤖 AI Response:\n{output}")
    print("-" * 30)

def main():
    # Example technical query about your project
    query = "Explain how Focal Loss helps in Image Segmentation Boundary Leak failure modes."
    llm_text_call(query)
    
if __name__ == "__main__":
    main()