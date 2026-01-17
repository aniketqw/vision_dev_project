# using iamge encoding and passing it to llm for the output
# to run this file we should add -m flag while running python3 -m exp.helperFunc.image.test from main visio_dev_project
import os
import sys
from logger_setup import logger
# fix : getting the firector name and adding it python'search path 
dir_name =os.path.dirname(os.path.abspath(__file__))
# add dirname is the Python search path 
if dir_name not in sys.path:
    sys.path.append(dir_name)

from imageConvertor import encode_image
def test_pipeline():
    
    # __file__ current file name i.e test
    # abspath give absolute path of the current file i.e test
    # dirname gives the directory name in which it is saved i.e image
    img_path=os.path.join(dir_name,"image.jpeg")
    logger.info(f"Absolute image path located: {img_path}")
    
    # abse64_string

def main():
    test_pipeline()
if __name__ =="__main__":
    main()

