# using iamge encoding and passing it to llm for the output
import os
from imageConvertor import encode_image

def test_pipeline():
    dir_name =os.path.dirname(os.path.abspath(__file__))
    # __file__ current file name i.e test
    # abspath give absolute path of the current file i.e test
    # dirname gives the directory name in which it is saved i.e image
    img_path=os.path.join(dir_name,"image.jpeg")
    print(f"this is abs image path {image_path}")

    