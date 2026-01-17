# using iamge encoding and passing it to llm for the output
import os
import sys

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
    print(f"this is abs image path {img_path}")
    

def main():
    test_pipeline()
if __name__ =="__main__":
    main()

