# for using imaeg we have to encode the image in base64 string
import base64
import os
def encode_image(image_path):
    with open(image_path,"rb") as image_file:#rb format is used to open image , audio and other non text things
        b64_bytes=base64.b64encode(image_file.read())
        b64_string=b64_bytes.decode('utf-8')
        return b64_string
def main():
    path='/home/pratik2/vision_dev_project/exp/helperFunc/image/image.jpeg'
    print(f"File exist:{ os.path.exists(path)}")
    print(encode_image(path))
if __name__ =="__main__":
    main()

# working