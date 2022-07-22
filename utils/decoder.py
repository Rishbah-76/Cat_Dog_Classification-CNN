import base64

def decodeImage(imgstring,filename):
    image=base64.b64decode(imgstring)
    with open(filename,'wb') as f:
        f.write(image)
        f.close()

def encodeImageToBase64(imagepath):
    with open(imagepath,'rb') as f:
        return base64.b64encode(f.read())