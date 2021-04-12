from PIL import Image 
import os,sys
import re
directory = "ESOBOT/"
for filename in os.listdir(directory):
    if filename.endswith(".png") and filename == "ghost.png":
        im = Image.open(directory + filename)
        name = re.sub('\.png$', '', filename).upper()
        print(name)
        im.show()
    else:
        continue
    break
# im = Image.open("ESOBOT/ghost.png")
# im.show()