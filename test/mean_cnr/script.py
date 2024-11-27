import os
from PIL import Image
import numpy as np

MAX_CNR_VALUE = 256
MARGIN = 0
INPUT_DIR = 'in'
OUT_FILE = 'out.txt'

if os.path.exists(OUT_FILE):
    os.remove(OUT_FILE)

for filename in os.listdir(INPUT_DIR):
    f = os.path.join(INPUT_DIR, filename)
    if os.path.isfile(f):
        with Image.open(f) as img:
            width, height = img.size
            img = img.convert("L").crop((MARGIN, MARGIN, width - MARGIN, height - MARGIN))

            #img.show()

            img_array = np.array(img, dtype=np.uint8)

            mean = (np.mean(img_array) / 2**8) * MAX_CNR_VALUE

            out = open(OUT_FILE, "a")
            out.write(f"{filename} \t {mean}\n")
            out.close()

            print(mean)