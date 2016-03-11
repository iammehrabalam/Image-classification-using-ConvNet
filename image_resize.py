""".."""

import os
from PIL import Image


def resize_image(path, size=(64, 64)):
    "Resizing image and saving it with adding _resized in the name."
    try:
        img = Image.open(path)
        folder, image_name = img.filename.rsplit('/', 1)
        if not os.path.exists(os.path.join(folder, 'resized')):
            os.makedirs(os.path.join(folder, 'resized'))
        resized = img.resize(size)
        resized_image_name = os.path.join(folder, 'resized', image_name)
        resized.save(resized_image_name)
        print 'resized image path: ', resized_image_name
    except Exception as e:
        print e
