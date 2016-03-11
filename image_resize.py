""".."""

import os
import sys
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

if __name__ == '__main__':
    """
    How to run.
        python image_resize.py  <imagefilename or imagefoldername>
    Examples:
        python image_resize.py   /Users/mehrabalam/test_images/0028
        python image_resize.py   /Users/mehrabalam/test_images/0028/abc.jpg
    """

    sys.argv.pop(0)
    if os.path.isdir(sys.argv[0]):
        images = [os.path.join(sys.argv[0], i)for i in os.listdir(sys.argv[0])]
    else:
        images = [sys.argv[0]]
    print images
    for image in images:
        if os.path.isfile(image):
            resize_image(image)
