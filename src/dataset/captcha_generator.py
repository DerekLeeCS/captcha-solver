"""
Original file is located at
    https://colab.research.google.com/drive/1IJ04G1Nyo-bm9nf9QRWyAGtVUJigU37d

Install on colab
!pip install Pillow
!pip install captcha
"""

import os
import random
import string
# from google.colab import files
# import json
from PIL import Image
from PIL import ImageFilter
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname('/Captcha')), 'data')

DEFAULT_FONTS = ['thorndale-bold_bigfontsite.com.ttf']  # Font type of the Jail Tracker website captchas

__all__ = ['ImageCaptcha']

table = []
for i in range(256):
    table.append(int(i * 1.97))


def random_color(start, end, opacity=None):
    red = random.randint(start, end)
    green = random.randint(start, end)
    blue = random.randint(start, end)
    if opacity is None:
        return red, green, blue
    return red, green, blue, opacity


class _Captcha(object):
    def generate(self, chars, format='png'):
        """Generate an Image Captcha of the given characters.
        :param chars: text to be generated.
        :param format: image file format
        """
        im = self.generate_image(chars)
        out = BytesIO()
        im.save(out, format=format)
        out.seek(0)
        return out

    def write(self, chars, output, format='png'):
        """Generate and write an image CAPTCHA data to the output.
        :param chars: text to be generated.
        :param output: output destination.
        :param format: image file format
        """
        im = self.generate_image(chars)
        return im.save(output, format=format)


# Based on: https://github.com/lepture/captcha/blob/master/captcha/image.py
# We do not generate noise dots
class ImageCaptcha(_Captcha):
    """Create an image CAPTCHA.
    Many of the codes are borrowed from wheezy.captcha, with a modification
    for memory and developer friendly.
    ImageCaptcha has one built-in font, DroidSansMono, which is licensed under
    Apache License 2. You should always use your own fonts::
        captcha = ImageCaptcha(fonts=['/path/to/A.ttf', '/path/to/B.ttf'])
    You can put as many fonts as you like. But be aware of your memory, all of
    the fonts are loaded into your memory, so keep them a lot, but not too
    many.
    :param width: The width of the CAPTCHA image.
    :param height: The height of the CAPTCHA image.
    :param fonts: Fonts to be used to generate CAPTCHA images.
    :param font_sizes: Random choose a font size from this parameters.
    """

    def __init__(self, width=128, height=32, fonts=None, font_sizes=None):
        self._width = width
        self._height = height
        self._fonts = fonts or DEFAULT_FONTS
        self._font_sizes = font_sizes or (20, 24, 28)
        self._truefonts = []

    @property
    def truefonts(self):
        if self._truefonts:
            return self._truefonts
        self._truefonts = tuple([
            truetype(n, s)
            for n in self._fonts
            for s in self._font_sizes
        ])
        return self._truefonts

    @staticmethod
    def create_noise_horizontal_line(image):
        """
        Generate 1-2 horizontal lines that can be slanted
        """
        draw = Draw(image)
        w, h = image.size
        width = random.randint(1, 2)
        line_count = random.randint(1, 2)

        for i in range(line_count):
            x1 = random.randint(0, w)
            x2 = random.randint(0, w)
            y1 = random.randint(0, h)
            y2 = random.randint(0, h)
            # points = [x1, y2, x2, y1]
            draw.line(((x1, y1), (x2, y2)), fill=random_color(10, 200, random.randint(220, 255)), width=width)
        return image

    @staticmethod
    def create_noise_vertical_line(image, width=1, number=random.randint(0, 3)):
        """
        Generate 0-3 vertical lines that can be tilted
        """
        draw = Draw(image)
        w, h = image.size

        while number:
            x1 = random.randint(0, w)
            if random.random() > 0.5:
                x2 = x1 - random.randint(0, 10)
            else:
                x2 = x1 + random.randint(0, 10)
            y1 = random.randint(0, h)
            y2 = random.randint(0, h)
            draw.line(((x1, y1), (x2, y2)), fill=random_color(10, 200, random.randint(220, 255)), width=width)
            number -= 1
        return image

    def create_captcha_image(self, chars, background):
        """Create the CAPTCHA image itself.
        :param chars: text to be generated.
        :param background: color of the background.
        """
        image = Image.new('RGB', (self._width, self._height), background)
        draw = Draw(image)

        def _draw_character(c, font):
            w, h = draw.textsize(c, font=font)

            dx = 5
            dy = 5
            im = Image.new('RGB', (w + dx, h + dy))
            Draw(im).text((dx, dy), c, font=font, fill=random_color(10, 200, 30))
            return im

        images = []
        font_arr = []
        for c in chars:
            if random.random() > 0.15:
                k = random.randint(0, 2)
                font = self.truefonts[k]
                images.append(_draw_character("   ", font))
                font_arr.append(k)

            k = random.randint(0, 2)
            font = self.truefonts[k]
            images.append(_draw_character(c, font))
            font_arr.append(k)

        text_width = sum([im.size[0] for im in images])

        width = max(text_width, self._width)
        image = image.resize((width, self._height))

        average = int(text_width / len(chars))
        rand = int(.15 * average)
        offset = int(average * random.uniform(.05, .1))

        k = 0
        for im in images:
            w, h = im.size
            mask = im.convert('L').point(table)
            # im.putalpha(200)
            image.paste(im, (offset, -random.randint(0, int(self._font_sizes[font_arr[k]] / 4))), mask)
            # image.paste(im, (offset, int((self._height - h - int(h/4)))), mask)
            offset = offset + w + random.randint(-rand, 0)
            k += 1

        if width > self._width:
            image = image.resize((self._width, self._height))

        return image

    def generate_image(self, chars):
        """Generate the image of the given characters.
        :param chars: text to be generated.
        """
        background = random_color(100, 255, 30)
        im = self.create_captcha_image(chars, background)
        self.create_noise_vertical_line(im)
        self.create_noise_horizontal_line(im)
        im = im.filter(ImageFilter.SMOOTH)
        im = im.resize((self._width, self._height))
        return im


# Captcha characters are lowercase letters, uppercase letters and numbers
# Each captcha label has four characters
CAPTCHA_LETTERS = string.ascii_letters + string.digits
NUM_CHARACTERS = 4


# File names are formatted as "label_<unique identifier>.png" since it is possible to have the same captcha label but different image
def generate_captchas(filepath: str, num_captchas: int) -> None:
    captcha_generator = ImageCaptcha()
    for i in range(num_captchas):
        captcha = ''.join(random.choice(CAPTCHA_LETTERS) for _ in range(NUM_CHARACTERS))
        captcha_img = captcha_generator.generate_image(captcha)
        captcha_img.save(f'{filepath}/{captcha}_{str(i).rjust(5, "0")}.png')


# Testing for 1 image
cap = ImageCaptcha()
tmp = ''.join(random.choice(CAPTCHA_LETTERS) for _ in range(4))
cap_img = cap.generate_image(tmp)
cap_img.save('test' + '.png')

# Testing for 100 images
"""
parent_dir = 'Captcha2/'
os.makedirs(parent_dir, exist_ok=True)

generate_captchas(parent_dir, 100)

!zip -r /content/Captcha2.zip /content/Captcha2
files.download('Captcha2.zip')
"""

# Create directories for training, validating and testing data
parent_dir = 'Captcha/'
train_path = os.path.join(parent_dir, 'train')
valid_path = os.path.join(parent_dir, 'valid')
test_path = os.path.join(parent_dir, 'test')
os.makedirs(train_path, exist_ok=True)
os.makedirs(valid_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Generate 40,000 captchas for training
print("Generating training captchas")
generate_captchas(train_path, 40000)

# Generate 5,000 captchas for validating
print("Generating validation captchas")
generate_captchas(valid_path, 5000)

# Generate 5,000 captchas for testing
print("Generating testing captchas")
generate_captchas(test_path, 5000)

""" Export on colab
!zip -r /content/Captcha.zip /content/Captcha

files.download('Captcha.zip')
"""
