from PIL import Image
import cv2
import numpy as np
from io import BytesIO
import base64

#------------------------------------------------

def img2b4(img: Image, format = 'PNG'):
    im_file = BytesIO()
    img.save(im_file, format=format)
    im_bytes = im_file.getvalue()  
    im_b64 = base64.b64encode(im_bytes).decode('utf-8')
    return f'data:image/{format.lower()};base64,{im_b64}'   

def splitHeightTo2(img: Image):
    width, height = img.size
    box_count = 2
    box_height = height / box_count
    print(box_height)
    images = []
    for i in range(box_count):
        box = (0, box_height * i, width, box_height * (i + 1))
        a = img.crop(box)
        images.append(img2b4(a))
    return images

def splitImageTo9(img: Image):
    width, height = img.size
    box_width = width / 3
    box_height = height / 3
    images = []
    for i in range(3):
        for j in range(3):
            box = (box_width * j, box_height * i, box_width * (j + 1), box_height * (i + 1))
            a = img.crop(box)
            images.append(img2b4(a))
    return images

def cut(img: Image, format = 'PNG'):
    img = img.convert('RGBA')

    width, height = img.size
    pixels = img.getcolors(width * height)
    most_frequent_pixel = pixels[0]

    for count, colour in pixels:
        if count > most_frequent_pixel[0]:
            most_frequent_pixel = (count, colour)

    for x in range(width):
        for y in range(height):
            pixel = img.getpixel((x, y))
            if abs(pixel[0] - most_frequent_pixel[1][0]) < 10 and abs(pixel[1] - most_frequent_pixel[1][1]) < 10 and abs(pixel[2] - most_frequent_pixel[1][2]) < 10:
                img.putpixel((x, y), (255, 255, 255, 0))
                
    return img


#-----------------
# https://github.com/webaverse-studios/spritesheet-api
#-----------------

def getLeftOrRight(img: Image):
    img = img.convert('RGBA')

    width, height = img.size
    pixels = img.getcolors(width * height)
    most_frequent_pixel = pixels[0]

    for count, colour in pixels:
        if count > most_frequent_pixel[0]:
            most_frequent_pixel = (count, colour)

    for x in range(width):
        for y in range(height):
            pixel = img.getpixel((x, y))
            if abs(pixel[0] - most_frequent_pixel[1][0]) < 10 and abs(pixel[1] - most_frequent_pixel[1][1]) < 10 and abs(pixel[2] - most_frequent_pixel[1][2]) < 10:
                img.putpixel((x, y), (255, 255, 255, 0))
    
    
    img = img.convert('RGBA')
    width, height = img.size
    cel_width = width / 4
    cel_height = height / 4
    cell = img.crop((0, 0, cel_width, cel_height))
    pixel = cell.getpixel((cel_width / 2, cel_height / 2))
    alpha = pixel[3]
    if alpha == 0:
       return False #right
    else: 
        return True #left
    

def postprocessImg(img1: Image):
    isLeft = getLeftOrRight(img1)
    if not isLeft:
        width = img1.size[0]
        height = img1.size[1]
        qHeight = height / 4

        def cutImgTo4Rows(img: Image):
            imgs = []
            for i in range(4):
                imgs.append(img.crop((0, i * qHeight, width, (i + 1) * qHeight)))
            return imgs

        imgs1 = cutImgTo4Rows(img1)

        imgs1[1], imgs1[2] = imgs1[2], imgs1[1]

        img1 = Image.new("RGBA", (width, height))
        for i in range(4):
            img1.paste(imgs1[i], (0, int(i * qHeight)))

        return img1
    return img1

def crop(img):
    img = img.convert('RGBA')
    width, height = img.size
    cel_width = width / 4
    cel_height = height / 4
    cell = img.crop((0, 0, cel_width, cel_height))
    pixel = cell.getpixel((cel_width / 2, cel_height / 2))
    alpha = pixel[3]
    if alpha == 0:
        img = img.crop((cel_width, 0, width, height))
    else: 
        img = img.crop((0, 0, width - cel_width, height))
    return img

def cleanImage(img: Image):
   # Convert the image to RGBA format
    img = img.convert('RGB')
    im_array = np.array(img)

    # Get the shape of the image
    height, width, channels = im_array.shape

    # Set the color thresholds for the background colors
    color_threshold_r = 10
    color_threshold_g = 10
    color_threshold_b = 10

    # Loop through the pixels in the image
    for y in range(height):
        for x in range(width):
            # Get the RGB values of the pixel
            r, g, b = im_array[y, x]

            # Check if the pixel is a shade of green
            if abs(r - 0) < color_threshold_r and abs(b - 0) < color_threshold_b:
                # Check the surrounding pixels to see if the current pixel is surrounded by pixels with the same color
                surrounding_pixels = [(y-1, x-1), (y-1, x), (y-1, x+1), (y, x-1), (y, x+1), (y+1, x-1), (y+1, x), (y+1, x+1)]
                same_color = True
                for pixel in surrounding_pixels:
                    y_pos, x_pos = pixel
                    if y_pos < 0 or y_pos >= height or x_pos < 0 or x_pos >= width:
                        continue
                    r_surrounding, g_surrounding, b_surrounding = im_array[y_pos, x_pos]
                    if abs(r - r_surrounding) > color_threshold_r or abs(g - g_surrounding) > color_threshold_g or abs(b - b_surrounding) > color_threshold_b:
                        same_color = False
                        break
                if same_color:
                    # Set the alpha value of the pixel to 0 to make it transparent
                    im_array[y, x] = 0

    # Create a new image from the modified NumPy array
    transparent_im = Image.fromarray(im_array)
    
      # Create a new image from the modified NumPy array
    transparent_im = Image.fromarray(im_array)

    #remove white background
    transparent_im = transparent_im.convert("RGBA")
    datas = transparent_im.getdata()

    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    transparent_im.putdata(newData)

    # Save the transparent image
    return transparent_im


def cleanImage_wrapper(img):
  img = postprocessImg(img)
  img = img.convert('RGBA')

  width, height = img.size
  pixels = img.getcolors(width * height)
  most_frequent_pixel = pixels[0]

  for count, colour in pixels:
      if count > most_frequent_pixel[0]:
          most_frequent_pixel = (count, colour)

  for x in range(width):
      for y in range(height):
          pixel = img.getpixel((x, y))
          if abs(pixel[0] - most_frequent_pixel[1][0]) < 10 and abs(pixel[1] - most_frequent_pixel[1][1]) < 10 and abs(pixel[2] - most_frequent_pixel[1][2]) < 10:
              img.putpixel((x, y), (255, 255, 255, 0))
  
  img = crop(img)   
  img = cleanImage(img)
  return img