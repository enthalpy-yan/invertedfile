from PIL import Image
import os

def crop_dir(top_dir):
    log = open(os.path.join(top_dir, "log.txt"), 'a')
    for dir_path,subpaths,files in os.walk(top_dir,False):
        for file in files:
            img = os.path.join(dir_path, file)
            crop_img(img, log)


def crop_img(img, log):

    try:
        original = Image.open(img)
        f, ext = os.path.splitext(img)
        width, height = original.size

        if width > 300 or height > 300:
            start_x = (width - 300) / 2
            start_y = (height - 300) / 2
            box = (start_x, start_y, start_x + 300, start_y + 300)
            output = original.crop(box)
            output.save(f + ".jpg")
            
            log.write('Success to crop: ' + f + '\n')
        else:
            
            os.remove(img)
            log.write("Fail to crop: " + f + '\n')
    except:
        
        log.write("Unable to load image: " + img + '\n')
