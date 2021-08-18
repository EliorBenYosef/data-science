from PIL import Image


def segment(image_var, thrd):
    """ Binary segmentation of image im by threshold thrd """

    width, height = image_var.size
    out = Image.new('1',(width, height))  # '1' means black & white (no grays)
    mat = image_var.load()
    out_mat = out.load()

    for x in range(width):
        for y in range(height):
            if mat[x, y] >= thrd:
                out_mat[x, y] = 255  # white
            else:
                out_mat[x, y] = 0  # black

    return out


im = Image.open("practice_image.jpg")
im = im.convert('L')
im.show()

for th in [50,100,150,200,250]:
    out_im = segment(im, th)
    out_im.show()
    # out_im.save("./images/th/out"+str(th) + ".jpg")
