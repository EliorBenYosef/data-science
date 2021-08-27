from PIL import Image


def segment(img_pil_in, thrd):
    """ Binary segmentation of image im by threshold thrd """

    width, height = img_pil_in.size
    img_pil_out = Image.new('1', (width, height))  # '1' means black & white (no grays)
    in_mat = img_pil_in.load()
    out_mat = img_pil_out.load()

    for x in range(width):
        for y in range(height):
            if in_mat[x, y] >= thrd:
                out_mat[x, y] = 255  # white
            else:
                out_mat[x, y] = 0  # black

    return img_pil_out


img_pil = Image.open('../../../datasets/per_field/cv/color_eiffel.jpg')
img_pil = img_pil.convert('L')
img_pil.show()

for th in [50, 100, 150, 200, 250]:
    out_im = segment(img_pil, th)
    out_im.show()
    # out_im.save("./images/th/out"+str(th) + ".jpg")
