from PIL import Image


def segment(img_pil_in, threshold):
    """
    Binary segmentation of image im by threshold
    """
    width, height = img_pil_in.size
    img_pil_out = Image.new('1', (width, height))  # '1' means black & white (no grays)
    in_mat = img_pil_in.load()
    out_mat = img_pil_out.load()

    for x in range(width):
        for y in range(height):
            if in_mat[x, y] >= threshold:
                out_mat[x, y] = 255  # white
            else:
                out_mat[x, y] = 0  # black

    return img_pil_out


img_pil = Image.open('../../../../datasets/per_field/cv/color_eiffel.jpg')
img_pil = img_pil.convert('L')
img_pil.show()

for threshold in range(75, 175, 25):
    out_im = segment(img_pil, threshold)
    out_im.save(f'results/binary_segmentation_{threshold}.png')
    out_im.show()
