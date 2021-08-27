from PIL import Image


def add(img_pil, k):
    pixel_matrix = img_pil.load()

    img_pil_copy = img_pil.copy()
    pixel_matrix_copy = img_pil_copy.load()

    w, h = img_pil.size
    for x in range(w):
        for y in range(h):
            pixel_matrix_copy[x, y] = (pixel_matrix[x, y] + k) % 256

    return img_pil_copy


def negate(img_pil):
    pixel_matrix = img_pil.load()

    img_pil_copy = img_pil.copy()
    pixel_matrix_copy = img_pil_copy.load()

    w, h = img_pil.size
    for x in range(w):
        for y in range(h):
            pixel_matrix_copy[x, y] = 255 - pixel_matrix[x, y]

    return img_pil_copy


def upside_down(img_pil):
    pixel_matrix = img_pil.load()

    img_pil_copy = img_pil.copy()
    pixel_matrix_copy = img_pil_copy.load()

    w, h = img_pil.size
    for x in range(w):
        for y in range(h):
            pixel_matrix_copy[x, y] = pixel_matrix[x, h - y - 1]

    return img_pil_copy


def mirror_image(img_pil):
    pixel_matrix = img_pil.load()

    img_pil_copy = img_pil.copy()
    pixel_matrix_copy = img_pil_copy.load()

    w, h = img_pil.size
    for x in range(w):
        for y in range(h):
            pixel_matrix_copy[x, y] = pixel_matrix[w - x - 1, y]
    return img_pil_copy


img_pil = Image.open('../../../datasets/per_field/cv/color_eiffel.jpg')
img_pil.show()

img_pil = img_pil.convert('L')
img_pil.show()

add(img_pil, 50).show()
negate(img_pil).show()
upside_down(img_pil).show()
mirror_image(img_pil).show()
