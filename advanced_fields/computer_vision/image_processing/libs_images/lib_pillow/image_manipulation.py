from PIL import Image


def add(image_var, k):
    pixel_matrix = image_var.load()

    image_var_copy = image_var.copy()
    pixel_matrix_copy = image_var_copy.load()

    w,h = image_var.size
    for x in range(w):
        for y in range(h):
            pixel_matrix_copy[x,y] = (pixel_matrix[x,y]+k)%256

    return image_var_copy


def negate(image_var):
    pixel_matrix = image_var.load()

    image_var_copy = image_var.copy()
    pixel_matrix_copy = image_var_copy.load()

    w,h = image_var.size
    for x in range(w):
        for y in range(h):
            pixel_matrix_copy[x,y] = 255 - pixel_matrix[x,y]

    return image_var_copy


def upside_down(image_var):
    pixel_matrix = image_var.load()

    image_var_copy = image_var.copy()
    pixel_matrix_copy = image_var_copy.load()

    w,h = image_var.size
    for x in range(w):
        for y in range(h):
            pixel_matrix_copy[x,y] = pixel_matrix[x,h-y-1]

    return image_var_copy


def mirror_image(image_var):
    pixel_matrix = image_var.load()

    image_var_copy = image_var.copy()
    pixel_matrix_copy = image_var_copy.load()

    w,h = image_var.size
    for x in range(w):
        for y in range(h):
            pixel_matrix_copy[x,y] = pixel_matrix[w-x-1,y]
    return image_var_copy


im = Image.open("practice_image.jpg")
im.show()

im = im.convert('L')
im.show()

add(im,50).show()
negate(im).show()
upside_down(im).show()
mirror_image(im).show()


