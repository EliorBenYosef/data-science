from PIL import Image

# color= - pixels_initialization. by name ('white') or number (255)

image_var = Image.new(mode='L', size=(500, 500), color='white')  # mode='L' - 256 Gray Scale
# image_var = Image.open("./name.format")

image_var_copy = image_var.copy()
pixel_matrix_copy = image_var_copy.load()

image_var.convert(mode='1')  # B&W
image_var.crop((100, 100, 100, 100))
image_var.rotate(90)

k = 0

pixel_matrix = image_var.load()
w,h = image_var.size
for x in range(w):  # width_pixels
    for y in range(h):  # height_pixels
        # pixel_matrix_copy[x, y] = action  # [Collumn, Row], sets grey level
        pixel_matrix_copy[x, y] = (pixel_matrix[x, y] + k) % 256  # deflect by k
        pixel_matrix_copy[x, y] = 255 - pixel_matrix[x, y]  # Negate
        pixel_matrix_copy[x, y] = pixel_matrix[x, h - y - 1]  # upside down
        pixel_matrix_copy[x, y] = pixel_matrix[w - x - 1, y]  # mirror image

image_var.show()
# image_var.save("./name", "format")
