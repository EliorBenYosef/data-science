from PIL import Image

# color = - pixels_initialization. by name ('white') or number (255)

# img_pil = Image.new(mode='1', size=(500, 500))  # mode='1' - B&W
img_pil = Image.new(mode='L', size=(500, 500), color='white')  # mode='L' - 256 Grayscale
# img_pil = Image.new(mode='RGB', size=(500, 500), color='white')
# img_pil = Image.new(mode='RGBA', size=(500, 500), color=(255, 255, 255, 255))

# img_pil = Image.open("./name.format")

img_pil_copy = img_pil.copy()
pixel_matrix_copy = img_pil_copy.load()

img_pil.convert(mode='1')  # B&W
img_pil.crop((100, 100, 100, 100))
img_pil.rotate(90)

k = 0

pixel_matrix = img_pil.load()
w, h = img_pil.size
for x in range(w):  # width_pixels
    for y in range(h):  # height_pixels
        # pixel_matrix_copy[x, y] = action  # [Collumn, Row], sets grey level
        pixel_matrix_copy[x, y] = (pixel_matrix[x, y] + k) % 256  # deflect by k
        pixel_matrix_copy[x, y] = 255 - pixel_matrix[x, y]  # Negate
        pixel_matrix_copy[x, y] = pixel_matrix[x, h - y - 1]  # upside down
        pixel_matrix_copy[x, y] = pixel_matrix[w - x - 1, y]  # mirror image

img_pil.show()
# Image.fromarray(pixel_matrix_copy).show()  # not right
# img_pil.save("./name", "format")
