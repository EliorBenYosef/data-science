from skimage import io, transform
from ...image_processing_kernels import scale

grey = scale(io.imread('cat.jpg', as_gray=True)).astype('uint8')
grey_resized = transform.resize(grey, (300, 300), mode='symmetric', preserve_range=True)

io.imsave('cat_grey.jpg', grey)
io.imsave('cat_grey_resized.jpg', grey_resized)
