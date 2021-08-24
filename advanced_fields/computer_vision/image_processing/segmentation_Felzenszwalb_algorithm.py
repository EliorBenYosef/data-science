from skimage import io, segmentation
from matplotlib import pyplot as plt
from image_processing_kernels import scale

img = scale(io.imread('../../../datasets/per_field/cv/manu-2013.jpg', as_gray=True)).astype('uint8')  # grayscale for simplicity

segment_mask1 = segmentation.felzenszwalb(img, scale=100)
segment_mask2 = segmentation.felzenszwalb(img, scale=1000)

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(segment_mask1); ax1.set_xlabel("k=100")
ax2.imshow(segment_mask2); ax2.set_xlabel("k=1000")
fig.suptitle("Felsenszwalb's efficient graph based image segmentation")
plt.tight_layout()
plt.show()
