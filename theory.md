imp points :
* Images are stored as NumPy arrays (matrices of pixel values).
* `cv2.imread()` is used to read images.
* `img.shape` gives image dimensions and channels.
* `img.dtype` gives the data type of pixel values.
* For grayscale images: 0 = black and 255 = white.
* OpenCV stores color images in BGR(Blue,Green,Red) format.
* `cv2.split()` separates B, G and R channels.
* `cv2.merge()` combines channels back into a color image.
* `cv2.cvtColor()` is used to convert between color spaces (BGR, RGB, HSV).
* HSV stands for Hue, Saturation and Value.
* `cv2.imwrite()` is used to save images.
* `cv2.imshow()`, `cv2.waitKey()` and `cv2.destroyAllWindows()` are used to display images using OpenCV.
