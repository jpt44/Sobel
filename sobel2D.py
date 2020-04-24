from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def sobel2D(img):
    """
    Function convolves an image with 2-D version of Sobel kernel. Pixels at borders set to black

    Input: 2-D array of an image with 1 channel
    Output 2-D convolved array
    """
    # Sobel kernel for horizontal gradient
    sobel2Dx = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ], dtype=np.int32)

    # Sobel kernel for vertical gradient
    sobel2Dy = sobel2Dx.T

    # Flip sobel kernel along x and y axes for convolution
    sobel2Dx = np.flip(sobel2Dx, axis=[0, 1])

    # Flip sobel kernel along x and y axes for convolution
    sobel2Dy = np.flip(sobel2Dy, axis=[0, 1])

    # Arrays to store convolved image
    nx = np.ndarray(shape=img.shape, dtype=np.int32)
    ny = np.ndarray(shape=img.shape, dtype=np.int32)

    for r in range(img.shape[0]):
        ith = r - 1
        for c in range(img.shape[1]):
            jth = c - 1
            # If kernel exceeds image size, make pixel 0 at that location
            if ith < 0 or jth < 0:
                nx[r, c] = 0
                ny[r, c] = 0
            elif ith + 3 > img.shape[0] or jth + 3 > img.shape[1]:
                nx[r, c] = 0
                ny[r, c] = 0
            else:
                # Convolution operation
                nx[r, c] = round((img[ith:ith + 3, jth:jth + 3] * sobel2Dx).sum(), 0)
                ny[r, c] = round((img[ith:ith + 3, jth:jth + 3] * sobel2Dy).sum(), 0)

    return (nx, ny)


if __name__ == "__main__":
    # Image path + name
    pth1 = ""

    # import image
    inputImg = inputImg1 = Image.open(pth1)
    img = np.array(inputImg1.convert("L"), dtype=np.uint8) #Convert image to grayscale and numpy array

    # Convolve image with 2D kernel
    gx2D, gy2D = sobel2D(img)

    #Calculate gradient magnitudes
    gx2Dmag = np.sqrt(gx2D * gx2D)
    gy2Dmag = np.sqrt(gy2D * gy2D)

    #Visualize results
    fig1=plt.figure(figsize=(1,3))
    ax1 = fig1.add_subplot(1, 3, 1)
    ax1.axis("off")
    ax1.set_title("Original")
    plt.imshow(inputImg)

    ax2 = fig1.add_subplot(1, 3, 2)
    ax2.axis("off")
    ax2.set_title("Horizontal Gradient 2-D Kernel")
    plt.imshow(gx2D,cmap="gray")

    ax3 = fig1.add_subplot(1, 3, 3)
    ax3.axis("off")
    ax3.set_title("Vertical Gradient 2-D Kernel")
    plt.imshow(gy2D,cmap="gray")

    fig1 = plt.figure(figsize=(1, 3))
    ax1 = fig1.add_subplot(1, 3, 1)
    ax1.axis("off")
    ax1.set_title("Original")
    plt.imshow(inputImg)

    ax2 = fig1.add_subplot(1, 3, 2)
    ax2.axis("off")
    ax2.set_title("Horizontal Magnitude 2-D Kernel")
    plt.imshow(gx2Dmag, cmap="gray")

    ax3 = fig1.add_subplot(1, 3, 3)
    ax3.axis("off")
    ax3.set_title("Vertical Magnitude 2-D Kernel")
    plt.imshow(gy2Dmag, cmap="gray")

    plt.show()





