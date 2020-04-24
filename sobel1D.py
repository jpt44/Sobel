from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def sobel1D(img):
    """
    Function convolves an image with 1-D version of Sobel kernel. Pixels at borders set to black.
    
    Input: 2-D array of an image with 1 channel
    Output 2-D convolved array
    """
    #Sobel kernel for horizontal gradient
    sobel2Dx = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ],dtype=np.int32)

    #Sobel kernel for vertical gradient
    sobel2Dy = sobel2Dx.T

    #Horizontal 1-D kernels separated
    sobelx = np.flip(sobel2Dx[:, 0].reshape(3,1),axis=0)  #[1,2,1]
    sobelx2 = np.flip(sobel2Dx[0, :].reshape(1, 3),axis=1) # [-1,0,1]

    #Vertical 1-D kernels separated
    sobely = np.flip(sobel2Dy[:, 0].reshape(3, 1), axis=0)  # [-1,0,1]
    sobely2 = np.flip(sobel2Dy[0, :].reshape(1, 3),axis=1) # [1,2,1]

    # Intermediate Array to store convolved image
    nx_inter = np.ndarray(shape=img.shape, dtype=np.int32)
    ny_inter = np.ndarray(shape=img.shape, dtype=np.int32)

    #Horizontal convolution
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            jth=c-1
            #If kernel exceeds image border, set pixel to 0 at that location
            if jth<0:
                nx_inter[r, c] = 0
                ny_inter[r, c] = 0
            elif jth+3>img.shape[1]:
                nx_inter[r, c] = 0
                ny_inter[r, c] = 0
            else:
                nx_inter[r, c] = round(img[r, jth:jth + 3].dot(sobelx2.T)[0], 0)
                ny_inter[r, c] = round(img[r, jth:jth + 3].dot(sobely2.T)[0], 0)

    #Array to solve final version of convolved image
    nx = nx_inter.copy()
    ny = ny_inter.copy()

    #Vertical convolution
    for r in range(img.shape[0]):
        ith = r - 1
        for c in range(img.shape[1]):
            if ith<0:
                nx[r, c] = 0
                ny[r, c] = 0
            elif ith + 3 > img.shape[0]:
                nx[r, c] = 0
                ny[r, c] = 0
            else:
                nx[r, c] = round((nx_inter[ith:ith + 3, c].T).dot(sobelx)[0], 0)
                ny[r, c] = round((ny_inter[ith:ith + 3, c].T).dot(sobely)[0], 0)

    return (nx,ny)


if __name__ == "__main__":

    #Image path + name
    pth1=""

    #import image
    inputImg=inputImg1=Image.open(pth1)
    img=np.array(inputImg1.convert("L"),dtype=np.uint8) #Convert image to grayscale and numpy array

    #Convolve image with 1D kernel
    gx1D,gy1D=sobel1D(img)

    #Calculate gradient magnitudes
    gx1Dmag = np.sqrt(gx1D * gx1D)
    gy1Dmag = np.sqrt(gy1D * gy1D)

    #Visualize results
    fig1=plt.figure(figsize=(1,3))
    ax1 = fig1.add_subplot(1, 3, 1)
    ax1.axis("off")
    ax1.set_title("Original")
    plt.imshow(inputImg)

    ax2 = fig1.add_subplot(1, 3, 2)
    ax2.axis("off")
    ax2.set_title("Horizontal Gradient 1-D Kernel")
    plt.imshow(gx1D,cmap="gray")

    ax3 = fig1.add_subplot(1, 3, 3)
    ax3.axis("off")
    ax3.set_title("Vertical Gradient 1-D Kernel")
    plt.imshow(gy1D,cmap="gray")

    fig1 = plt.figure(figsize=(1, 3))
    ax1 = fig1.add_subplot(1, 3, 1)
    ax1.axis("off")
    ax1.set_title("Original")
    plt.imshow(inputImg)

    ax2 = fig1.add_subplot(1, 3, 2)
    ax2.axis("off")
    ax2.set_title("Horizontal Magnitude 1-D Kernel")
    plt.imshow(gx1Dmag, cmap="gray")

    ax3 = fig1.add_subplot(1, 3, 3)
    ax3.axis("off")
    ax3.set_title("Vertical Magnitude 1-D Kernel")
    plt.imshow(gy1Dmag, cmap="gray")

    plt.show()



