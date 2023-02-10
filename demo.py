import cv2
import numpy as np
# hist = cv2.calcHist([image], [0], None, [256], [0, 256])

ll_img = cv2.imread("Test/LIME/1.bmp")
u = 0.5

(B,G,R) = cv2.split(ll_img)
img_gray = cv2.cvtColor(ll_img, cv2.COLOR_BGR2GRAY)

img_gray_bilateral = cv2.bilateralFilter(img_gray, 5, 21, 21)
img_gray_bilateral[img_gray_bilateral==0]=1
img_d = img_gray/img_gray_bilateral
img_eq_bg = cv2.createCLAHE().apply(img_gray_bilateral)

shift = img_eq_bg.astype(np.int8) - img_gray_bilateral.astype(np.int8)
scale = shift/img_gray_bilateral
scale[img_gray_bilateral<=0.] = 0.

alpha = (1-u*shift)

R_ = R*(1+u*scale) + alpha
G_ = G*(1+u*scale) + alpha
B_ = B*(1+u*scale) + alpha

img_max = cv2.max(cv2.max(R,G), B)
img_max = cv2.merge((img_max,img_max,img_max))
# img_max[img_max>255] = 255
img_max[img_max<=0] = 1

img_min = cv2.min(cv2.min(R,G), B)
img_min = cv2.merge((img_min,img_min,img_min))
# img_min[img_min>255] = 255
# img_min[img_min<0] = 0

img_rgb_nl = cv2.merge((B_, G_, R_))
img_rgb_nl[img_rgb_nl>255] = 255
img_rgb_nl[img_rgb_nl<0] = 0

img_rgb_nl[img_rgb_nl>255.] = (255./img_max * img_rgb_nl)[img_rgb_nl>255.]
img_rgb_nl[img_rgb_nl<0.] = (img_max/(img_max-img_min)*(img_rgb_nl - img_min))[img_rgb_nl<0.]


img_ycbcr = cv2.cvtColor(img_rgb_nl.astype(np.uint8), cv2.COLOR_BGR2YCR_CB)
(Y, CR, CB) = cv2.split(img_ycbcr)

output = (img_d - 1) * img_gray
output = cv2.merge((output, output, output)) + img_rgb_nl
output[output>255] = 255
output[output<0] = 0
output = output.astype(np.uint8)


cv2.imshow('Low-Light Image', ll_img)
cv2.imshow('F_gray_bilateral', img_gray_bilateral)
cv2.imshow('F_d', img_d)
cv2.imshow('scale', scale)
cv2.imshow('output', output)

cv2.imwrite('output.jpg', output)

cv2.waitKey(0)
cv2.destroyAllWindows()
