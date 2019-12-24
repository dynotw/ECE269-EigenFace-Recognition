import cv2
import numpy as np

new_faceimage = cv2.imread(r'C:\Users\83414\PycharmProjects\ECE269project\1a.jpg', flags=cv2.IMREAD_GRAYSCALE)

print(new_faceimage)

height, width = new_faceimage.shape[:2]
center = (width / 2, height / 2)
rot_mat = cv2.getRotationMatrix2D(center, 225, 1)
new_faceimage = cv2.warpAffine(new_faceimage, rot_mat, (new_faceimage.shape[1], new_faceimage.shape[0]))
cv2.namedWindow('new_faceimage', cv2.WINDOW_AUTOSIZE)
cv2.imshow('new_faceimage', new_faceimage.reshape((193,162,1)))
cv2.imwrite(r'C:\Users\83414\Desktop\ECE 269\Project\o225_f.jpg', new_faceimage.reshape((193,162,1)))

cv2.waitKey(0)
cv2.destroyAllWindows()