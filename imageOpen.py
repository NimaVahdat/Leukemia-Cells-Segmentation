import cv2

def image_opener():
    images_Normal = []
    images_BW = []
    
    for i in range(1, 50):
        img = cv2.imread('ALL_IDB2\img\Im%.3d_1.tif'%i, 1)
        images_Normal.append(img)
        
        img = cv2.imread('ALL_IDB2\img\Im%.3d_1.tif'%i, 0)
        images_BW.append(img)

    return images_Normal, images_BW