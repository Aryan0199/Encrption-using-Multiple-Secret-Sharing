from Encrypt import *
import cv2
import os
import shutil
folder_path = "C:\\Users\\itsme\\Documents\\ImgEnc2\\Image-Multi-Secret-Sharing-master\\Shadows"
for file_object in os.listdir(folder_path):
    file_object_path = os.path.join(folder_path, file_object)
    if os.path.isfile(file_object_path) or os.path.islink(file_object_path):
        os.unlink(file_object_path)
    else:
        shutil.rmtree(file_object_path)
print ("Please enter the filename of the image to be encrypted")
file_name = input()
img = cv2.imread(file_name)
cv2.imwrite("img4.jpg", img)
#a, b, c = cv2.split(img)
print ("The initial shape of the image is {} and the type of the matrix is {}".format(img.shape, type(img)))
print ("Please enter the value of n, t and k seperated by spaces")
n, t, k = list(map(int, input().split(" ")))
print ("Enter 1 to display the histogram else enter 0")
temp = int(input())
plot_hist = True if temp == 1 else False
print ("Enter 1 to show the input image else enter 0")
temp = int(input())
show_image = True if temp == 1 else False

test = Image_Encryption(n=n, t=t, k=k, img=img, show_image=temp, self_debug=True)
#testb = Image_Encryption(n=n, t=t, k=k, img=b, show_image=temp, self_debug=False)
#testc = Image_Encryption(n=n, t=t, k=k, img=c, show_image=temp, self_debug=False)

shadow_imagesa = test.generate_shadow_images(store_shadows=True)
#shadow_imagesb = testb.generate_shadow_images(store_shadows=True)
#shadow_imagesc = testc.generate_shadow_images(store_shadows=True)
print ("Shadow Images stored in folder Shadows")


