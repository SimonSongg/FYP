import glob
import os.path
import cv2

root = "C:\\Users\\sf995\\iCloudDrive\\DL\\SNE-RoadSeg\\datasets\\GMRPD_dataset"
image_list = sorted(glob.glob(os.path.join(root, 'train', 'rgb', '*.png')))

print(image_list)

useDir = "/".join(image_list[1].split('\\')[:-2])
name = image_list[1].split('\\')[-1]
print(name)

rgb_image = cv2.cvtColor(cv2.imread(os.path.join(useDir, 'rgb', name)), cv2.COLOR_BGR2RGB)
depth_image = cv2.imread(os.path.join(useDir, 'depth_u16', name), cv2.IMREAD_ANYDEPTH)

cv2.imshow("image1",rgb_image)
cv2.imshow("depth",depth_image)

cv2.waitKey(0)