
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def is_segment_free(point1, point2,gridmap):
    distance=np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
    mid_point=(int((point1[0]+point2[0])/2), int((point1[1]+point2[1])/2))
    if (distance<1):
        if gridmap[point1[0],point1[1]]==0 and gridmap[point2[0],point2[1]]==0:
            return True
        else:
            return False
    if (gridmap[mid_point[0],mid_point[1]]==1):
        return False
    if is_segment_free(point1,mid_point,gridmap) and is_segment_free(mid_point,point2,gridmap): 
        return True

class TreeNode:
    def __init__(self,position,parent):
        self.position=position
        self.parent=parent

# Load grid map
image = Image.open('map0.png').convert('L')
grid_map = np.array(image.getdata()).reshape(image.size[0],
image.size[1])/255
# binarize the image
grid_map[grid_map > 0.5] = 1
grid_map[grid_map <= 0.5] = 0
# Invert colors to make 0 -> free and 1 -> occupied
grid_map = (grid_map * -1) + 1
# Show grid map
plt.matshow(grid_map)
plt.colorbar()
plt.show()
print(grid_map.shape)

#xxx=is_segment_free((10,10),(40,40),grid_map)
print(grid_map[2.5,2.5])
