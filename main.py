"""
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

xxx=is_segment_free((10,10),(40,40),grid_map)
print(xxx)
"""
import numpy as np
"""
def is_segment_free(point1, point2, gridmap):
    stack = [(point1, point2)]

    while stack:
        start, end = stack.pop()
        distance = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
        mid_point = (int((start[0] + end[0]) / 2), int((start[1] + end[1]) / 2))

        if distance <= 1:
            if gridmap[start[1], start[0]] == 0 and gridmap[end[1], end[0]] == 0:
                continue
            else:
                return False

        if gridmap[mid_point[1], mid_point[0]] == 1:
            return False

        stack.append((start, mid_point))
        stack.append((mid_point, end))

    return True
"""
def is_segment_free(q1, q2, map):
    distance=np.sqrt((q1[0]-q2[0])**2+(q1[1]-q2[1])**2)
    mid_point=(int((q1[0]+q2[0])/2), int((q1[1]+q2[1])/2))
    if q2[0]>(map.shape[0]-1) or q2[1]>(map.shape[1]-1):
        return False
    if (distance>1.5):
        if map[mid_point[0],mid_point[1]]==1:
            return False
    else:
        if map[q1[0],q1[1]]==0 and map[q2[0],q2[1]]==0:
            return True
        else:
            return False
    return is_segment_free(q1,mid_point,map) and is_segment_free(mid_point,q2,map)
# Example usage:
# Assuming 'gridmap' is a 2D numpy array representing the environment
# where 0 is free space and 1 is an obstacle

# Example gridmap (5x5)
example_gridmap = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
])

# Points to check
start_point = (0, 0)
end_point = (4, 0)

if is_segment_free(start_point, end_point, example_gridmap):
    print("The segment is free!")
else:
    print("Obstacle detected!")
