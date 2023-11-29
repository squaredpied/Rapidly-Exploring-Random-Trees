# Title  : Rapidly-Exploring Random Tree (RRT) algorithm
# Date   : 28/11/2023
# Authors : Precious I. Philip-Ifabiyi, Atharva Anil Patwe

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="Rapidly-Exploring Random Tree (RRT) algorithm implementation.")
parser.add_argument("path_to_grid_map_image", type=str, help="Path to the grid map image")
parser.add_argument("k", type=int, help="Maximum Number of Iterations")
parser.add_argument("del_q", type=int, help="Step size of the tree edge in the RRT")
parser.add_argument("prob", type=float, help="Probability of samplig goal as q_rand")
parser.add_argument("qstart_x", type=int, help="X-coordinate of the starting point")
parser.add_argument("qstart_y", type=int, help="Y-coordinate of the starting point")
parser.add_argument("qgoal_x", type=int, help="X-coordinate of the goal point")
parser.add_argument("qgoal_y", type=int, help="Y-coordinate of the goal point")

args = parser.parse_args()
class tree_node:
    """
    Represents a node in a tree used for pathfinding algorithms.

    Attributes:
    - parent: TreeNode
        The parent node in the tree.
    - pos: tuple
        The position coordinates (x, y) of the node.
    """

    def __init__(self, parent:tuple, pos:tuple) -> None:
        self.pos=pos
        self.parent=parent

    def __eq__(self, __o: object) -> bool:
        """
        Checks if two TreeNode objects are equal based on their positions.

        Args:
        - other: object
            Another TreeNode object to compare.

        Returns:
        - bool
            True if positions are equal, False otherwise.
        """
        if self.pos == __o.pos:
            return True
        else:
            return False
    
class RRT:
    """
    Rapidly-Exploring Random Tree (RRT) algorithm implementation. 

    Attributes:
    - tree_nodes: list
        List of tree nodes.
    - edges: list
        List of edges connecting tree nodes.
    - goal_reached: bool
        Indicates if the goal has been reached with default being False.
    - path: list
        List containing the path from start to goal if the goal is reached.
    - smooth_path: list
        List of nodes in the smoothen path
    - smooth_path_dis: float
        Cost of smoothen path
    """
    tree_nodes= []
    edges= []
    smooth_path=[]
    path_dis=0
    smooth_path_dis=0
    goal_reached=False
    path=[]

    def __init__(self, start,goal,map:str) -> None:
        self.start=start
        self.goal=goal
        self.map=self.gridmap(map)

    def gridmap(self,map):
        image = Image.open(map).convert('L')
        grid_map = np.array(image.getdata()).reshape(image.size[0],
        image.size[1])/255
        # binarize the image
        grid_map[grid_map > 0.5] = 1
        grid_map[grid_map <= 0.5] = 0
        # Invert colors to make 0 -> free and 1 -> occupied
        grid_map= (grid_map * -1) + 1
        return grid_map


    def add_node(self, q:tree_node):
        """
        Adds a node to the RRT tree.

        Args:
        - q: TreeNode
            Node to be added to the tree.
        """
        self.tree_nodes.append(q)

    def add_edge(self, q1,q2):
        """
        Adds an edge between two nodes in the RRT tree.

        Args:
        - q1: TreeNode
            First node for the edge.
        - q2: TreeNode
            Second node for the edge.
        """
        self.edges.append((q1,q2))

    def q_random(self, prob):
        """
        Generates a random point with a probability threshold.

        Args:
        - prob: float
            Probability threshold for choosing the goal as a random point.

        Returns:
        - tuple
            Randomly sampled position (x, y) coordinates.
        """
        x= np.random.uniform(0,1)
        if x<prob:
            q_rand= self.goal
        else:
            q_rand=(np.random.uniform(0,self.map.shape[0]-1) ,np.random.uniform(0,self.map.shape[1]-1) )  
        return q_rand

    
    def q_nearest(self,q_rand):
        """
        Finds the nearest node to a given point q_rand among the tree nodes.

        Args:
        - q_rand: TreeNode
            Randomly sampled node for which the nearest node is to be found.

        Returns:
        - TreeNode or None
            Nearest node to q_rand among the tree nodes, or None if tree_nodes is empty.
        """
        distance = np.inf
        qnear=None
        for vertex in self.tree_nodes:
            distance_from_vertex_to_rand_point=self.dist(vertex.pos,q_rand)
            if distance>distance_from_vertex_to_rand_point:
                qnear=vertex
                distance=distance_from_vertex_to_rand_point
        return qnear

    def is_segment_free(self,q1,q2):
        """
        Checks if the segment between two points q1 and q2 is free from obstacles. The function is implemented using recursion.

        Args:
        - q1: tuple
            Coordinates of point q1 (x, y).
        - q2: tuple
            Coordinates of point q2 (x, y).

        Returns:
        - bool
            True if the segment between q1 and q2 is free from obstacles, False otherwise.
        """
        distance=int(self.dist(q1,q2))
        mid_point=(int((q1[0]+q2[0])/2), int((q1[1]+q2[1])/2))

        if q2[0]>(self.map.shape[0]-1) or q2[1]>(self.map.shape[1]-1) or q2[0]<0 or q2[1]<0:
            return False

        if (distance>2):
            if self.map[int(mid_point[0]),int(mid_point[1])]==1:
                return False
        else:
            if self.map[int(q1[0]),int(q1[1])]==0 and self.map[int(q2[0]),int(q2[1])]==0:
                return True
            else:
                return False
        return self.is_segment_free(q1,mid_point) and self.is_segment_free(mid_point,q2)

    def extend_tree(self,q_near, q_rand, del_q):
        """
        Extends the RRT tree from a near node (q_near) towards a random node (q_rand) with a specified step size (del_q).

        Args:
        - q_near: TreeNode
            Node closest to the randomly sampled node.
        - q_rand: tuple
            Randomly sampled position (x, y) coordinates.
        - del_q: float
            Step size for extending the tree.

        Returns:
        - q_new: TreeNode
            New node added to the tree towards q_rand from q_near.
        """
        if (self.dist((q_near.pos[0],q_near.pos[1]),q_rand))<del_q:
            q_new=tree_node(q_near,q_rand)
            return q_new

        theta= np.arctan2(q_rand[1]-q_near.pos[1],q_rand[0]-q_near.pos[0])
        row= np.round((q_near.pos[0]+ del_q* np.cos(theta)),2)
        col= np.round((q_near.pos[1]+ del_q* np.sin(theta)),2)
        q_new= tree_node(q_near,(row,col))
        return q_new

    def dist(self, q1,q2):
        return(np.linalg.norm(np.array(q1)-np.array(q2)))

    def generate_RRT(self,iteration,prob,del_q):
        self.goal_reached=False
        q_start=tree_node(None, self.start)
        self.add_node(q_start)
        for iter in range(iteration):
            q_rand= self.q_random(prob)
            q_near= self.q_nearest(q_rand)
            q_new = self.extend_tree(q_near,q_rand,del_q)

            if self.is_segment_free(q_near.pos,q_new.pos):
                self.add_node(q_new)
                self.add_edge(q_near,q_new)
                if q_new.pos==self.goal:
                    self.goal_reached=True
                    print("Path found in ", iter, "iterations")
                    break
            else:
                continue


    def generate_path(self):
        current_q=self.tree_nodes[-1]
        if self.goal_reached:
            while current_q!=self.tree_nodes[0]:
                self.path.append(current_q.pos)
                self.path_dis+=self.dist(current_q.pos, current_q.parent.pos)
                current_q=current_q.parent
            if current_q==self.tree_nodes[0]:
                self.path.append(current_q.pos)
            self.path=list(reversed(self.path))
            print("Distance", self.path_dis)
            print(self.path)

        else: 
            print("Goal is not reached!! Please generate a new tree or increase the number of iterations.")

    def draw_path(self):
        plt.figure(1)
        plt.matshow(self.map,fignum=1)
        plt.colorbar()
        for v in self.tree_nodes:
            plt.plot(v.pos[1], v.pos[0], 'k+')
        
        for e in self.edges:
            plt.plot([e[0].pos[1], e[1].pos[1]],[e[0].pos[0], e[1].pos[0]] ,
                    "g--")        
            
        points=np.zeros((len(self.path),2))
        for i in range(len(self.path)):
            points[i,0]=self.path[i][1]
            points[i,1]=self.path[i][0]
        plt.plot(points[:,0],points[:,1],'r')
        plt.plot([self.start[1]],[self.start[0]],"g*")
        plt.plot([self.goal[1]],[self.goal[0]],"r*")
        plt.title("Path Found by Rapidly-Exploring Random Tree")

        plt.figure(2)
        plt.matshow(self.map,fignum=2)
        plt.colorbar()
        for v in self.tree_nodes:
            plt.plot(v.pos[1], v.pos[0], 'k+')
        
        for e in self.edges:
            plt.plot([e[0].pos[1], e[1].pos[1]],[e[0].pos[0], e[1].pos[0]] ,
                    "g--")        
        plt.title("Smoothed Path")
        points=np.zeros((len(self.smooth_path),2))
        for i in range(len(self.smooth_path)):
            points[i,0]=self.smooth_path[i][1]
            points[i,1]=self.smooth_path[i][0]
        plt.plot(points[:,0],points[:,1],'r')
        plt.plot([self.start[1]],[self.start[0]],"g*")
        plt.plot([self.goal[1]],[self.goal[0]],"r*")
        plt.show()

    def smoothen_path(self):
        if len(self.path)!=0:
            self.smooth_path.append(self.path[-1])
            current_pos=self.path[-1]
            current_index=self.path.index(current_pos)
            while (self.path[0] in self.smooth_path) == False:
                new_list=self.path[0:current_index]
                for i in new_list:
                    if (self.is_segment_free(i,current_pos)):
                        self.smooth_path.append(i)
                        self.smooth_path_dis+=self.dist(current_pos,i)
                        current_pos=i
                        current_index=self.path.index(current_pos)
                        break
        self.smooth_path=list(reversed(self.smooth_path))
        print("Smooth distance:", self.smooth_path_dis)
        print(self.smooth_path)

if __name__=="__main__":
    #mp=RRT((10,10),(90,70),'map0.png')
    #mp=RRT((60,60),(90,60),'map1.png') 
    #mp=RRT((8,31),(139,38),'map2.png')
    #mp=RRT((50,90),(375,375),'map3.png')   
    mp=RRT((args.qstart_x,args.qstart_y),(args.qgoal_x,args.qgoal_y),args.path_to_grid_map_image)                    
    #mp.generate_RRT(30000,0.2,5)
    mp.generate_RRT(args.k,args.prob,args.del_q)
    mp.generate_path()
    mp.smoothen_path()
    mp.draw_path()

