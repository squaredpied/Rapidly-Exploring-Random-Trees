import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

import argparse

parser = argparse.ArgumentParser(description="Rapidly-Exploring Random Tree star (RRT*) algorithm implementation.")
parser.add_argument("path_to_grid_map_image", type=str, help="Path to the grid map image")
parser.add_argument("k", type=int, help="Maximum Number of Iterations")
parser.add_argument("del_q", type=int, help="Step size of the tree edge in the RRT")
parser.add_argument("prob", type=float, help="Probability of samplig goal as q_rand")
parser.add_argument("max_distance", type=int, help="Maximum distance of influence for rewiring the tree")
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
    - cost: int or None, optional
        The cost associated with reaching this node from the start node.
    """
    def __init__(self, parent, pos:tuple,cost=None) -> None:
        self.pos=pos
        self.parent=parent
        self.cost=cost
        

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
    
    def __str__(self) -> str:
        return f"({self.pos[0]},{self.pos[1]})"
    
    def __hash__(self) -> int:
        return hash(self.pos)
    
class RRT_star:
    """
    Rapidly-Exploring Random Tree star (RRT_star) algorithm implementation. 

    Attributes:
    - tree_nodes: list
        List of tree nodes.
    - edges: list
        List of edges connecting tree nodes.
    - goal_reached: bool
        Indicates if the goal has been reached with default being False.
    - path: list
        List containing the path from start to goal if the goal is reached.
    """
    tree_nodes= []
    edges= []
    goal_reached=False
    path=[]
    first_path=[]

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
    
    def remove_edge(self,e):
        """
        Removes an edge from the RRT tree.

        Args:
        - e: tuple
            Edge to be removed (tuple of two nodes).
        """
        try:
            self.edges.remove(e)
        except:
            print(f"{e[0],e[1]} not in the edge list")

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

    def q_near(self,q_rand):
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

    def q_nearest(self, q_rand, n):
        """
        Finds the nodes that are less than n distance from a given point q_rand.

        Args:
        - q_rand: TreeNode
            Randomly sampled node for which nearest neighbors are to be found.
        - n: int
            Distance threshold for considering a node as a neighbour

        Returns:
        - list[TreeNode]
            List of neighbors to q_rand.
        """
        qnear=[]
        for vertex in self.tree_nodes:
            distance_from_vertex_to_new_point=self.dist(vertex.pos,q_rand.pos)
            if distance_from_vertex_to_new_point<n:
                qnear.append(vertex)
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
            if (q_rand==self.goal):
                row=q_rand[0]
                col=q_rand[1]
            else:
                row=round(q_rand[0],2)
                col=round(q_rand[1],2)
            q_new=tree_node(q_near,(row,col),q_near.cost+self.dist(q_near.pos,q_rand))
            return q_new
                
        theta= np.arctan2(q_rand[1]-q_near.pos[1],q_rand[0]-q_near.pos[0])
        row= round((q_near.pos[0]+ del_q* np.cos(theta)),2)
        col= round((q_near.pos[1]+ del_q* np.sin(theta)),2)
        q_new= tree_node(q_near,(row,col),q_near.cost +self.dist(q_near.pos,(row,col)))
        return q_new

    def dist(self, q1,q2):
        return(np.linalg.norm(np.array(q2)-np.array(q1)))


    def generate_RRT_star(self,iteration,prob,del_q,n=10):
        """
        Generates an RRT* tree by iteratively sampling a random point in the space and advancing towards it if the edge
        formed is free. The advancement is done from the node of least cost (from start). Other nodes of the tree are
        also checked if they can be reached through this new node with less cost. If it is possible then the tree
        structure is updated.

        Args:
        - iteration: int
            Number of iterations for tree generation.
        - prob: float
            Probability threshold for choosing the goal as a random point.
        - del_q: float
            Delta Q for tree extension.
        - goal_thresh: int, optional
            Threshold distance to consider reaching the goal.
        - n: int, optional
            Distance threshold for node neighbours

        Returns:
        - None
        """
        self.goal_reached=False
        # Initialize start node with zero cost
        q_start=tree_node(None, self.start,0)
        # Add start node to the node list
        self.add_node(q_start)

        for iter in range(iteration):
            # Generate Random node
            q_rand= self.q_random(prob)
            q_near= self.q_near(q_rand)
            q_new = self.extend_tree(q_near,q_rand,del_q)

            if self.is_segment_free(q_near.pos,q_new.pos):
                q_min=q_near
                q_near_list=self.q_nearest(q_new,n)
                # Find the node with least resistance towards start
                for q in q_near_list:
                    if self.is_segment_free(q.pos,q_new.pos):
                        qcost=q.cost +  self.dist(q.pos,q_new.pos)
                        if qcost<q_new.cost:
                            # Update the new q with parent and cost found
                            q_min=q
                            q_new.parent=q_min
                            q_new.cost=q_min.cost +  self.dist(q_min.pos,q_new.pos)

                self.add_node(q_new)
                self.add_edge(q_min,q_new)            
                # Find nodes which can be reached with q_new at less cost
                for q in q_near_list:
                    if ((q!=q_min) and (self.is_segment_free(q_new.pos,q.pos)) and (q.cost> q_new.cost + self.dist(q_new.pos,q.pos))):
                        self.remove_edge((q.parent,q))
                        q.parent=q_new
                        q.cost=q_new.cost + self.dist(q_new.pos,q.pos)
                        self.add_edge(q_new,q)
                # If goal is reached change the goal_reached value
                if q_new.pos==self.goal and self.goal_reached==False:
                    self.goal_reached=True
                    print(f"Goal reached in {iter} iterations!!!!")
                    self.generate_path()
                    print("\nPlease close the map window to continue the iterations....\n")
                    self.draw_path()
                    self.path=[]
                    self.distance=0
                    
            else:
                continue


    def generate_path(self):
        """
        Generates a path from the RRT* tree if the goal_reached is True. It uses the tree structure stored in the nodes
        and edges to find out the path.

        Returns:
        - None
        """
        goal_index=None
        
        goal_vertex=self.tree_nodes[0].pos
        goal_index=None

        for count,i in enumerate(self.tree_nodes):
            if (self.dist(i.pos,self.goal) < self.dist(goal_vertex,self.goal)) and self.is_segment_free(self.goal,i.parent.pos):
                goal_index=count
                goal_vertex=i.pos
        
        if (goal_index!=None):
            self.tree_nodes[goal_index].pos=self.goal
            current_q=self.tree_nodes[goal_index]

        self.distance=0
        
        if goal_index!=None:
            while current_q!=self.tree_nodes[0]:
                self.path.append(current_q.pos)
                current_q=current_q.parent
            if current_q==self.tree_nodes[0]:
                self.path.append(current_q.pos)
            for i in range(len(self.path)-1):
                self.distance+=self.dist(self.path[i],self.path[i+1])
            print(f"Path Distance : {self.distance}")
            self.path=list(reversed(self.path))
            print(self.path)
        else: 
            print("Goal is not reached!! Please generate a new tree or increase the number of iterations.")


    def draw_path(self):
        """
        Draws the RRT* tree and path on the map using the nodes and edges stored in the tree structure.

        Returns:
        - None
        """
        plt.matshow(self.map)
        plt.colorbar()
        for v in self.tree_nodes:
            plt.plot(v.pos[1], v.pos[0], 'k+')
        
        for e in self.edges:
            plt.plot([e[0].pos[1], e[1].pos[1]],[e[0].pos[0], e[1].pos[0]] , "w-")        

        points=np.zeros((len(self.path),2))
        for i in range(len(self.path)):
            points[i,0]=self.path[i][1]
            points[i,1]=self.path[i][0]
        plt.plot(points[:,0],points[:,1],'r')        
        plt.plot([self.start[1]],[self.start[0]],"g*")
        plt.plot([self.goal[1]],[self.goal[0]],"r*")
        plt.show()

    

if __name__=="__main__":
    #mp=RRTstar((60,60),(90,60),'map1.png') 
    #mp=RRT((50,90),(375,375),'map3.png')
    #mp=RRT((8,31),(139,38),'map2.png')
    #mp=RRT((10,10),(90,70),'map0.png')
    #mp.generate_RRT_star(3000,0.2,5,30)
    mp=RRT_star((args.qstart_x,args.qstart_y),(args.qgoal_x,args.qgoal_y),args.path_to_grid_map_image)
    mp.generate_RRT_star(args.k,args.prob,args.del_q,args.max_distance)
    print(f"\nAfter {args.k} Iterations, cost of the path is ")
    
    mp.generate_path()
    mp.draw_path()
    
