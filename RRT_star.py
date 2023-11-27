import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import heapq


class tree_node:
    def __init__(self, parent, pos:tuple,cost=None) -> None:
        self.pos=pos
        self.parent=parent
        self.cost=cost
        

    def __eq__(self, __o: object) -> bool:
        if self.pos == __o.pos:
            return True
        else:
            return False
    
    def __str__(self) -> str:
        return f"({self.pos[0]},{self.pos[1]})"
    
    def __hash__(self) -> int:
        return hash(self.pos)
    
class RRT:
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
        self.tree_nodes.append(q)

    def add_edge(self, q1,q2):
        self.edges.append((q1,q2))
    
    def remove_edge(self,e):
        try:
            self.edges.remove(e)
        except:
            print(f"{e[0],e[1]} not in the edge list")

    def q_random(self, prob):
        x= np.random.uniform(0,1)
        if x<prob:
            q_rand= self.goal
        else:
            q_rand=(np.random.uniform(0,self.map.shape[0]-1) ,np.random.uniform(0,self.map.shape[1]-1) )  
        return q_rand

    def q_near(self,q_rand):
        distance = np.inf
        qnear=None
        for vertex in self.tree_nodes:
            distance_from_vertex_to_rand_point=self.dist(vertex.pos,q_rand)
            if distance>distance_from_vertex_to_rand_point:
                qnear=vertex
                distance=distance_from_vertex_to_rand_point
        return qnear

    def q_nearest(self, q_rand, n):
        qnear=[]
        for vertex in self.tree_nodes:
            distance_from_vertex_to_new_point=self.dist(vertex.pos,q_rand.pos)
            if distance_from_vertex_to_new_point<n:
                qnear.append(vertex)
        """
        distances = []  # To store distances and corresponding nodes
        for vertex in self.tree_nodes:
            vertex_id= id(vertex)
            distance_from_vertex_to_rand_point = self.dist(vertex.pos, q_rand.pos)
            heapq.heappush(distances, (distance_from_vertex_to_rand_point, vertex_id,vertex))

        # Retrieve the nearest n neighbors
        qnear = [heapq.heappop(distances)[2] for _ in range(min(n, len(distances)))]
        """
        return qnear

    def is_segment_free(self,q1,q2):
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
        if (self.dist((q_near.pos[0],q_near.pos[1]),q_rand))<del_q:
            q_new=tree_node(q_near,q_rand,q_near.cost+self.dist(q_near.pos,q_rand))
            return q_new
                
        theta= np.arctan2(q_rand[1]-q_near.pos[1],q_rand[0]-q_near.pos[0])
        row= round((q_near.pos[0]+ del_q* np.cos(theta)),2)
        col= round((q_near.pos[1]+ del_q* np.sin(theta)),2)
        q_new= tree_node(q_near,(row,col),q_near.cost +self.dist(q_near.pos,(row,col)))
        return q_new

    def dist(self, q1,q2):
        return(np.linalg.norm(np.array(q2)-np.array(q1)))


    def generate_RRT_star(self,iteration,prob,del_q,goal_thresh=5,n=10):
        self.goal_reached=False
        q_start=tree_node(None, self.start,0)
        self.add_node(q_start)

        for iter in range(iteration):
            q_rand= self.q_random(prob)
            q_near= self.q_near(q_rand)
            q_new = self.extend_tree(q_near,q_rand,del_q)

            if self.is_segment_free(q_near.pos,q_new.pos):
                q_min=q_near
                q_near_list=self.q_nearest(q_new,n)

                for q in q_near_list:
                    if self.is_segment_free(q.pos,q_new.pos):
                        qcost=q.cost +  self.dist(q.pos,q_new.pos)
                        if qcost<q_new.cost:
                            q_min=q
                            q_new.parent=q_min
                            q_new.cost=q_min.cost +  self.dist(q_min.pos,q_new.pos)

                self.add_node(q_new)
                self.add_edge(q_min,q_new)            
                
                for q in q_near_list:
                    if ((q!=q_min) and (self.is_segment_free(q_new.pos,q.pos)) and (q.cost> q_new.cost + self.dist(q_new.pos,q.pos))):
                        self.remove_edge((q.parent,q))
                        q.parent=q_new
                        q.cost=q_new.cost + self.dist(q_new.pos,q.pos)
                        self.add_edge(q_new,q)
                
            else:
                continue


    def generate_path(self):
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
            """
            for count,i in enumerate(self.tree_nodes):
                if i.pos == self.goal:
                    goal_index=count
                    break
            current_q=self.tree_nodes[goal_index]
            while current_q!=self.tree_nodes[0]:
                self.first_path.append(current_q.pos)
                current_q=current_q.parent
            if current_q==self.tree_nodes[0]:
                self.first_path.append(current_q.pos)
            for i in range(len(self.path)-1):
                self.distance+=self.dist(self.path[i],self.path[i+1])
            #print(f"First Path Distance : {self.distance}")
            print(self.first_path)            
            """
        else: 
            print("Goal is not reached!! Please generate a new tree or increase the number of iterations.")


    def draw_path(self):
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
        # plt.scatter([self.start[1],self.goal[1]],[self.start[0],self.goal[0]],c=["r","g"],marker="*")
        # plt.plot([self.start[1],self.goal[1]],[self.start[0],self.goal[0]],"g*")
        plt.plot([self.start[1]],[self.start[0]],"g*")
        plt.plot([self.goal[1]],[self.goal[0]],"r*")
        plt.show()

    

if __name__=="__main__":
    #mp=RRT((60,60),(90,60),'map1.png') 
    #mp=RRT((50,90),(375,375),'map3.png')
    #mp=RRT((8,31),(139,38),'map2.png')
    mp=RRT((10,10),(90,70),'map0.png')
    mp.generate_RRT_star(1000,0.2,5,5,30)
    mp.generate_path()
    mp.draw_path()
    
