#K-means Clustering and Hierarchical Agglomerative Clustering (Single-Link/Complete-Link/Average-Link) on 2D and 3D or more data randomly generated
import numpy as np
import matplotlib.pyplot as plt
import time
from random import randint
import os


def array_cover(list,num):  #K-means   #check if all random centroids cover at least one point
                            # we mustn't have a centroid with no points in its cluster because we need exactly K clusters
  number_of_points=len(list)
  for i in range(num):
    flag=1
    for j in range(number_of_points):
      if list[j]==i: flag=2
    if(flag==1): return 0
  return 1


def re_assign(data, centroids): #K-means  #take the previous assignment of points to centroids and reassigning them to the new centroids
    c = []
    for i in data:
        min=1500000
        cluster_index=-1
        for j in range (len(centroids)):
           dist=np.linalg.norm(i - centroids[j])
           if dist<min: 
               min=dist
               cluster_index=j
        c.append(cluster_index)
    return c

def centroids_find_mean(data, num_clusters, assignments): #K-means   #finds the means of the clusters and sets them as the new centroids
    cen = []
    for c in range(len(num_clusters)):
        cen.append(np.mean([data[x] for x in range(len(data)) if assignments[x] == c], axis=0))
    return cen

def min_lower(matrix):           #finds the min element in the lower triangle of the distance matrix and returns the position of it
  le=len(matrix)
  min=1500000
  position=[]
  for x in range (0,le):
    for y in range (x+1,le):
     if matrix[y][x]<=min: 
       min=matrix[y][x]
       position=[y,x]
  return position
 
def shrink_matrix_single_link(matrix):              #take a n by n matrix and shrink it by 1 to get (n-1) by (n-1)
                                        #each shrinking step means two clusters are merging
  le=len(matrix)
  index_to_remove=min_lower(matrix)[0]
  index_to_update=min_lower(matrix)[1]
  matrix2=np.delete(matrix, index_to_remove, 0)
  matrix2=np.delete(matrix2, index_to_remove, 1)
  
  #update the required entries of the distance matrix using (single-link or complete-link or average-link)
  for i in range(0,index_to_update):
    matrix2[index_to_update][i]=min(matrix[index_to_update][i],matrix[index_to_remove][i])      #single-link  
  for i in range(index_to_update+1,index_to_remove):
    matrix2[index_to_update][i]=min(matrix[index_to_update][i],matrix[index_to_remove][i])
  for i in range(index_to_remove,len(matrix)-1):
    matrix2[index_to_update][i]=min(matrix[index_to_update][i],matrix[index_to_remove][i])

  points[index_to_update]= points[index_to_update]+points[index_to_remove]
  del points[index_to_remove]

  return matrix2     #finally return the smaller n-1 by n-1 matrix to be given to the same function again

def shrink_matrix_complete_link(matrix):              #take a n by n matrix and shrink it by 1 to get (n-1) by (n-1)
                                        #each shrinking step means two clusters are merging
  le=len(matrix)
  index_to_remove=min_lower(matrix)[0]
  index_to_update=min_lower(matrix)[1]
  matrix2=np.delete(matrix, index_to_remove, 0)
  matrix2=np.delete(matrix2, index_to_remove, 1)
  
  #update the required entries of the distance matrix using (single-link or complete-link or average-link)
  for i in range(0,index_to_update):
    matrix2[index_to_update][i]=max(matrix[index_to_update][i],matrix[index_to_remove][i])      #complete-link  
  for i in range(index_to_update+1,index_to_remove):
    matrix2[index_to_update][i]=max(matrix[index_to_update][i],matrix[index_to_remove][i])
  for i in range(index_to_remove,len(matrix)-1):
    matrix2[index_to_update][i]=max(matrix[index_to_update][i],matrix[index_to_remove][i])


  points[index_to_update]= points[index_to_update]+points[index_to_remove]
  del points[index_to_remove]

  return matrix2     #finally return the smaller n-1 by n-1 matrix to be given to the same function again

def shrink_matrix_average_link(matrix):              #take a n by n matrix and shrink it by 1 to get (n-1) by (n-1)
                                        #each shrinking step means two clusters are merging
  le=len(matrix)
  index_to_remove=min_lower(matrix)[0]
  index_to_update=min_lower(matrix)[1]
  matrix2=np.delete(matrix, index_to_remove, 0)
  matrix2=np.delete(matrix2, index_to_remove, 1)
  
  #update the required entries of the distance matrix using (single-link or complete-link or average-link)
  for i in range(0,index_to_update):
    matrix2[index_to_update][i]=(matrix[index_to_update][i]+matrix[index_to_remove][i])/2        #average-link  
  for i in range(index_to_update+1,index_to_remove):
    matrix2[i][index_to_update]=(matrix[i][index_to_update]+matrix[index_to_remove][i])/2
  for i in range(index_to_remove,len(matrix)-1):
    matrix2[i][index_to_update]=(matrix[i+1][index_to_update]+matrix[i+1][index_to_remove])/2

  points[index_to_update]= points[index_to_update]+points[index_to_remove]
  del points[index_to_remove]

  return matrix2     #finally return the smaller n-1 by n-1 matrix to be given to the same function again

print("\n ***Welcome To the Clustering Algorithms K-means and HAC*** \n")
dimension=int(input('\n What is the dimension (2 and 3 recommended for visualization) '))
num_of_points=int(input('\n How many points to generate '))
range_of_value=int(input('\n Points Values in each Dimension vary in [0,x] what is x '))








colors=['blue','red','green','violet','orange','teal','navy','magenta','lime','maroon'] +['black']*90



#data = np.random.randint(0, range_of_value, size=(num_of_points, dimension))

data=np.random.uniform(0, range_of_value, size=(num_of_points, dimension))

print("\n Data was successfully generated and saved.")

keep=1
while 1:
  
  if(keep==0):
    dimension=int(input('\n What is the dimension (2 and 3 recommended for visualization)'))
    num_of_points=int(input('\n How many points to generate '))
    range_of_value=int(input('\n Points Values in each Dimension vary in [0,x] what is x '))
    data=  np.random.uniform(0, range_of_value, size=(num_of_points, dimension))

  choice=int(input("\n What algorithm do you want to use? 1: K-means  2: HAC "))



  if choice==1:
    num_of_clusters=int(input("\n How many Clusters do you wish to have? noticeably smaller than "+str(num_of_points)+" " ))
    start = time.time()
    
    check=0
    counter=0
    while check==0:       #continue generating random centroids until there is no centroid without a point in its cluster
#     centroids = np.random.randint(0,range_of_value, size=(num_of_clusters, dimension)) 
#      centroids = (np.random.normal(size=(num_of_clusters, dimension))*(range_of_value/2) ) + np.mean(data, axis=0).reshape((1, dimension)) (latest)
        #try new
      centroids=[]
      temp_arr=list(range(0, num_of_points))
      for i in range (0,num_of_clusters):
        chosen=np.random.randint(0,len(temp_arr))
        centroids.append(data[temp_arr[chosen]])
        del temp_arr[chosen]
        #try new

      firsassignment = re_assign(data, centroids)
      #print(firsassignment)
      if (array_cover(firsassignment,num_of_clusters))==1: check=1
      counter=counter+1
      if(counter==1000): 
        print(" !!! You chose too many clusters for your Data. Try again !!!")
        break

    if(counter!=1000): 
      dummy=[-1]*num_of_points
      iter=0
      while 1:    #keep iterating until convergence criterion is satisfied
        iter=iter+1
        a = re_assign(data, centroids)
        centroids = centroids_find_mean(data, centroids, a)
        centroids = np.array(centroids)
        if a==dummy: break
        dummy=a

      if dimension==2:
          for j in range (num_of_clusters):
            temp=[]
            for i in range (num_of_points):
              if a[i]==j: temp.append(data[i])
            temp=np.array(temp)
            plt.scatter(temp[:, 0], temp[:, 1],c=colors[j])
          plt.scatter(centroids[:, 0], centroids[:, 1],c='yellow',edgecolors='black')
          end = time.time()
          plt.title('K-means Algo / Run Time '+str(end-start)+" / iterations: "+str(iter))
          plt.show()       
        
      elif dimension==3:
          plt.scatter(centroids[:,0], centroids[:,1], centroids[:,2],c='yellow',edgecolors='black')
          for j in range (num_of_clusters):
            temp=[]
            for i in range (num_of_points):
              if a[i]==j: temp.append(data[i])
            temp=np.array(temp)
            plt.scatter(temp[:, 0], temp[:, 1],temp[:, 2],c=colors[j])
          
          end = time.time()
          plt.title('K-means Algo / Run Time '+str(end-start)+" / iterations: "+str(iter))
          plt.show()

          fig = plt.figure()
          ax = fig.add_subplot(111, projection='3d')
          
          for j in range (num_of_clusters):
            temp=[]
            for i in range (num_of_points):
              if a[i]==j: temp.append(data[i])
            temp=np.array(temp)
            ax.scatter(temp[:, 0], temp[:, 1],temp[:, 2],c=colors[j])
          ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2],c='yellow',edgecolors='black')
          plt.title('K-means Algo / Run Time '+str(end-start)+" / iterations: "+str(iter))
          plt.show()
      
      elif dimension>=4:
          for j in range (num_of_clusters):
            temp=[]
            for i in range (num_of_points):
              if a[i]==j: temp.append(data[i])
            temp=np.array(temp)
            print('\n Cluster ', j+1,' contains: \n', temp)
          end = time.time()
          print('\n K-means Algo / Run Time '+str(end-start)+" / iterations: "+str(iter))     
      
    

  elif choice==2:
    num_of_clusters=int(input("\n How many Clusters do you wish to have?" ))
    type_of_HAC=int(input("\n Choose 1: HAC Single Link  2: HAC Complete Link  3: HAC Average Link: "))
    if type_of_HAC==1: typestr=" Single-Link "
    elif type_of_HAC==2: typestr=" Complete-Link "
    elif type_of_HAC==3: typestr=" Average-Link "

    start = time.time()
    points=[]               #this is the list of the points to be updated as clusters merge through the process
    for i in range(0,num_of_points):
      points.append([i])

    distance_matrix=np.zeros((num_of_points,num_of_points))  

    for x in range (0,num_of_points):          #calculate all the mutual distances and insert them into the lower triangle of the distance matrix
      for y in range (x+1,num_of_points):
        distance_matrix[y][x]=np.linalg.norm(data[x] - data[y])

    if type_of_HAC==1:
      #print(points)
      for i in range (0,num_of_points-num_of_clusters):       #keep shrinking until you get k clusters
        #print("--------------------------------------------")
        #print(distance_matrix)
        distance_matrix=shrink_matrix_single_link(distance_matrix)
        #print(points)
        #print('********************************************')
    elif type_of_HAC==2:
      #print(points)
      for i in range (0,num_of_points-num_of_clusters):       #keep shrinking until you get k clusters
        #print("--------------------------------------------")
        #print(distance_matrix)
        distance_matrix=shrink_matrix_complete_link(distance_matrix)
        #print(points)
        #print('********************************************')
    elif type_of_HAC==3:
      #print(points)
      for i in range (0,num_of_points-num_of_clusters):       #keep shrinking until you get k clusters
        #print("--------------------------------------------")
        #print(distance_matrix)
        distance_matrix=shrink_matrix_average_link(distance_matrix)
        #print(points)
        #print('********************************************')
    
    if dimension==2:
      for i in range (num_of_clusters):
        temp=[]
        for j in range (len(points[i])):
          temp.append(data[points[i][j]])
        temp=np.array(temp)
        plt.scatter(temp[:, 0], temp[:, 1],c=colors[i])
      end = time.time()
      plt.title('HAC Algo '+typestr+'/ Run Time '+str(end-start))
      plt.show()       
      
    elif dimension==3:
      for i in range (num_of_clusters):
        temp=[]
        for j in range (len(points[i])):
          temp.append(data[points[i][j]])
        temp=np.array(temp)
        plt.scatter(temp[:, 0], temp[:, 1],temp[:,2],c=colors[i])
      end = time.time()
      plt.title('HAC Algo '+typestr+'/ Run Time '+str(end-start))
      plt.show()          

      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
        
      for i in range (num_of_clusters):
        temp=[]
        for j in range (len(points[i])):
          temp.append(data[points[i][j]])
        temp=np.array(temp)
        ax.scatter(temp[:, 0], temp[:, 1],temp[:,2],c=colors[i])
      plt.title('HAC Algo '+typestr+'/ Run Time '+str(end-start))
      plt.show()  

    elif dimension>=4:
      for i in range (num_of_clusters):
        temp=[]
        for j in range (len(points[i])):
          temp.append(data[points[i][j]])
        temp=np.array(temp)
        print("\n Cluster ", i+1, " contains: \n", temp)
      
      end = time.time()
      print('\n HAC Algo '+typestr+'/ Run Time '+str(end-start))
         
  
  keep=int(input('\n Do you want to keep the data 1:YES 0:No?'))




