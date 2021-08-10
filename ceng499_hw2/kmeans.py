import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def initialize_clusters(data,k):
    number_of_data = data.shape[0]
    number_of_dimensions = data.shape[1]
    mean=np.mean(data, axis=0)
    std=np.std(data,axis=0)
    centers=np.random.randn(k,number_of_dimensions)*std + mean
    return centers
    

def assign_clusters(data, cluster_centers):
    global location
    cluster_table=np.zeros(len(data),dtype=int)
    i=0
    for point in data:
        index=0
        dist = float("inf")
        for cluster in cluster_centers:
            distance = np.linalg.norm(point - cluster)
            if distance < dist:
                dist=distance
                location = index
            index=index+1
        cluster_table[i]=location
        i=i+1
        
    return cluster_table
        
    """
    Assigns every data point to its closest (in terms of euclidean distance) cluster center.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :return: An (N, ) shaped numpy array. At its index i, the index of the closest center
    resides to the ith data point.
    """


def calculate_cluster_centers(data, assignments, cluster_centers, k):
    
    number_of_data = data.shape[0]
    number_of_dimensions = data.shape[1]
    
    
    centers_old = np.zeros(cluster_centers.shape) 
    centers_new = deepcopy(cluster_centers) 
    
    
    error = np.linalg.norm(centers_new - centers_old)
    while error != 0:
        h=0
    
        centers_old = deepcopy(centers_new)
        
        for i in range(k): 
            m=0
            f=0   
            total=np.zeros(data.shape)
            for datas in data:
                if(assignments[m]==i):
                    total[f]=datas
                    f=f+1
                m=m+1
            if(f!=0):
                centers_new[i]=np.mean(total[:f],axis=0)
                
        
            error = np.linalg.norm(centers_new - centers_old)
    return centers_new
   
   
    """
    Calculates cluster_centers such that their squared euclidean distance to the data assigned to
    them will be lowest.
    If none of the data points belongs to some cluster center, then assign it to its previous value.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param assignments: An (N, ) shaped numpy array with integers inside. They represent the cluster index
    every data assigned to.
    :param cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :param k: Number of clusters
    :return: A (K, D) shaped numpy array that contains the newly calculated cluster centers.
    """


def kmeans(data, initial_cluster_centers):
    k=initial_cluster_centers.shape[0]
    number_of_data = data.shape[0]
    summ=0
    
    old_assignments = np.zeros(number_of_data)
    
    #centers_old = np.zeros(initial_cluster_centers.shape) # to store old centers
    #centers_new = deepcopy(initial_cluster_centers) # Store new centers
    
    assignments1 = assign_clusters(data,initial_cluster_centers)
    
    while np.any(assignments1 != old_assignments): #I also think that make a while loop with center_old and center new comparision.  
    #But this give me a good result, I obtain good plot threfore I don't change it. It give an enough time to converge for center points     
        initial_cluster_centers = calculate_cluster_centers(data, assignments1, initial_cluster_centers, k)
        old_assignments = deepcopy(assignments1)
        assignments1 = assign_clusters(data,initial_cluster_centers)
    
    h=0
    for datas in data:
        summ = summ + np.linalg.norm(datas - initial_cluster_centers[assignments1[h]])**2
        h=h+1        
    
    
    
    return initial_cluster_centers, summ
    
    """
    Applies k-means algorithm.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param initial_cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :return: cluster_centers, objective_function
    cluster_center.shape is (K, D).
    objective function is a float. It is calculated by summing the squared euclidean distance between
    data points and their cluster centers.
    """
if __name__ == '__main__':
    clustering1 = np.load('hw2_data/kmeans/clustering1.npy') #k=2 #to not change clustering1 variable, I prefer changing the path
    #clustering2 = np.load('hw2_data/kmeans/clustering2.npy') #k=3
    #clustering3 = np.load('hw2_data/kmeans/clustering3.npy') #k=4
    #clustering4 = np.load('hw2_data/kmeans/clustering4.npy') #k=5
    list_of_obj=[]
    list_of_center=[]
    colors=['orange', 'blue', 'purple','black','red']
    x_axis=[1,2,3,4,5,6,7,8,9,10]
    for k in range(2,3):  #change it 1-10 while finding k, change it 3,4 or 4,5 or 5,6 for k = value and plot the data and center now it give the result for k=2
        objective_min=float("inf")
        for i in range(10):
            initial_cluster_centers = initialize_clusters(clustering1,k)
            resultant_cluster_centers,obj=kmeans(clustering1,initial_cluster_centers)
        
            if(objective_min>obj):
                objective_min=obj
                best_cluster_centers=resultant_cluster_centers
            
        #list_of_obj.append(objective_min)     #use for finding k
        #list_of_center.append(best_cluster_centers) #use for finding k
            
    category = assign_clusters(clustering1, best_cluster_centers)
    plt.clf()
    n = clustering1.shape[0]
    for i in range(n):
        #I prefer plt.scatter while plot data and center
        plt.scatter(clustering1[i, 0], clustering1[i,1], s=7, color = colors[int(category[i])])
    plt.scatter(best_cluster_centers[:,0], best_cluster_centers[:,1], marker='*', c='g', s=150)
    #plt.plot(x_axis, list_of_obj) #use for finding k
    
    plt.show()
        
    
    