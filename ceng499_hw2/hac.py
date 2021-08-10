import numpy as np
import matplotlib.pyplot as plt

def single_linkage(c1, c2):
    dist = float("inf")
    for p1 in c1:
        for p2 in c2:
            new_dist= np.linalg.norm(p1 - p2)
            if new_dist < dist:
                dist=new_dist 
                
    return dist
            
    
    """
    Given clusters c1 and c2, calculates the single linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """


def complete_linkage(c1, c2):
    dist = -1
    for p1 in c1:
        for p2 in c2:
            new_dist= np.linalg.norm(p1 - p2)
            if new_dist > dist:
                dist=new_dist 
                
    return dist

    """
    Given clusters c1 and c2, calculates the complete linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """


def average_linkage(c1, c2):
    dist=0
    for p1 in c1:
        for p2 in c2:
           dist= dist + np.linalg.norm(p1 - p2)
    dist=dist/(len(c1)*len(c2))
    return dist
    
    
    """
    Given clusters c1 and c2, calculates the average linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """


def centroid_linkage(c1, c2):
    
    column_sums_c1 = c1.sum(axis=0)
    column_sums_c2 = c2.sum(axis=0)
    c1_center=column_sums_c1/len(c1)
    c2_center=column_sums_c2/len(c2)
    distance = np.linalg.norm(c1_center - c2_center)
    return distance
    
    
    """
    Given clusters c1 and c2, calculates the centroid linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """


def hac(data, criterion, stop_length):
    global distance
    #print(np.shape(data))
    row,col = np.shape(data)

    distance=float("inf")
    data_list=[]
    
    data_list=[item for item in data]
    print(np.shape(data_list))
    while (len(data_list)!=stop_length):
        distance_table=np.zeros((len(data_list),len(data_list))) #for debug
        coordinat1=0
        coordinat2=0
        dist = float("inf")
        n=len(data_list)
        for i in range(n-1):
            for j in range(i+1,n):
                #print(np.shape(data_list[i])) #for debug
                x=data_list[i].reshape(-1,col)
                y=data_list[j].reshape(-1,col)
                #print(np.shape(x)) #for debug
                distance = criterion(x,y)
                if distance < dist:
                    coordinat1=i
                    coordinat2=j
                    dist=distance
                distance_table[i][j]=distance #for debug
        #print(type(data_list))  #for debug
        #print(data_list[coordinat1])#for debug
        #print(data_list[coordinat2])#for debug
        new_cluster=np.vstack([data_list[coordinat1],data_list[coordinat2]])
        data_list[coordinat1]=new_cluster
        del(data_list[coordinat2])
        n=len(data_list)
    return data_list
                
    
    
    
       
    """
    Applies hierarchical agglomerative clustering algorithm with the given criterion on the data
    until the number of clusters reaches the stop_length.
    :param data: An (N, D) shaped numpy array containing all of the data points.
    :param criterion: A function. It can be single_linkage, complete_linkage, average_linkage, or
    centroid_linkage
    :param stop_length: An integer. The length at which the algorithm stops.
    :return: A list of numpy arrays with length stop_length. Each item in the list is a cluster
    and a (Ni, D) sized numpy array.
    """
    
    
if __name__ == '__main__':
    
    data1 = np.load ('hw2_data/hac/data1.npy') #I prefer change the path instead of change data1 variable
    #data2 = np.load ('hw2_data/hac/data2.npy')
    #data3 = np.load ('hw2_data/hac/data3.npy')
    #data4 = np.load ('hw2_data/hac/data4.npy') 
     
    cluster1 = hac(data1,average_linkage,2) #change that part for other criterion
    positives=cluster1[0]
    negatives = cluster1[1]
    #notr1 = cluster1[2] #for dataset4
    #notr2 = cluster1[3] #for dataset4
    plt.clf()
    plt.plot(positives[:, 0], positives[:, 1], 'or')
    plt.plot(negatives[:, 0], negatives[:, 1], 'ok')
    #plt.plot(notr1[:, 0], notr1[:, 1], 'ob') # for dataset4
    #plt.plot(notr2[:, 0], notr2[:, 1], 'og') #for dataset4
    plt.show()
    
        
        
 
    
    