import numpy as np
import matplotlib.pyplot as plt
def calculate_distances(train_data, test_datum):
    i=0
    distance_table=np.zeros(len(train_data))
    for data in train_data:
        distance = np.linalg.norm(test_datum - data)
        distance_table[i]=distance
        i=i+1
    return distance_table
    
    """
    Calculates euclidean distances between test_datum and every train_data
    :param train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param test_datum: A (D, ) shaped numpy array
    :return: An (N, ) shaped numpy array that contains distances
    """


def majority_voting(distances, labels, k):
    
    list_of_labels=[]
    index1 = np.argpartition(distances, k)
    index = index1[:k]
    for i in range(len(index)):
        list_of_labels.append(labels[index[i]])
    majority_class = np.bincount(list_of_labels).argmax()
    return majority_class
         
    
    
    """
    Applies majority voting. If there are more then one major class, returns the smallest label.
    :param distances: An (N, ) shaped numpy array that contains distances
    :param labels: An (N, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :return: An integer. The label of the majority class.
    """


def knn(train_data, train_labels, test_data, test_labels, k):
    i = 0
    accuracy=0
    for test_datum in test_data:
        distances = calculate_distances(train_data, test_datum)
        majority_class= majority_voting(distances, train_labels, k)
        if(majority_class==test_labels[i]):
            accuracy=accuracy+1
        i=i+1
    accuracy=accuracy/(test_data.shape[0])
    return accuracy
    """
    Calculates accuracy of knn on test data using train_data.
    :param train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param train_labels: An (N, ) shaped numpy array that contains labels
    :param test_data: An (M, D) shaped numpy array where M is the number of examples
    and D is the dimension of the data
    :param test_labels: An (M, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :return: A float. The calculated accuracy.
    """


def split_train_and_validation(whole_train_data, whole_train_labels, validation_index, k_fold):
    
    whole_train_data_length = whole_train_data.shape[0]
    validation_length = int( whole_train_data_length/k_fold)
    validation_index=validation_length*(validation_index)
    new_train_data=np.zeros((whole_train_data.shape[0]-validation_length,whole_train_data.shape[1]))
    new_train_labels=np.zeros((whole_train_labels.shape[0]-validation_length))
    validation_index_end = validation_index+validation_length
    validation_data=whole_train_data[validation_index:validation_index_end]
    validation_labels=whole_train_labels[validation_index:validation_index_end]
    if validation_index !=0:
        new_train_data[0:validation_index]=whole_train_data[0 : validation_index]
        new_train_data[validation_index:whole_train_data_length-validation_length]=whole_train_data[validation_index_end:whole_train_data_length]
        new_train_labels[0:validation_index]=whole_train_labels[0 : validation_index]
        new_train_labels[validation_index:whole_train_data_length-validation_length]=whole_train_labels[validation_index_end:whole_train_data_length]
    else:
        new_train_data=whole_train_data[validation_index_end:whole_train_data_length]
        new_train_labels=whole_train_labels[validation_index_end:whole_train_data_length]
    

    return new_train_data, new_train_labels, validation_data, validation_labels
    """
    Splits training dataset into k and returns the validation_indexth one as the
    validation set and others as the training set. You can assume k_fold divides N.
    :param whole_train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param whole_train_labels: An (N, ) shaped numpy array that contains labels
    :param validation_index: An integer. 0 <= validation_index < k_fold. Specifies which fold
    will be assigned as validation set.
    :param k_fold: The number of groups that the whole_train_data will be divided into.
    :return: train_data, train_labels, validation_data, validation_labels
    train_data.shape is (N-N/k_fold, D).
    train_labels.shape is (N-N/k_fold, ).
    validation_data.shape is (N/k_fold, D).
    validation_labels.shape is (N/k_fold, ).
    """


def cross_validation(whole_train_data, whole_train_labels, k, k_fold):
    accuracy=0
    global new_acc
    for i in range(k_fold):
        td,tl,vd,vl = split_train_and_validation(whole_train_data,whole_train_labels,i,k_fold)
        new_acc = knn(td,tl,vd,vl,k)
        accuracy=accuracy + new_acc
    average_accuracy=accuracy/k_fold
    return average_accuracy
        
    """
    Applies k_fold cross-validation and averages the calculated accuracies.
    :param whole_train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param whole_train_labels: An (N, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :param k_fold: An integer.
    :return: A float. Average accuracy calculated.
    """
if __name__ == '__main__':

    train_data = np.load ('hw2_data/knn/train_data.npy')
    train_labels = np.load ('hw2_data/knn/train_labels.npy')
    test_data = np.load ( 'hw2_data/knn/test_data.npy' )
    test_labels = np.load ( 'hw2_data/knn/test_labels.npy' )
    accuracy_list=[]
    k_list=[i for i in range(1,200) ]
    global accuracy
    best_accuracy = -1
    global best_k
    global test_accuracy
    for i in range(1,200):
        accuracy = cross_validation(train_data,train_labels,i,10)
        if(accuracy>best_accuracy):
            best_accuracy=accuracy
            best_k=i
        accuracy_list.append(accuracy)
    print("BEST k is {}".format(best_k)) 
    
    test_accuracy =  cross_validation(test_data,test_labels,best_k,10)
    print("TEST ACCURACY for k = {} is {}".format(best_k, test_accuracy))
    
     
    plt.clf()
    plt.plot(k_list,accuracy_list)
    plt.show()
    
    
