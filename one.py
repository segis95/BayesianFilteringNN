'''
Created on 

@author: Sergey
'''
import numpy as np
import sklearn.metrics as metrics
from pyDOE import lhs
import sklearn.linear_model
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn import preprocessing
import random
from sklearn import svm
import sklearn.metrics
from sklearn.neural_network import MLPClassifier
import time
from scipy.linalg import get_blas_funcs

def get_train_set(dim, number_of_steps):
    z = [[y] for y in np.linspace(-4, 4, number_of_steps)]
    x = [[y] for y in np.linspace(-4, 4, number_of_steps)]
    
    for i in range(dim - 1):
        x = [r + t for r in x for t in z]
    return np.array(x)

def activation_function(z, mode = "sigmoid"):
    
    if (mode == 'arctan'):
        return ((2.0 * np.arctan(z)/np.pi) + 1.0)/2.0
    if (mode == 'tanh'):
        return (1.0 + np.tanh(z))/2.0
    
    return 1.0/(1.0 + np.exp(-z))

  
def loss_function(y_predicted, y_true, mode_loss = "class"):
    
    if (mode_loss == "class"):
        return   - metrics.log_loss([y_true], y_predicted, labels = [0.0,1.0])
    else:
        return   - np.linalg.norm(y_predicted - y_true, 2)
    
    
def calculate_loss(X_in, y, linear_models, list_dimensions_ahead, mode_loss, activation_mode, ind):
    return loss_function(calculate_prediction(X_in, linear_models, list_dimensions_ahead, mode_loss, activation_mode, ind),y, mode_loss)


def learn_network_for_one_x(Y, list_dimensions_behind = [2,2], list_dimensions_ahead = [], linear_models = [], mode_loss = "reg", activation_mode = "sigma"):
    #class sklearn.preprocessing.MinMaxScaler
    #print(len(linear_models))
    if (len(list_dimensions_behind) == 1):
        return linear_models
    
    #Parameters
    
    number_x_items = 15
    number_expectation_items = 15
    number_items_per_expectation = 5
    
    
    dim1 = list_dimensions_behind[-2]
    dim2 = list_dimensions_behind[-1]
    
    
    len_parameters_vector = dim1 * dim2 + dim2
    
    
    X_input_set =  2.0 * lhs(dim1, number_x_items, criterion='center') - 1.0
    #X_input_set = get_train_set(dim1, 6)
    
    x_to_expectation = list()
    
    Loses_for_expectation = [0 for q in range(number_items_per_expectation)]
    
    for x_in in X_input_set:
        
        expectations = 4.0 * lhs(len_parameters_vector, number_expectation_items, criterion = 'maximin') - 2.0
        loses_expectations = dict()
        
        
        
        for expectation in expectations:
 
            vectors_parameters = np.random.multivariate_normal(expectation, np.identity(len_parameters_vector, float), number_items_per_expectation)
            
            
            for i in range(len(vectors_parameters)):
                
                W = vectors_parameters[i][:dim1 * dim2].reshape(dim1, dim2)
                
                b = vectors_parameters[i][dim1 * dim2:]
                
                
                
                
                X_next = x_in.dot(W)  + b
                
                if (len(list_dimensions_behind) == 2):
                    ind = 0
                else:
                    ind = 1
                #print(x_in.shape, W.shape, b.shape, X_next.shape)
                
                Loses_for_expectation[i] = calculate_loss(X_next, Y, linear_models, list_dimensions_ahead, mode_loss, activation_mode, ind)
                
                
                
            loses_expectations[np.array(Loses_for_expectation).mean()] = expectation
            
            

        max1 = max(loses_expectations.keys())
        
        #print(max1)   
        x_to_expectation.append(loses_expectations[max1])

           
        
    #print(dim1, dim2)    
    model = sklearn.linear_model.LinearRegression()
    #model = DecisionTreeRegressor(max_depth = 8)
    
    model.fit(X_input_set, x_to_expectation)
    
  
    return learn_network_for_one_x( Y, list_dimensions_behind[:-1] , [ list_dimensions_behind[-1] ] + list_dimensions_ahead, [model] + linear_models, mode_loss, activation_mode)
            
            

def calculate_prediction(X_in_, linear_models, list_dimensions_ahead, mode_loss, activation_mode, ind):
    begin_time = time.clock()
    if (len(list_dimensions_ahead) == 0) and (mode_loss == "reg"):
        return X_in_
    
    if (len(list_dimensions_ahead) == 0) and (mode_loss == "class"):
        return activation_function(X_in_, activation_mode)
    
    if ind > 0:
        
        X_in = activation_function(X_in_, activation_mode)

    else:
        X_in = X_in_.copy()
        ind = 1
    
    

    model = linear_models[0]

    #Check here!!!
    
    parameters = model.predict([X_in])#4e-5


    dim1 = len(X_in)
    dim2 = list_dimensions_ahead[0]
    
    
    #W = parameters[0][:dim1 * dim2].reshape(dim1,dim2)
    
    W = parameters[0][:dim1 * dim2].reshape(dim1, dim2)#1e-5


    
    b = parameters[0][dim1 * dim2:]
    

 
    X_next = X_in.dot(W) + b
    
    end_time = time.clock()
    if (end_time - begin_time > 1e-4):
        print("%.80f" % (end_time - begin_time))

    #print(W.dtype)

    #print(X_in.shape, W.shape, b.shape, X_next.shape)

    
    return calculate_prediction(X_next, linear_models[1:], list_dimensions_ahead[1:], mode_loss, activation_mode, ind)

def calculate_optimal_coeffs(X_in, list_coeffs, linear_models, list_dimensions_ahead, activation_mode):
    
    if (len(linear_models) == 0):
       return list_coeffs
    
    model = linear_models[0]
    
    #Check here!!!
    #print(len(X_in))
    parameters = model.predict([X_in])
    
    dim1 = len(X_in)
    dim2 = list_dimensions_ahead[0]
    
    W = parameters[0][:dim1 * dim2].reshape(dim1,dim2)
    
    b = parameters[0][dim1 * dim2:]
    
    X_next = activation_function(X_in.dot(W) + b, activation_mode)
    
    list_coeffs[0].append(W)
    list_coeffs[1].append(b)
    list_coeffs[2].append(X_in)
    list_coeffs[3].append(parameters[0])
    
    return calculate_optimal_coeffs(X_next, list_coeffs, linear_models[1:], list_dimensions_ahead[1:], activation_mode)
    
def calculate_prediction_from_coeffs(X_in_, list_coeffs, mode_loss, activation_mode):
    
    X = X_in_.copy()
    
    n = len(list_coeffs[0])
    
    for i in range(n - 1):
        X = activation_function(X.dot(list_coeffs[0][i]) + list_coeffs[1][i], mode = activation_mode) 
        
    if mode_loss == "reg":
        return X.dot(list_coeffs[0][n - 1]) + list_coeffs[1][n - 1]
    else:
        return activation_function( X.dot(list_coeffs[0][n - 1]) + list_coeffs[1][n - 1], mode = activation_mode) 
    
    
    
    
def download_data_and_learn_all_reg(net_architecture, mode_loss , activation_mode):
    
    #DataSetX = np.array([[0.1],[0.2],[0.3],[0.4],[0.5],[0.6],[0.7],[0.8],[0.9]])
    #DataSetY = np.array([0.0])
    
    DataSetX_0 = np.linspace(-np.pi, np.pi, 11)
    DataSetY = np.array([[(1.0 + np.sin(t)) / 2.0] for t in DataSetX_0])
    DataSetX = np.linspace(0.0, 1.0, 11)
    DataSetX = np.array([[x] for x in DataSetX])
    DataSetX1 = np.linspace(0, 1, 13)
    DataSetX1 = np.array([[x] for x in DataSetX1])
    DataSetX1_0 = np.linspace(-np.pi, np.pi, 13)
    

    
    
    
    dim0 = len(DataSetX[0])
    
    List_of_coeffs_for_different_pairs = []

    for i in range(len(DataSetY)):
        
        print("Working for******************************************************** ", i)
        
        models = learn_network_for_one_x(DataSetY[i], list_dimensions_behind = [dim0] + net_architecture, list_dimensions_ahead = [], linear_models = [], mode_loss = mode_loss, activation_mode = activation_mode)

        coeffs = calculate_optimal_coeffs(DataSetX[i], [[],[],[],[]], models,  net_architecture, activation_mode)
        List_of_coeffs_for_different_pairs.append(coeffs)
    
    coeffs_total = [[],[],[],[]]
    
    Generic_models = []
    
    for i in range(len(List_of_coeffs_for_different_pairs[0][0])):
        W_list = [List_of_coeffs_for_different_pairs[j][0][i] for j in range(len(DataSetX))]
        W = sum(W_list)
        coeffs_total[0].append(W / len(DataSetX))#
        b_list = [List_of_coeffs_for_different_pairs[j][1][i] for j in range(len(DataSetX))]
        b = sum(b_list)
        coeffs_total[1].append(b / len(DataSetX))
        
        XSetOnThisLayer = [List_of_coeffs_for_different_pairs[j][2][i] for j in range(len(DataSetX))]
        ParamSetOnThisLayer = [List_of_coeffs_for_different_pairs[j][3][i] for j in range(len(DataSetX))]
        #model = sklearn.linear_model.LinearRegression()
        model = DecisionTreeRegressor(max_depth = 8)
        #print(XSetOnThisLayer[0].shape)
        #print(ParamSetOnThisLayer[0].shape)
        model.fit(np.array(XSetOnThisLayer), np.array(ParamSetOnThisLayer))
        Generic_models.append(model)
        
    #
    Predictions1 = [calculate_prediction_from_coeffs(DataSetX1[i], coeffs_total, mode_loss, activation_mode) for i in range(len(DataSetX1))]
    
    Predictions2 = [calculate_prediction_from_coeffs(DataSetX[i], List_of_coeffs_for_different_pairs[i], mode_loss, activation_mode) for i in range(len(DataSetX))]
    Predictions3 = [calculate_prediction(DataSetX1[i], Generic_models,  net_architecture, mode_loss, activation_mode, 0) for i in range(len(DataSetX1))]
    plt.plot(DataSetX_0,[x[0] for x in DataSetY])
    #plt.plot(DataSetX, Predictions1, 'o')
    #plt.plot(DataSetX, Predictions2, 'o')
    plt.plot(DataSetX1_0, Predictions3, 'o')
    plt.title("[1,2,2,2,2,2,2,2,2,2,2,2,1]")
    plt.show()
    
    #print(coeffs[0][0].shape, coeffs[0][1].shape, coeffs[1][0].shape, coeffs[1][1].shape)
    #print( activation_function(activation_function(DataSetX[0].dot( coeffs[0][0]) + coeffs[1][0], mode= "sigmoid").dot( coeffs[0][1]) + coeffs[1][1], mode="sigmoid"))
    
    """
    print(calculate_prediction(DataSetX[0], models,  net_architecture, mode_loss, activation_mode, 0))

    print(calculate_prediction_from_coeffs(DataSetX[0], coeffs, mode_loss, activation_mode))
    
    print(calculate_prediction(DataSetX[1], models,  net_architecture, mode_loss, activation_mode, 0))
    
    print(calculate_prediction(DataSetX[2], models,  net_architecture, mode_loss, activation_mode, 0))
    
    print(calculate_prediction(DataSetX[3], models,  net_architecture, mode_loss, activation_mode, 0))
    
    print(calculate_prediction(DataSetX[4], models,  net_architecture, mode_loss, activation_mode, 0))
   
    print(calculate_prediction(DataSetX[5], models,  net_architecture, mode_loss, activation_mode, 0))
    
    print(calculate_prediction(DataSetX[6], models,  net_architecture, mode_loss, activation_mode, 0))
   
    print(calculate_prediction(DataSetX[7], models,  net_architecture, mode_loss, activation_mode, 0))
    """

def download_data_and_learn_all_class(net_architecture, mode_loss , activation_mode, dataset):
    
    [DataSetX, DataSetY, DataSetX1, DataSetY1] = download_data(dataset)
    
   
    #print(DataSetX[0])
    
    #DataSetX = np.linspace(-np.pi, np.pi, 11)
    #DataSetX = np.array([[x] for x in DataSetX])
    

    dim0 = len(DataSetX[0])
    
    
    List_of_coeffs_for_different_pairs_ones = []
    List_of_coeffs_for_different_pairs_zeros = []

    List_of_coeffs_for_different_pairs = []
        
    #print("Working for******************************************************** ", 0)

    models_zero = learn_network_for_one_x(np.array([0.0]), list_dimensions_behind = [dim0] + net_architecture, list_dimensions_ahead = [], linear_models = [], mode_loss = mode_loss, activation_mode = activation_mode)

    #print("Working for******************************************************** ", 1)
    models_one = learn_network_for_one_x(np.array([1.0]), list_dimensions_behind = [dim0] + net_architecture, list_dimensions_ahead = [], linear_models = [], mode_loss = mode_loss, activation_mode = activation_mode)
    
    X_ones = [i for i in range(len(DataSetX)) if (DataSetY[i][0] == 1.0)]
    X_zeros = [i for i in range(len(DataSetX)) if (DataSetY[i][0] == 0.0)]
    
    
    for i in X_ones:
        coeffs = calculate_optimal_coeffs(DataSetX[i], [[],[],[],[]], models_one,  net_architecture, activation_mode)
        List_of_coeffs_for_different_pairs_ones.append(coeffs)
        List_of_coeffs_for_different_pairs.append(coeffs)
        
    for i in X_zeros:
        coeffs = calculate_optimal_coeffs(DataSetX[i], [[],[],[],[]], models_zero,  net_architecture, activation_mode)
        List_of_coeffs_for_different_pairs_zeros.append(coeffs)
        List_of_coeffs_for_different_pairs.append(coeffs)
        
    coeffs_total_zeros = [[],[],[],[]]
    
    Generic_models_zeros = []
    
    coeffs_total_ones = [[],[],[],[]]
    
    Generic_models_ones = []
    
    Generic_models = []
    
    for i in range(len(List_of_coeffs_for_different_pairs_ones[0][0])):
        
        
        W_list = [List_of_coeffs_for_different_pairs_ones[j][0][i] for j in range(len(X_ones))]
        
        W = sum(W_list)
        coeffs_total_ones[0].append(W / len(X_ones))#
        b_list = [List_of_coeffs_for_different_pairs_ones[j][1][i] for j in range(len(X_ones))]
        b = sum(b_list)
        coeffs_total_ones[1].append(b / len(X_ones))
        
        XSetOnThisLayer = [List_of_coeffs_for_different_pairs_ones[j][2][i] for j in range(len(X_ones))]
        ParamSetOnThisLayer = [List_of_coeffs_for_different_pairs_ones[j][3][i] for j in range(len(X_ones))]
        model = sklearn.linear_model.LinearRegression()
        #model = DecisionTreeRegressor(max_depth = 5)
        #print(XSetOnThisLayer[0].shape)
        #print(ParamSetOnThisLayer[0].shape)
        model.fit(np.array(XSetOnThisLayer), np.array(ParamSetOnThisLayer))
        Generic_models_ones.append(model)    
        
    
    for i in range(len(List_of_coeffs_for_different_pairs_zeros[0][0])):
        W_list = [List_of_coeffs_for_different_pairs_zeros[j][0][i] for j in range(len(X_zeros))]
        W = sum(W_list)
        coeffs_total_zeros[0].append(W / len(X_zeros))#
        b_list = [List_of_coeffs_for_different_pairs_zeros[j][1][i] for j in range(len(X_zeros))]
        b = sum(b_list)
        coeffs_total_zeros[1].append(b / len(X_zeros))
        
        XSetOnThisLayer = [List_of_coeffs_for_different_pairs_zeros[j][2][i] for j in range(len(X_zeros))]
        ParamSetOnThisLayer = [List_of_coeffs_for_different_pairs_zeros[j][3][i] for j in range(len(X_zeros))]
        model = sklearn.linear_model.LinearRegression()
        #model = DecisionTreeRegressor(max_depth = 5)
        #print(XSetOnThisLayer[0].shape)
        #print(ParamSetOnThisLayer[0].shape)
        model.fit(np.array(XSetOnThisLayer), np.array(ParamSetOnThisLayer))
        Generic_models_zeros.append(model)
        
        
    for i in range(len(List_of_coeffs_for_different_pairs[0][0])):
            
        XSetOnThisLayer = [List_of_coeffs_for_different_pairs[j][2][i] for j in range(len(X_zeros) + len(X_ones))]
        ParamSetOnThisLayer = [List_of_coeffs_for_different_pairs[j][3][i] for j in range(len(X_zeros) + len(X_ones))]
        
        model = sklearn.linear_model.LinearRegression()
        #model = DecisionTreeRegressor(max_depth = 5)
        #print(XSetOnThisLayer[0].shape)
        #print(ParamSetOnThisLayer[0].shape)
        model.fit(np.array(XSetOnThisLayer), np.array(ParamSetOnThisLayer))
        Generic_models.append(model)

    
    #X_ones_test = [i for i in range(len(DataSetX1)) if (DataSetY1[i][0] == 1.0)]
    #X_zeros_test = [i for i in range(len(DataSetX1)) if (DataSetY1[i][0] == 0.0)]
     
    Zeros_border_mean = np.array([calculate_prediction_from_coeffs(DataSetX[i], coeffs_total_zeros, mode_loss, activation_mode) for i in X_ones]).mean()
    Ones_border_mean = np.array([calculate_prediction_from_coeffs(DataSetX[i], coeffs_total_ones, mode_loss, activation_mode) for i in X_zeros]).mean()
    Zeros_border_var = np.array([calculate_prediction_from_coeffs(DataSetX[i], coeffs_total_zeros, mode_loss, activation_mode) for i in X_ones]).var()
    Ones_border_var = np.array([calculate_prediction_from_coeffs(DataSetX[i], coeffs_total_ones, mode_loss, activation_mode) for i in X_zeros]).var()
    
    Front_zero = Zeros_border_mean - np.sqrt(Zeros_border_var)
    Front_ones = Ones_border_mean + np.sqrt(Ones_border_var)
    Predictions10 = [calculate_prediction_from_coeffs(DataSetX1[i], coeffs_total_zeros, mode_loss, activation_mode) for i in range(len(DataSetX1))]
    Predictions11 = [calculate_prediction_from_coeffs(DataSetX1[i], coeffs_total_ones, mode_loss, activation_mode) for i in range(len(DataSetX1))]
    
    
    #print(Predictions10[1:20])
    #print(Predictions11[1:20])
    #Predictions1 = [0 if Predictions10[i] < 1.0 - Predictions11[i] else 1 for i in range(len(DataSetX1)) ]
    Predictions1 = [pred(Predictions10[i], Predictions11[i], Front_zero, Front_ones, np.sqrt(Zeros_border_var), np.sqrt(Ones_border_var)) for i in range(len(DataSetX1))]
    
    
    
    #print("From mean:")
    #print(1.0 - np.abs(np.array(Predictions1) - np.array([x[0] for x in DataSetY1])).sum()/len(DataSetX1)) 

    #Predictions2 = [calculate_prediction_from_coeffs(DataSetX[i], List_of_coeffs_for_different_pairs[i], mode_loss, activation_mode) for i in range(len(DataSetX1))]
    
   
   
   
   
    Zeros_border_mean = np.median(np.array([calculate_prediction(DataSetX[i], Generic_models_zeros,  net_architecture, mode_loss, activation_mode, 0) for i in X_ones]).mean())
    Ones_border_mean = np.median(np.array([calculate_prediction(DataSetX[i], Generic_models_ones,  net_architecture, mode_loss, activation_mode, 0) for i in X_zeros]).mean())
    Zeros_border_var = np.array([calculate_prediction(DataSetX[i], Generic_models_zeros,  net_architecture, mode_loss, activation_mode, 0) for i in X_ones]).var()
    Ones_border_var = np.array([calculate_prediction(DataSetX[i], Generic_models_ones,  net_architecture, mode_loss, activation_mode, 0) for i in X_zeros]).var()
    
    Front_zero = Zeros_border_mean + 1.5 * np.sqrt(Zeros_border_var) 
    Front_ones = Ones_border_mean - 1.5 * np.sqrt(Ones_border_var) 
    
    Predictions30 = [calculate_prediction(DataSetX[i], Generic_models_zeros,  net_architecture, mode_loss, activation_mode, 0) for i in range(len(DataSetX))]
    Predictions31 = [calculate_prediction(DataSetX[i], Generic_models_ones,  net_architecture, mode_loss, activation_mode, 0) for i in range(len(DataSetX))]
    
    Predictions30test = [calculate_prediction(DataSetX1[i], Generic_models_zeros,  net_architecture, mode_loss, activation_mode, 0) for i in range(len(DataSetX1))]
    Predictions31test = [calculate_prediction(DataSetX1[i], Generic_models_ones,  net_architecture, mode_loss, activation_mode, 0) for i in range(len(DataSetX1))]
    
    #for "regression" method
    Predictions3R = [calculate_prediction(DataSetX1[i], Generic_models,  net_architecture, mode_loss, activation_mode, 0) for i in range(len(DataSetX1))]
    Predictions3R = np.array([1.0 if (Predictions3R[i] >= 0.5) else 0.0 for i in range(len(Predictions3R))  ])
    
    svm_train = np.array([[Predictions30[i][0], Predictions31[i][0]] for i in range(len(DataSetX))])
    svm_test = np.array([[Predictions30test[i][0], Predictions31test[i][0]] for i in range(len(DataSetX1))])
    clf = svm.NuSVC( kernel = 'poly', degree = 7)
    clf.fit(svm_train, DataSetY[:,0])
    
    
    Z = clf.predict(svm_test)
    

    #print(Z)
    #plt.plot([Predictions30test[i] for i in range(len(DataSetY1)) if Z[i] == 0.0], [Predictions31test[i] for i in range(len(DataSetY1)) if Z[i] == 0.0],'k^')
    #plt.plot([Predictions30test[i] for i in range(len(DataSetY1)) if Z[i] == 1.0], [Predictions31test[i] for i in range(len(DataSetY1)) if Z[i] == 1.0],'g^')
    
    plt.plot([Predictions30[i] for i in range(len(DataSetY)) if DataSetY[i] == 0.0], [Predictions31[i] for i in range(len(DataSetY)) if DataSetY[i] == 0.0],'ro', label = '0')
    plt.plot([Predictions30[i] for i in range(len(DataSetY)) if DataSetY[i] == 1.0], [Predictions31[i] for i in range(len(DataSetY)) if DataSetY[i] == 1.0],'bo', label = '1')
    Predictions3 = [pred(Predictions30[i], Predictions31[i], Front_zero, Front_ones, np.sqrt(Zeros_border_var), np.sqrt(Ones_border_var)) for i in range(len(DataSetX1))]
    
    plt.legend()
    
    clf = MLPClassifier(activation='tanh', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,5,5,5,5))
    
    clf.fit(DataSetX, DataSetY[:,0])
    
    prediction = clf.predict(DataSetX1)
    
    print(str(sklearn.metrics.accuracy_score(DataSetY1[:,0], prediction, normalize = True)) + ' ' + str(sklearn.metrics.accuracy_score(DataSetY1, Predictions3R, normalize = True)) + ' ' + str(sklearn.metrics.accuracy_score(DataSetY1, Z, normalize = True)))
    
    
    #print("From linear models:")
    #print(DataSetY1[:,0])
    #print(1.0 - np.abs(Z - DataSetY1[:,0]).sum()/len(DataSetY1) )
    #print(sklearn.metrics.accuracy_score(DataSetY1, Z, normalize = True))
    #print(Predictions3R)
    #print(sklearn.metrics.accuracy_score(DataSetY1, Predictions3R, normalize = True))
    #print(len(Predictions3))
    #print(Predictions30[1:20])
    #print(Predictions31[1:20])
    #print(len([i for i in range(len(DataSetX1)) if Predictions3[i] == 1.0 and DataSetY1[i][0] == 0.0]))
    
    #plt.plot([Predictions30[i] for i in range(len(DataSetY)) if DataSetY[i] == 0.0], [Predictions31[i] for i in range(len(DataSetY)) if DataSetY[i] == 0.0],'ro')
    #plt.plot([Predictions30[i] for i in range(len(DataSetY)) if DataSetY[i] == 1.0], [Predictions31[i] for i in range(len(DataSetY)) if DataSetY[i] == 1.0],'bo')

    #print(Predictions30)
    #plt.plot(Predictions11,[1 for i in range(len(Predictions11))],'o')
    #plt.plot(Predictions2,[x[0] for x in DataSetY],'o')
    #plt.plot(Predictions3,[x[0] for x in DataSetY1],'o')
    #///plt.plot([0.5 for i in range(11)],np.linspace(0,1, 11))
    #plt.plot(DataSetX, Predictions2, 'o')
    #plt.plot(DataSetX1, Predictions3, 'o')
    plt.ylim(min(Predictions31),max(Predictions31))
    plt.xlim(min(Predictions30),max(Predictions30))
    plt.savefig("3.jpg")
    #plt.show()
    '''
    f = open("pima-indians-diabetes.data.txt",'r')
    l = f.readlines()
    l = [l[i].split("\n") for i in range(100)]
    l = [l[i][0].split(',') for i in range(len(l))]
    l = [[float(l[i][j]) for j in range(len(l[i]))] for i in range(len(l))]
    
    f.close()
    Train_x = np.array([l[i][:8] for i in range(len(l))])
    DataSetY = np.array([ [l[i][8]] for i in range(len(l))])
    #Train_y = ( np.multiply(Train_x,Train_x) * np.matrix([[1],[1]])) / 2.0
    
    #we scale dataset
    normalizer = preprocessing.Normalizer().fit(Train_x);
    #Train_x_scaled = np.array(Train_x.copy())
    DataSetX = normalizer.transform(Train_x);
    
    #DataSetX = np.linspace(-np.pi, np.pi, 11)
    #DataSetX = np.array([[x] for x in DataSetX])
    

    dim0 = len(DataSetX[0])
    
    
    
    List_of_coeffs_for_different_pairs = []

    for i in range(len(DataSetY)):
        
        print("Working for******************************************************** ", i)
        models = learn_network_for_one_x(DataSetY[i], list_dimensions_behind = [dim0] + net_architecture, list_dimensions_ahead = [], linear_models = [], mode_loss = mode_loss, activation_mode = activation_mode)
        coeffs = calculate_optimal_coeffs(DataSetX[i], [[],[],[],[]], models,  net_architecture, activation_mode)
        List_of_coeffs_for_different_pairs.append(coeffs)
    
    coeffs_total = [[],[],[],[]]
    
    Generic_models = []
    
    for i in range(len(List_of_coeffs_for_different_pairs[0][0])):
        W_list = [List_of_coeffs_for_different_pairs[j][0][i] for j in range(len(DataSetX))]
        W = sum(W_list)
        coeffs_total[0].append(W / len(DataSetX))#
        b_list = [List_of_coeffs_for_different_pairs[j][1][i] for j in range(len(DataSetX))]
        b = sum(b_list)
        coeffs_total[1].append(b / len(DataSetX))
        
        XSetOnThisLayer = [List_of_coeffs_for_different_pairs[j][2][i] for j in range(len(DataSetX))]
        ParamSetOnThisLayer = [List_of_coeffs_for_different_pairs[j][3][i] for j in range(len(DataSetX))]
        #model = sklearn.linear_model.LinearRegression()
        model = DecisionTreeRegressor(max_depth = 7)
        #print(XSetOnThisLayer[0].shape)
        #print(ParamSetOnThisLayer[0].shape)
        model.fit(np.array(XSetOnThisLayer), np.array(ParamSetOnThisLayer))
        Generic_models.append(model)
        
    f = open("pima-indians-diabetes.data.txt",'r')
    l = f.readlines()
    l = [l[i].split("\n") for i in range(100,len(l))]
    l = [l[i][0].split(',') for i in range(len(l))]
    l = [[float(l[i][j]) for j in range(len(l[i]))] for i in range(len(l))]
    
    Train_x = np.array([l[i][:8] for i in range(len(l))])
    DataSetY1 = np.array([ [l[i][8]] for i in range(len(l))])
    DataSetX1 = normalizer.transform(Train_x);
    f.close()
    
    Predictions3 = [calculate_prediction(DataSetX1[i], Generic_models,  net_architecture, mode_loss, activation_mode, 0) for i in range(len(DataSetX1))]
    Predictions3 = [1.0 if Predictions3[i] > 0.5 else 0.0 for i in range(len(Predictions3))]
    print(np.abs(np.array(Predictions3) - np.array([x[0] for x in DataSetY1])).sum())
    plt.plot(Predictions3,[x[0] for x in DataSetY1],'o')
    
    plt.ylim(-2,2)
    plt.show()
    '''
    
def pred(x, y, front_zero, front_one, div_left, div_right):
    
    if (x <= front_zero) and (y >= front_one):
        if ((front_zero - x) > (y - front_one)):
            #print("RANDOOOOOM1")
            return 0.0
            
        else:
            #print("RANDOOOOOM2")
            return 1.0
        
    if (x <= front_zero) :
        return 0.0
    if (y >= front_one):
        return 1.0
    
    if ((x - front_zero) < (front_one - y)):#div_right):
        return 0.0
    else:
        return 1.0
    
    
    
    


def download_data(n):
    if (n == 0):
        f = open("pima-indians-diabetes.data.txt",'r')
        l = f.readlines()
        s1 = set(np.random.permutation([i  for i in range(len(l))])[:400])
        l = [l[i].split("\n") for i in s1]
        l = [l[i][0].split(',') for i in range(len(l))]
        l = [[float(l[i][j]) for j in range(len(l[i]))] for i in range(len(l))]
        
        f.close()
        Train_x = np.array([l[i][:8] for i in range(len(l))])
        DataSetY = np.array([ [l[i][8]] for i in range(len(l))])
        #Train_y = ( np.multiply(Train_x,Train_x) * np.matrix([[1],[1]])) / 2.0
            #we scale dataset
        normalizer = preprocessing.Normalizer().fit(Train_x);
        #DataSetX = np.array(Train_x.copy())
        DataSetX = normalizer.transform(Train_x);
           
        f = open("pima-indians-diabetes.data.txt",'r')
        l = f.readlines() 
        s2 = set(np.random.permutation([i  for i in range(len(l))]))
        s2 = s2 - s1
        l = [l[i].split("\n") for i in s2]
        l = [l[i][0].split(',') for i in range(len(l))]
        l = [[float(l[i][j]) for j in range(len(l[i]))] for i in range(len(l))]
        f.close()
        Train_x = np.array([l[i][:8] for i in range(len(l))])
        DataSetY1 = np.array([ [l[i][8]] for i in range(len(l))])
        DataSetX1 = normalizer.transform(Train_x);
        
        
    if (n == 1):
        f = open("diabetes.txt",'r')
        l = f.readlines()
        
        s1 = set(np.random.permutation([i  for i in range(len(l))])[:400])
        l = [l[i].split("\n") for i in s1]
        
        l = [l[i][0].split(' ') for i in range(len(l))]
        l = [x for x in l if (len(x) == 10)]
        
        DataSetY = np.array([ [1.0] if float(l[i][0]) == 1.0 else [0.0] for i in range(len(l))])
        
        l = np.array([[(l[i][j].split(':')) for j in range(1,len(l[i]) - 1)] for i in range(len(l))])
        
        l = [ np.array([ float(l[i][j][1]) for j in range(len(l[i]) ) ]) for i in range(len(l)) ]

        
        DataSetX = np.array(l) 
        
        f.close()
        
        f = open("diabetes.txt",'r')
        l = f.readlines()
        s2 = set(np.random.permutation([i  for i in range(len(l))]))
        s2 = s2 - s1
        l = [l[i].split("\n") for i in s2]
        l = [l[i][0].split(' ') for i in range(len(l))]
        l = [x for x in l if (len(x) == 10)]
        
        DataSetY1 = np.array([ [1.0] if float(l[i][0]) == 1.0 else [0.0]  for i in range(len(l))])
        l = np.array([[(l[i][j].split(':')) for j in range(1,len(l[i]) - 1)] for i in range(len(l))])
        l = [ np.array([ float(l[i][j][1]) for j in range(len(l[i])) ]) for i in range(len(l)) ]
        DataSetX1 = np.array(l) 
        f.close()
        
    if (n == 2):
        f = open("breast.txt",'r')
        l = f.readlines()
        
        s1 = set(np.random.permutation([i  for i in range(len(l))])[:350])
        l = [l[i].split("\n") for i in s1]
        
        l = [l[i][0].split(' ') for i in range(len(l))]
        
        l = [x for x in l if (len(x) == 12)]
        
        DataSetY = np.array([ [1.0] if float(l[i][0]) == 4.0 else [0.0] for i in range(len(l))])
        
        l = np.array([[(l[i][j].split(':')) for j in range(1,len(l[i]) - 1)] for i in range(len(l))])
        
        l = [ np.array([ float(l[i][j][1]) for j in range(len(l[i]) ) ]) for i in range(len(l)) ]

        
        DataSetX = np.array(l) 
        
        f.close()
        
        f = open("breast.txt",'r')
        l = f.readlines()
        s2 = set(np.random.permutation([i  for i in range(len(l))]))
        s2 = s2 - s1
        
        l = [l[i].split("\n") for i in s2]
        l = [l[i][0].split(' ') for i in range(len(l))]
        l = [x for x in l if (len(x) == 12)]
        
        DataSetY1 = np.array([ [1.0] if float(l[i][0]) == 4.0 else [0.0]  for i in range(len(l))])
        l = np.array([[(l[i][j].split(':')) for j in range(1,len(l[i]) - 1)] for i in range(len(l))])
        l = [ np.array([ float(l[i][j][1]) for j in range(len(l[i])) ]) for i in range(len(l)) ]
        DataSetX1 = np.array(l) 
        f.close()
        
    if (n == 3):
        f = open("fouclass.txt",'r')
        l = f.readlines()
        
        s1 = set(np.random.permutation([i  for i in range(len(l))])[:550])
        l = [l[i].split("\n") for i in s1]
        
        l = [l[i][0].split(' ') for i in range(len(l))]
        
        l = [x for x in l if (len(x) == 4)]
        
        DataSetY = np.array([ [1.0] if float(l[i][0]) == 1.0 else [0.0] for i in range(len(l))])
        
        l = np.array([[(l[i][j].split(':')) for j in range(1,len(l[i]) - 1)] for i in range(len(l))])
        
        l = [ np.array([ float(l[i][j][1]) for j in range(len(l[i]) ) ]) for i in range(len(l)) ]

        
        DataSetX = np.array(l) 
        
        f.close()
        
        f = open("fouclass.txt",'r')
        l = f.readlines()
        s2 = set(np.random.permutation([i  for i in range(len(l))]))
        s2 = s2 - s1
        
        l = [l[i].split("\n") for i in s2]
        l = [l[i][0].split(' ') for i in range(len(l))]
        l = [x for x in l if (len(x) == 4)]
        
        DataSetY1 = np.array([ [1.0] if float(l[i][0]) == 1.0 else [0.0]  for i in range(len(l))])
        l = np.array([[(l[i][j].split(':')) for j in range(1,len(l[i]) - 1)] for i in range(len(l))])
        l = [ np.array([ float(l[i][j][1]) for j in range(len(l[i])) ]) for i in range(len(l)) ]
        DataSetX1 = np.array(l) 
        f.close()

    if (n == 4):
        f = open("german.txt",'r')
        l = f.readlines()
        
        s1 = set(np.random.permutation([i  for i in range(len(l))])[:900])
        l = [l[i].split("\n") for i in s1]
        
        l = [l[i][0].split(' ') for i in range(len(l))]
        
        l = [x for x in l if (len(x) == 26)]
        
        DataSetY = np.array([ [1.0] if float(l[i][0]) == 1.0 else [0.0] for i in range(len(l))])
        
        l = np.array([[(l[i][j].split(':')) for j in range(1,len(l[i]) - 1)] for i in range(len(l))])
        
        l = [ np.array([ float(l[i][j][1]) for j in range(len(l[i]) ) ]) for i in range(len(l)) ]

        
        DataSetX = np.array(l) 
        
        f.close()
        
        f = open("german.txt",'r')
        l = f.readlines()
        s2 = set(np.random.permutation([i  for i in range(len(l))]))
        s2 = s2 - s1
        
        l = [l[i].split("\n") for i in s2]
        l = [l[i][0].split(' ') for i in range(len(l))]
        l = [x for x in l if (len(x) == 26)]
        
        DataSetY1 = np.array([ [1.0] if float(l[i][0]) == 1.0 else [0.0]  for i in range(len(l))])
        l = np.array([[(l[i][j].split(':')) for j in range(1,len(l[i]) - 1)] for i in range(len(l))])
        l = [ np.array([ float(l[i][j][1]) for j in range(len(l[i])) ]) for i in range(len(l)) ]
        DataSetX1 = np.array(l) 
        f.close()
    
    if (n == 5):
        f = open("heart.txt",'r')
        l = f.readlines()
        
        s1 = set(np.random.permutation([i  for i in range(len(l))])[:170])
        l = [l[i].split("\n") for i in s1]
        
        l = [l[i][0].split(' ') for i in range(len(l))]
        
        l = [x for x in l if (len(x) == 15)]
        
        DataSetY = np.array([ [1.0] if float(l[i][0]) == 1.0 else [0.0] for i in range(len(l))])
        
        l = np.array([[(l[i][j].split(':')) for j in range(1,len(l[i]) - 1)] for i in range(len(l))])
        
        l = [ np.array([ float(l[i][j][1]) for j in range(len(l[i]) ) ]) for i in range(len(l)) ]

        
        DataSetX = np.array(l) 
        
        f.close()
        
        f = open("heart.txt",'r')
        l = f.readlines()
        s2 = set(np.random.permutation([i  for i in range(len(l))]))
        s2 = s2 - s1
        
        l = [l[i].split("\n") for i in s2]
        l = [l[i][0].split(' ') for i in range(len(l))]
        l = [x for x in l if (len(x) == 15)]
        
        DataSetY1 = np.array([ [1.0] if float(l[i][0]) == 1.0 else [0.0]  for i in range(len(l))])
        l = np.array([[(l[i][j].split(':')) for j in range(1,len(l[i]) - 1)] for i in range(len(l))])
        l = [ np.array([ float(l[i][j][1]) for j in range(len(l[i])) ]) for i in range(len(l)) ]
        DataSetX1 = np.array(l) 
        f.close()
    
    if (n == 6):
        f = open("liver.txt",'r')
        l = f.readlines()
        
        s1 = set(np.random.permutation([i  for i in range(len(l))])[:100])
        l = [l[i].split("\n") for i in s1]
        
        l = [l[i][0].split(' ') for i in range(len(l))]
        
        l = [x for x in l if (len(x) == 7)]
        
        DataSetY = np.array([ [1.0] if float(l[i][0]) == 1.0 else [0.0] for i in range(len(l))])
        
        l = np.array([[(l[i][j].split(':')) for j in range(1,len(l[i]) - 1)] for i in range(len(l))])
        
        l = [ np.array([ float(l[i][j][1]) for j in range(len(l[i]) ) ]) for i in range(len(l)) ]

        
        DataSetX = np.array(l) 
        
        f.close()
        
        f = open("liver.txt",'r')
        l = f.readlines()
        s2 = set(np.random.permutation([i  for i in range(len(l))]))
        s2 = s2 - s1
        
        l = [l[i].split("\n") for i in s2]
        l = [l[i][0].split(' ') for i in range(len(l))]
        l = [x for x in l if (len(x) == 7)]
        
        DataSetY1 = np.array([ [1.0] if float(l[i][0]) == 1.0 else [0.0]  for i in range(len(l))])
        l = np.array([[(l[i][j].split(':')) for j in range(1,len(l[i]) - 1)] for i in range(len(l))])
        l = [ np.array([ float(l[i][j][1]) for j in range(len(l[i])) ]) for i in range(len(l)) ]
        DataSetX1 = np.array(l) 
        f.close()
    
    return [DataSetX, DataSetY, DataSetX1, DataSetY1]


def network(c):
    
   
    from sklearn.neural_network import MLPClassifier
    
    [DataSetX, DataSetY, DataSetX1, DataSetY1] = download_data(c)
   
    clf = MLPClassifier(activation='tanh', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,5,5,5,5))
    
    clf.fit(DataSetX, DataSetY[:,0])
    
    prediction = clf.predict(DataSetX1)
    
    print( str(sklearn.metrics.accuracy_score(DataSetY1[:,0], prediction, normalize = True)))
        
    
    #print(clf.coefs_[0])

#download_data_and_learn_all_reg(net_architecture = [2,2,2,2,2,2,2,2,2,2,2,1], mode_loss = "reg" , activation_mode = "sigmoid")

#print(get_train_set(3, 10).shape)
for j in [5]:
    print("This is for" + str(j))
    for i in range(1):
        #print(i)
        #network(j)
        time.sleep(1)
        try:
            download_data_and_learn_all_class(net_architecture = [5,5,5,5,5,1], mode_loss = "class" , activation_mode = "arctan",dataset = j)
        except:
            print('Caught')
            i = i - 1

        #print("!!!")


