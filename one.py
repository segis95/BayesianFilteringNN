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
    print(len(linear_models))
    if (len(list_dimensions_behind) == 1):
        return linear_models
    
    #Parameters
    
    number_x_items = 10
    number_expectation_items = 20
    number_items_per_expectation = 10
    
    
    dim1 = list_dimensions_behind[-2]
    dim2 = list_dimensions_behind[-1]
    
    
    len_parameters_vector = dim1 * dim2 + dim2
    
    
    X_input_set =  lhs(dim1, number_x_items, criterion='center')
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
    #model = DecisionTreeRegressor(max_depth = 16)
    
    model.fit(X_input_set, x_to_expectation)
    
  
    return learn_network_for_one_x( Y, list_dimensions_behind[:-1] , [ list_dimensions_behind[-1] ] + list_dimensions_ahead, [model] + linear_models, mode_loss, activation_mode)
            
            

def calculate_prediction(X_in_, linear_models, list_dimensions_ahead, mode_loss, activation_mode, ind):
    
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
    parameters = model.predict([X_in])
    

    dim1 = len(X_in)
    dim2 = list_dimensions_ahead[0]
    
    
    W = parameters[0][:dim1 * dim2].reshape(dim1,dim2)
    
    b = parameters[0][dim1 * dim2:]
    
    X_next = X_in.dot(W) + b
    
    #print(X_in.shape, W.shape, b.shape, X_next.shape)

    
    return calculate_prediction(X_next, linear_models[1:], list_dimensions_ahead[1:], mode_loss, activation_mode, ind)

def calculate_optimal_coeffs(X_in, list_coeffs, linear_models, list_dimensions_ahead, activation_mode):
    
    if (len(linear_models) == 0):
       return list_coeffs
    
    model = linear_models[0]
    
    #Check here!!!
    parameters = model.predict([X_in])
    
    dim1 = len(X_in)
    dim2 = list_dimensions_ahead[0]
    
    W = parameters[0][:dim1 * dim2].reshape(dim1,dim2)
    
    b = parameters[0][dim1 * dim2:]
    
    X_next = activation_function(X_in.dot(W) + b, activation_mode)
    
    list_coeffs[0].append(W)
    list_coeffs[1].append(b)
    
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
    
    
    
    
def download_data_and_learn_all(net_architecture, mode_loss , activation_mode):
    
    #DataSetX = np.array([[0.1],[0.2],[0.3],[0.4],[0.5],[0.6],[0.7],[0.8],[0.9]])
    #DataSetY = np.array([0.0])
    
    DataSetX = np.linspace(-np.pi, np.pi, 10)
    DataSetX = np.array([[x] for x in DataSetX])
    DataSetY = np.array([[(1.0 + np.sin(t)) / 2.0] for t in DataSetX])
    
    
    dim0 = len(DataSetX[0])
    
    List_of_coeffs_for_different_pairs = []

    for i in range(len(DataSetY)):
        
        print("Working for******************************************************** ", i)
        models = learn_network_for_one_x(DataSetY[i], list_dimensions_behind = [dim0] + net_architecture, list_dimensions_ahead = [], linear_models = [], mode_loss = mode_loss, activation_mode = activation_mode)
        coeffs = calculate_optimal_coeffs(DataSetX[i], [[],[]], models,  net_architecture, activation_mode)
        List_of_coeffs_for_different_pairs.append(coeffs)
    
    coeffs_total = [[],[]]
    
    for i in range(len(List_of_coeffs_for_different_pairs[0][0])):
        W_list = [List_of_coeffs_for_different_pairs[j][0][i] for j in range(len(DataSetX))]
        W = sum(W_list)
        coeffs_total[0].append(W / len(DataSetX))#
        b_list = [List_of_coeffs_for_different_pairs[j][1][i] for j in range(len(DataSetX))]
        b = sum(b_list)
        coeffs_total[1].append(b / len(DataSetX))
    #
    Predictions1 = [calculate_prediction_from_coeffs(DataSetX[i], coeffs_total, mode_loss, activation_mode) for i in range(len(DataSetX))]
    
    Predictions2 = [calculate_prediction_from_coeffs(DataSetX[i], List_of_coeffs_for_different_pairs[i], mode_loss, activation_mode) for i in range(len(DataSetX))]
    plt.plot(DataSetX,[x[0] for x in DataSetY], DataSetX, Predictions1, 'o')
    plt.plot(DataSetX,[x[0] for x in DataSetY], DataSetX, Predictions2, 'o')
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



    
download_data_and_learn_all(net_architecture = [2,2,2,2,2,2,1], mode_loss = "reg" , activation_mode = "sigmoid")
print("!!!")



#print(get_train_set(3, 10).shape)