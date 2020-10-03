"""

i hope this code helps you to get the linear regression

you can use this code where you want

@author
Berkant DEMIRCI

contact
mail : berkantdemirci1905@gmail.com
instagram : berkant.py


This code establishes a linear relation between the price of a house and its square meter.

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as lr
import matplotlib.pyplot as plt

df = pd.read_csv("homeprice.csv")
x = df["squaremeter"].to_numpy()[:,np.newaxis]
y = df["price"].to_numpy()[:,np.newaxis]

plt.figure(figsize=(20,10))

xtrain , xtest , ytrain, ytest = tts(x , y, test_size = 0.2,random_state = 1) #spliting data frame (we use eighty percent of the data frame for training)

request = np.array(float(input("enter the size of the field that you want to know its price : "))).reshape(1,1) 

linear = lr() #assign an object that will learn 

def main() :
    linear.fit(xtrain,ytrain) #learning prosses (calculates predict line)
    ypredict = linear.predict(xtest) #the machine (linear object) predicts y column using xtest column
    point = linear.coef_*request + linear.intercept_ 
    # the point variable is result of y = mx + n equation (m = coef_ or inclination of equation, n = intercept_ or equation constant, y = the price that you want to know)
    predict = linear.predict(request) # finds the price that you want to know

    def printing():
        plt.subplot(2,2,1)
        plt.scatter(request,point,marker="s",c="orange") # sets the prices as point on the graph   
        plt.scatter(xtrain, ytrain) # sets the datas as the point on the graph (you can also type it like this "plt.scatter(xtest,ytest)")
        plt.plot(xtest,ypredict,"red") # shows the predict line on the graph (you can also type it like this  "plt.plot(xtrain,linear.predict(xtrain),"r")")
        plt.xlabel("Square Meters")
        plt.ylabel("Price")      # coordinate labels
        plt.title("House Square Meters Graph")    
        print(f"predict : {predict[0][0]}") # prints on the screen the price that you want to know
        plt.legend(["predict line","your point","test datas"],loc= "best") # indicates objects on the graph 
    
        def accuracyrate() :
            plt.subplot(2,2,2)
            plt.plot(ytest,"g",ypredict,"m")
            plt.title("Accuracy Rate")  
            plt.legend(["test datas","predict datas"],loc="best")
            plt.show()
        accuracyrate()
    printing()

main()