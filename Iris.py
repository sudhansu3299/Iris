#########------------    Classification of IRIS flower  using ML     --------#############

##load iris data to program
from sklearn.datasets import load_iris
iris=load_iris()
print(iris)
##separate input and output
X=iris.data ##numpy array
Y=iris.target ##numpy array
print(X)
print(Y)
print(X.shape) #(150,4)
print(Y.shape)  #(150,)

################################
## split the dataset for training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=10)  ## random state is used for making fixed training and testing dataset
print(X_train.shape) #(120,4)
print(Y_train.shape) #(120,)
print(X_test.shape)  #(30,4)
print(Y_test.shape)  #(30,)
###########################

############ defining the functions#####################
def knn():
    ##create  a model ##KNN
    ## K Nearest Neighbors Algorithm
    global acc_knn
    global K   #K is declared global as it can be accessed later when it turns out be the best algorithm
    from sklearn.neighbors import KNeighborsClassifier
    K=KNeighborsClassifier(n_neighbors=5)
    ## Train the model by training dataset
    K.fit(X_train,Y_train)
    ## Test the model by testing data
    Y_pred=K.predict(X_test)
    ## Find Accuracy
    from sklearn.metrics import accuracy_score
    acc_knn=accuracy_score(Y_test,Y_pred)
    acc_knn=round(acc_knn*100,2)
    m.showinfo(title="Accuracy Result",message="Accuracy of KNN is"+str(acc_knn)+"%")
    ###predict for a new flower
    ##print(K.predict([[6,4,3,4]]))  #ans:2  ## here we are entering data randomly in cms. to predict the flower.
def lg():
    ######create a model using Logistic Regression (LG)
    global acc_lg
    global L
    from sklearn.linear_model import LogisticRegression
    L=LogisticRegression(solver='liblinear',multi_class='auto')  ## written to stop warning
    L.fit(X_train,Y_train)
    Y_pred=L.predict(X_test)
    from sklearn.metrics import accuracy_score
    acc_lg=accuracy_score(Y_test,Y_pred)
    acc_lg=round(acc_lg*100,2)
    m.showinfo(title="Accuracy Result",message="Accuracy of Logistic Regression is"+str(acc_lg)+"%")
  
def dt():
    ######create a model using Decision Tree   (DT)
    global acc_dt
    global D
    from sklearn.tree import DecisionTreeClassifier
    D=DecisionTreeClassifier()
    D.fit(X_train,Y_train)
    Y_pred=D.predict(X_test)
    from sklearn.metrics import accuracy_score
    acc_dt=accuracy_score(Y_test,Y_pred)
    acc_dt=round(acc_dt*100,2)
    m.showinfo(title="Accuracy Result",message="Accuracy of Decision Tree"+str(acc_dt)+"%")
def nb():
    global acc_nb
    global N
    from sklearn.naive_bayes import GaussianNB
    N=GaussianNB()
    N.fit(X_train,Y_train)
    Y_pred=N.predict(X_test)
    from sklearn.metrics import accuracy_score
    acc_nb=accuracy_score(Y_test,Y_pred)
    acc_nb=round(acc_nb*100,2)
    m.showinfo(title="Accuracy Result",message="The accuracy of naive bayes is"+str(acc_nb)+"%")
    #print("accuracy score in Naive Bayes is",acc_nb,"%")

def compare():
    ## drwaing a bar plot
    import matplotlib.pyplot as plt
    model=["KNN","LG","DT","NB"]
    accuracy=[acc_knn,acc_lg,acc_dt,acc_nb]
    plt.bar(model,accuracy,color=["orange","blue","green","yellow"])
    plt.xlabel("MODELS")
    plt.ylabel("ACCURACY")
    plt.show()
def submit():
    sl=float(v1.get())
    sw=float(v2.get())
    pl=float(v3.get())
    pw=float(v4.get())
    A={K:acc_knn,L:acc_lg,D:acc_dt,N:acc_nb}
    #model=max(A).getkey()
    result=N.predict([[sl,sw,pl,pw]])
    if result==0:
        flower="setosa"
    elif result==1:
        flower="versicolor"
    else:
        flower="virginica"
    m.showinfo(title="IRIS FLOWER",message=flower)
def reset():
    v1.set("")
    v2.set("")
    v3.set("")
    v4.set("")

###########---------------------      Design of GUI     ----------------- for this project #########
from tkinter import *
import tkinter.messagebox as m
w=Tk()
w.configure(bg="cyan")
w.title("Comparing ML Algorithms for IRIS Project")
v1=StringVar()
v2=StringVar()
v3=StringVar()
v4=StringVar()
b1=Button(w,text="KNN", font=('arial',15,'bold'),command=knn)
b1.grid(row=1,column=1,columnspan=1)
b2=Button(w,text="LG", font=('arial',15,'bold'),command=lg)
b2.grid(row=2,column=1,columnspan=1)
b3=Button(w,text="DT", font=('arial',15,'bold'),command=dt)
b3.grid(row=3,column=1,columnspan=1)
b4=Button(w,text="NB", font=('arial',15,'bold'),command=nb)
b4.grid(row=4,column=1,columnspan=1)
b5=Button(w,pady=10,text="Compare", font=('arial',15,'bold'),bg="lightgreen",relief="sunken",command=compare)
b5.grid(row=6,column=1,columnspan=1)
b5=Button(w,pady=10,text="Submit", font=('arial',15,'bold'),bg="green",relief="sunken",command=submit)
b5.grid(row=6,column=2,columnspan=1)
b5=Button(w,pady=10,text="Reset", font=('arial',15,'bold'),bg="red",relief="sunken",command=reset)
b5.grid(row=6,column=3,columnspan=1)
L1=Label(w,bg="cyan",text="Enter the following data",font=('arial',15,'bold'))
L1.grid(row=1,ipadx=10,column=2)
L2=Label(w,bg="cyan",text="Sepal Width",font=('arial',15,'bold'))
L2.grid(row=2,ipadx=10,column=2)
L3=Label(w,bg="cyan",text="Sepal Length",font=('arial',15,'bold'))
L3.grid(row=3,ipadx=10,column=2)
L4=Label(w,bg="cyan",text="Petal Length",font=('arial',15,'bold'))
L4.grid(row=4,ipadx=10,column=2)
L5=Label(w,bg="cyan",text="Petal Width",font=('arial',15,'bold'))
L5.grid(row=5,ipadx=10,column=2)
E1=Entry(w,textvariable=v1,font=('arial',15,'bold'))
E1.grid(row=2,column=3)
E2=Entry(w,textvariable=v2,font=('arial',15,'bold'))
E2.grid(row=3,column=3)
E3=Entry(w,textvariable=v3,font=('arial',15,'bold'))
E3.grid(row=4,column=3)
E4=Entry(w,textvariable=v4,font=('arial',15,'bold'))
E4.grid(row=5,column=3)
w.mainloop()










