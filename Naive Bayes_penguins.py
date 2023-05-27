import pandas as pd
import numpy as np
import math
dataset=pd.read_csv("Data\\penguins.csv")
dataset_size=dataset.shape[0]
no_featurs=dataset.shape[1]-1
"""

data preprocessing

"""
total=0;
class subFrame():
    """
    we create a class to extract the training samples for each class and also
    seperate our training set from our test set
    we define three options, 
    one for classes spliting and preprocessing,
    one for creatign 1vs rest naive bayes by combining the training samples of the two other classes
    and one for global training set.
    """
    def __init__(self,dataframe,column=None,class_name=None,option=1,dataframe2=None):
        if option==1:
            self.name=class_name
            self.dataframe=dataframe.loc[(dataframe[column]==class_name)]
            self.cleans()
            self.toNumeric()
            self.toKG()
            #spliting sub dataframe to test set and data set
            self.testset=self.dataframe.iloc[:round((self.dataframe.shape[0]/100)*20)]
            self.dataframe=self.dataframe.iloc[round((self.dataframe.shape[0]/100)*20):]
        elif option == 0 :
            self.dataframe=pd.concat([dataframe2,dataframe],ignore_index=True)
            self.trainset=None
        elif option==2:
            self.dataframe=dataframe
            self.trainset=None
            
    def return_std(self):
        #creating a std function for each feature of dataframe
        self.std={};
        for i in list(self.dataframe)[1:]:
            self.std[i]=round(self.dataframe[i].std(),3)
        return self.std
    def return_mean(self):
        self.mean={};
        for i in list(self.dataframe)[1:]:
            self.mean[i]=round(self.dataframe[i].mean(),3)
        return self.mean
    def toKG(self):
        #converting body masses from g to kg to normalize the distribution of data
        self.dataframe['body_mass_g']=self.dataframe['body_mass_g'].div(1000).round(1)
    
    def cleans(self):
        """
        deleting empty rows per each subFrame
        
        """
        self.dataframe=self.dataframe[self.dataframe['body_mass_g']!='x']
        
    def toNumeric(self):
        #converting string datas to numeric datas
        for i in list(self.dataframe)[1:]:
            self.dataframe[i]=pd.to_numeric(self.dataframe[i])
            
    def returnFrame(self):
        return self.dataframe
    def returnTest(self):
        return self.testset
    def set_total(self,total):
        self.total=total
    def p_wi(self):
        return round(self.dataframe.shape[0]/self.total,6)
    def p_xi_wi(self):
        pass
    
        
        
classes= set(dataset['species'])
#slicing datafram to get the mean, min, max and variance
Gentoo=subFrame(dataset,'species','Gentoo')
Adelie=subFrame(dataset,'species','Adelie')
Chinstrap=subFrame(dataset,'species','Chinstrap')
Gentoo_Adelie=subFrame(Gentoo.dataframe,dataframe2=Adelie.dataframe,option=0)
Adelie_Chinstrap=subFrame(Adelie.dataframe,dataframe2=Chinstrap.dataframe,option=0)
Gentoo_Chinstrap=subFrame(Gentoo.dataframe,dataframe2=Chinstrap.dataframe,option=0)
classes=[Gentoo,Adelie,Chinstrap]
combined_classes=[Gentoo_Adelie,Gentoo_Chinstrap,Adelie_Chinstrap]
for i in classes:
    total=i.dataframe.shape[0]+total
for i in range(3):
    classes[i].set_total(total)
    combined_classes[i].set_total(total)
global_trainset=pd.concat([Gentoo.returnFrame(),Adelie.returnFrame(),Chinstrap.returnFrame()],ignore_index=True)
global_testset=pd.concat([Gentoo.returnTest(),Adelie.returnTest(),Chinstrap.returnTest()],ignore_index=True)

global_trainset=subFrame(global_trainset,option=2)
#print(global_testset)
def likelihood(feature,class1:subFrame,label):
    std=class1.return_std()[label];
    mean=class1.return_mean()[label];
    f_x=(1/(std*math.sqrt(2*math.pi)))*(np.e**(-(((feature-mean)**2)/(2*(std**2)))))
    return f_x
def posterior (feature_vector:list,class1:subFrame,class2:subFrame):
    labels=list(dataset)[1:]
    f_xc1=[]
    f_xc2=[]
    px_w1=1
    px_w2=1
    for i in range(len(feature_vector)):
        f_xc1.append(likelihood(feature_vector[i],class1,labels[i]))
        f_xc2.append(likelihood(feature_vector[i],class2,labels[i]))
    #print (f_xc1)
    #print(f_xc2)
    for i in range(len(feature_vector)):
        px_w1=px_w1*f_xc1[i]
        px_w2=px_w2*f_xc2[i]
    #print (px_w1)
    #print (px_w2)
    posterior_w1=(px_w1*class1.p_wi())/((px_w1*class1.p_wi())+(px_w2*class2.p_wi()))
    posterior_w2=(px_w2*class2.p_wi())/((px_w1*class1.p_wi())+(px_w2*class2.p_wi()))
    return posterior_w1
def naive_bayes_model(feature_vector:list,class1:subFrame,
                     class2:subFrame,class3:subFrame,c_class1:subFrame,c_class2:subFrame,
                     c_class3:subFrame):
    """
    main objective :
    arg_max(P(w1|x),P(w2|x),P(w3|x))"""
    posteriors={}
    posteriors['Gentoo']=posterior(feature_vector,class1,c_class1)#GentooVsRest
    posteriors['Adelie']=posterior(feature_vector,class2,c_class2)#AdelieVsRest
    posteriors['Chinstrap']=posterior(feature_vector,class3,c_class3)#ChinstrapVsRest
    lst=list(posteriors)
    if posteriors['Gentoo']<=0.5:
        c1="ng"
    else :
        c1="g"
    if posteriors['Adelie']<=0.5:
        c2="na"
    else:
        c2="a"
    if posteriors['Chinstrap']<=0.5:
        c3="nc"
    else:
        c3="c"
    max=0
    choice=''
    for i in lst:
        if posteriors[i]>max:
            max=posteriors[i]
            choice=i
    return (choice,max,c1,c2,c3)
features=[]
for i in range(global_testset.shape[0]):
    features.append(global_testset.iloc[i])
misses=0
c1x11=0
c1x12=0
c1x21=0
c1x22=0
c2x11=0
c2x12=0
c2x21=0
c2x22=0
c3x11=0
c3x12=0
c3x21=0
c3x22=0
for i in features:
    output=naive_bayes_model([i[1],i[2],i[3],i[4]],Gentoo,Adelie,Chinstrap,Adelie_Chinstrap,Gentoo_Chinstrap,Gentoo_Adelie)
    #print ("answer :",i[0])
    #print ("predict : ",output[0])
    #print ("========\n\n")
    if output[0]!=i[0]:
        misses+=1   
    if i[0]=='Gentoo' and output[2]=='ng':
        c1x12=c1x12+1
    if i[0]=='Gentoo' and output[2]=='g':
        c1x11=c1x11+1
    if i[0]!='Gentoo' and output[2]=='g':
        c1x21=c1x21+1
    if i[0]!='Gentoo' and output[2]=='ng':
        c1x22=c1x22+1
         
         
    if i[0]=='Adelie' and output[3]=='na':
        c2x12=c2x12+1
    if i[0]=='Adelie' and output[3]=='a':
        c2x11=c2x11+1
    if i[0]!='Adelie' and output[3]=='a':
        c2x21=c2x21+1
    if i[0]!='Adelie' and output[3]=='na':
        c2x22=c2x22+1
         
    if i[0]=='Chinstrap' and output[4]=='nc':
        c3x12=c3x12+1
    if i[0]=='Chinstrap' and output[4]=='c':
        c3x11=c3x11+1
    if i[0]!='Chinstrap' and output[4]=='c':
        c3x21=c3x21+1
    if i[0]!='Chinstrap' and output[4]=='nc':
        c3x22=c3x22+1
cm1=np.array([[c1x11,c1x12],[c1x21,c1x22]])
cm2=np.array([[c2x11,c1x12],[c2x21,c1x22]])
cm3=np.array([[c3x11,c1x12],[c3x21,c1x22]])
print ("for classified 1 confusion matrix\n",cm1)
print ("for classifier 2 confusion matrix\n",cm2)
print ("for classifier 3 confusion matrix\n",cm3)
print ("recall for the first classifier ",cm1[0][0]/(cm1[0][0]+cm1[1][0]))
print ("recall for the second classifier ",cm2[0][0]/(cm2[0][0]+cm2[1][0]))
print ("recall for the third classifier ",cm3[0][0]/(cm3[0][0]+cm3[1][0]))
print ("percision for the first classifier ",cm1[0][0]/(cm1[0][0]+cm1[0][1]))
print ("percision for the second classifier ",cm2[0][0]/(cm2[0][0]+cm2[0][1]))
print ("percision for the third classifier ",cm3[0][0]/(cm3[0][0]+cm3[0][1]))
print ("Total accuracy", (global_trainset.dataframe.shape[0]-misses)/global_trainset.dataframe.shape[0])
        
