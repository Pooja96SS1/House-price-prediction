import pandas as pd
import numpy as np

# Import files which have the code for preprocessing each of the features. Each feature pre-processing
# is implemented in seperate files.
import VehiclesDataPreprocessing as vdp
import HDateDataPreprocessing as adp
import HelpProvidedDataPreprocessing as hdp
import HTimeDataPreprocessing as atdp
import DataPreProcessingUtils as dpu
import PrintClassifierPerformance as pcp
from sklearnmetrics import score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier





from mlxtend.frequent_patterns import apriori, association_rules 

data = pd.read_csv('hdata.csv',dtype={})
print ('Data shape before pre-processing-',data.shape)

datad = pd.read_csv('hdata.csv',dtype={})
datasd=datad
print(datad)

datad.head()

datad.columns 

print("Unique Types of Rent")
actype=datad.Type.unique()
for i in range(len(actype)):
    print(actype[i])



# apriori
#Cleaning the Data
'''
# Stripping extra spaces in the description 
data['Remarks'] = datad["Remarks"].str.strip()
from sklearn.datasets.samples_generator import make_blobs
  
# Dropping the rows without any data 
datad.dropna(axis = 0, subset =['Remarks'], inplace = True) 
datad['Remarks'] = datad['Remarks'].astype('str') 
  
# Dropping all transactions which were done on credit 
datad = datad[~datad['Remarks'].str.contains('C')] '''


print("Patterns for the Floor")
actype=datad.Floor.unique()
for i in range(len(actype)):
    print(actype[i])



# Transactions done  in Overspeed
basket_OS = datad[datad['State']=="Karnataka"]
print("---------Karnataka Seggregation------------")
print(basket_OS)
  
# Transactions done DND 
basket_Drink = datad.loc[datad['State']=="Tamil Nadu"]
print("---------Tamil Nadu Seggregation------------")
print(basket_Drink)
  
# Transactions done in Normal Accident 
basket_NA = datad.loc[datad['State']=="Telangana"]
print("---------Telangana Seggregation------------")
print(basket_NA)



def hot_encode(x):
    print(x)
    if(x<= 0): 
        return 0
    if(x>= 1): 
        return 1

try :
    # Encoding the datasets 
    basket_encoded = basket_OS.TypeofAccident.applymap(hot_encode) 
    basket_OS = basket_encoded 
      
    basket_encoded = basket_Drink.applymap(hot_encode) 
    basket_Drink = basket_encoded 
      
    basket_encoded = basket_Por.applymap(hot_encode) 
    basket_Por = basket_encoded 
      
    basket_encoded = basket_NA.applymap(hot_encode) 
    basket_NA = basket_encoded 


    # Building the model
    frq_items = apriori(basket_OS, min_support = 0.05, use_colnames = True)

    # Collecting the inferred rules in a dataframe
    rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
    rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])

    print("Apriori output")
    print(rules.head())
except:
    print('Apriori Processing Done')




from matplotlib import pyplot as plt


locs=datad.State.unique()
print(locs)
accds=datad.Floor.unique()
print(accds)

cntop = len(datad[datad['State']=="Karnataka"])
cntdd = len(datad[datad['State']=="Tamil Nadu"])
cnttb = len(datad[datad['State']=="Telangana"])
cnttp = len(datad[datad['State']=="Madhya Pradesh"])

df=datad.State.unique()
print(df)
print('------------------------Diffrent State Rent----------------------------')

#for i in range(len(accds)):   #old
for i in range(1):
    datacollector=[]
    locations=[]
    r1=[]
    '''for j in range(len(locs)):
        nalist=datad.query('Location.str.contains("'+locs[j]+'")  and DriverFault=="'+df[0]+'"', engine='python')
        datacollector.append(nalist.DriverFault.count())
        locations.append(locs[j])
        r1.append(int(j+1))
    print(datacollector)'''
    
    # Create bars
    barWidth = 0.9
    #bars1 = [int(locs[0]),int(Acc[1]),int(Acc[2]),int(Acc[3]),int(Acc[4])]
    bars1 = [int(cntop),int(cntdd),int(cnttb),int(cnttp)]
    bars4 = bars1 

    # The X position of bars
    r1 = [1,2,3,4]
    r4=r1

    # Create barplot
    #lblr=accds[i]
    lblr='State wise Rent'
    plt.bar(r1, bars4, width = barWidth, label=lblr)
    # Note: the barplot could be created easily. See the barplot section for other examples.

    # Create legend
    plt.legend()

    # Text below each barplot with a rotation at 90°
    plt.xticks([r + barWidth for r in range(len(r4))], ['Karnataka','Tamil Nadu','Telangana','Madhya Pradesh'], rotation=90)
    # Create labels
        
        
    '''label = []#[str(Acc[0]),str(Acc[1]),str(Acc[2]),str(Acc[3]),str(Acc[4])]
    for i in range(len(datacollector)):
        label.append(str(datacollector[i]))'''
    label = [str(cntop),str(cntdd),str(cnttb),str(cnttp)]

    # Text on the top of each barplot
    for i in range(len(r4)):
        plt.text(x = r4[i]-0.5 , y = bars4[i]+0.1,  s = label[i], size = 6)

    # Adjust the margins
    plt.subplots_adjust(bottom= 0.4, top = 0.98)

    # Show graphic
    #plt.show(block=False)
    plt.show()
    #plt.pause(3)
    plt.close()

        
    

print('------------------------Diffrent State Rent end----------------------------')


tcv=datad.Area.unique()
#print(tcv)
print("Factor2")
print(tcv)
print('------------------------House Rent Area start----------------------------')
cntpc = len(datad[datad['Area']==600])
cnttl = len(datad[datad['Area']==800])
cntfs = len(datad[datad['Area']==1500])
cntc = len(datad[datad['Area']==2400])

#for i in range(len(accds)):   #old
'''for i in range(len(1)):'''
datacollector=[]
locations=[]
r1=[]

    
# Create bars
barWidth = 0.9
bars1 = [int(cntpc),int(cnttl),int(cntfs),int(cntc)]
bars4 = bars1 

# The X position of bars
r1 = [1,2,3,4]
r4=r1

# Create barplot
#lblr=accds[i]
lblr='Area Wise'
plt.bar(r1, bars4, width = barWidth, label=lblr)
# Note: the barplot could be created easily. See the barplot section for other examples.

# Create legend
plt.legend()

# Text below each barplot with a rotation at 90°
#plt.xticks([r + barWidth for r in range(len(r4))], ['20x30 ','30x40 ','20X40 ','60X40','50X80'], rotation=90)
# Create labels
    
    
'''label = []#[str(Acc[0]),str(Acc[1]),str(Acc[2]),str(Acc[3]),str(Acc[4])]
for i in range(len(datacollector)):
    label.append(str(datacollector[i]))'''
label = [str(cntpc),str(cnttl),str(cntfs),str(cntc)]

# Text on the top of each barplot
for i in range(len(r4)):
    plt.text(x = r4[i]-0.5 , y = bars4[i]+0.1, s = label[i], size = 6)

# Adjust the margins
plt.subplots_adjust(bottom= 0.4, top = 0.98)

# Show graphic
#plt.show(block=False)
plt.show()
#plt.pause(3)
plt.close()

        
    

print('------------------------House Rent Area display end----------------------------')



coll=datad.Rooms.unique()
#print(coll)
print("Factor3")
print(coll)
print('------------------------Number Of BHK start----------------------------')

cnthb = len(datad[datad['Rooms']==1])
#print(cnthb)
cntca = len(datad[datad['Rooms']==2])
cntp = len(datad[datad['Rooms']==3])
#for i in range(len(accds)):   #old
'''for i in range(len(1)):'''
datacollector=[]
locations=[]
r1=[]
'''for j in range(len(locs)):
    nalist=datad.query('Location.str.contains("'+locs[j]+'")  and Collusion=="'+coll[i]+'"', engine='python')
    datacollector.append(nalist.Collusion.count())
    locations.append(locs[j])
    r1.append(int(j+1))
print(datacollector)'''
block=5476
# Create bars
barWidth = 0.9
#bars1 = [int(locs[0]),int(Acc[1]),int(Acc[2]),int(Acc[3]),int(Acc[4])]
bars1 = [int(cnthb),int(cntca),int(cntp)]
bars4 = bars1 

# The X position of bars
r1 = [1,2,3]
r4=r1

# Create barplot
#lblr=accds[i]
#lblr=coll[i]
lblr='number of BHK'
plt.bar(r1, bars4, width = barWidth, label=lblr)
# Note: the barplot could be created easily. See the barplot section for other examples.

# Create legend
plt.legend()

# Text below each barplot with a rotation at 90°
plt.xticks([r + barWidth for r in range(len(r4))], ['1BHK ','2BHK','3BHK'], rotation=90)
# Create labels
    
    
'''label = []#[str(Acc[0]),str(Acc[1]),str(Acc[2]),str(Acc[3]),str(Acc[4])]
for i in range(len(datacollector)):
    label.append(str(datacollector[i]))'''
label = [str(cnthb),str(cntca),str(cntp)]

# Text on the top of each barplot
for i in range(len(r4)):
    plt.text(x = r4[i]-0.5 , y = bars4[i]+0.1, s = label[i], size = 6)

# Adjust the margins
plt.subplots_adjust(bottom= 0.4, top = 0.98)

# Show graphic
#plt.show(block=False)
plt.show()
#plt.pause(3)
plt.close()

        
    

print('------------------------Number Of BHK end----------------------------')




rc1=datad.Floor.unique()
#print(rc1)
print("Factor3")
print(rc1)
cntmr = len(datad[datad['Floor']==0])
cntph = len(datad[datad['Floor']==1])
cntkr = len(datad[datad['Floor']==2])
cntsg = len(datad[datad['Floor']==3])
print('------------------------Floor wise rent start----------------------------')

#for i in range(len(accds)):   #old
'''for i in range(len(rc1)):'''
datacollector=[]
locations=[]
r1=[]
'''for j in range(len(locs)):
    nalist=datad.query('Location.str.contains("'+locs[j]+'")  and RoadCondition1=="'+rc1[i]+'"', engine='python')
    datacollector.append(nalist.RoadCondition1.count())
    locations.append(locs[j])
    r1.append(int(j+1))
print(datacollector)'''

# Create bars
barWidth = 0.9
#bars1 = [int(locs[0]),int(Acc[1]),int(Acc[2]),int(Acc[3]),int(Acc[4])]
bars1 = [int(cntmr),int(cntph),int(cntkr),int(cntsg)]
bars4 = bars1 

# The X position of bars
r1 = [1,2,3,4]
r4=r1

# Create barplot
#lblr=accds[i]
#lblr=rc1[i]
lblr='Floor wise'
plt.bar(r1, bars4, width = barWidth, label=lblr)
# Note: the barplot could be created easily. See the barplot section for other examples.

# Create legend
plt.legend()

# Text below each barplot with a rotation at 90°
plt.xticks([r + barWidth for r in range(len(r4))], ['Ground Floor ','First Floor ','Second Floor','Third Floor'], rotation=90)
# Create labels
    
    
'''label = []#[str(Acc[0]),str(Acc[1]),str(Acc[2]),str(Acc[3]),str(Acc[4])]
for i in range(len(datacollector)):
    label.append(str(datacollector[i]))'''
label = [str(cntmr),str(cntph),str(cntkr),str(cntsg)]

# Text on the top of each barplot
for i in range(len(r4)):
    plt.text(x = r4[i]-0.5 , y = bars4[i]+0.1, s = label[i], size = 6)

# Adjust the margins
plt.subplots_adjust(bottom= 0.4, top = 0.98)

# Show graphic
#plt.show(block=False)
plt.show()
#plt.pause(3)
plt.close()

    


print('------------------------Floor wise rent end----------------------------')



print("Factor6")
data14=datad.query('Date.str.contains("2014")', engine='python')
data2014=len(data14)
data15=datad.query('Date.str.contains("2015")', engine='python')
data2015=len(data15)
data16=datad.query('Date.str.contains("2016")', engine='python')
data2016=len(data16)
data17=datad.query('Date.str.contains("2017")', engine='python')
data2017=len(data17)
data18=datad.query('Date.str.contains("2018")', engine='python')
data2018=len(data18)

print('------------------------Total no of revenue collected of each year----------------------------')

#for i in range(len(accds)):   #old
'''for i in range(len(nwh)):'''
datacollector=[]
locations=[]
r1=[]
'''for j in range(len(locs)):
    nalist=datad.query('Location.str.contains("'+str(locs[j])+'")  and HelmetorSeatbelt=="'+str(nwh[i])+'"', engine='python')
    datacollector.append(nalist.HelmetorSeatbelt.count())
    locations.append(locs[j])
    r1.append(int(j+1))
print(datacollector)'''

# Create bars
barWidth = 0.9
#bars1 = [int(locs[0]),int(Acc[1]),int(Acc[2]),int(Acc[3]),int(Acc[4])]
bars1 = [int(data2014),int(data2015),int(data2016),int(data2017),int(data2018)]
bars4 = bars1 

# The X position of bars
r1 = [1,2,3,4,5]
r4=r1

# Create barplot
#lblr=accds[i]
#lblr=nwh[i]
lblr='Year wise revenue Rate'
plt.bar(r1, bars4, width = barWidth, label=lblr)
# Note: the barplot could be created easily. See the barplot section for other examples.

# Create legend
plt.legend()

# Text below each barplot with a rotation at 90°
plt.xticks([r + barWidth for r in range(len(r4))], ['Data 2014 ','Data 2015','Data 2016 ','Data 2017','Data 2018 '], rotation=90)
# Create labels
    
    
'''label = []#[str(Acc[0]),str(Acc[1]),str(Acc[2]),str(Acc[3]),str(Acc[4])]
for i in range(len(datacollector)):
    label.append(str(datacollector[i]))'''
label = [str(data2014),str(data2015),str(data2016),str(data2017),str(data2018)]

# Text on the top of each barplot
for i in range(len(r4)):
    plt.text(x = r4[i]-0.5 , y = bars4[i]+0.1, s = label[i], size = 6)

# Adjust the margins
plt.subplots_adjust(bottom= 0.4, top = 0.98)

# Show graphic
#plt.show(block=False)
plt.show()
#plt.pause(10)
plt.close()

#--------------------------Prediction Phase------------------------------
locs=datad.State.unique()
print(locs)
accds=datad.Type.unique()
print(accds)


for i in range(len(accds)):
    if(accds[i]!='4'):
            
        datacollector=[]
        locations=[]
        r1=[]
        for j in range(len(locs)):
            nalist=datad.query('State.str.contains("'+locs[j]+'")  and Type=="'+accds[0]+'"', engine='python')
            datacollector.append(nalist.Floor.count())
            locations.append(locs[j])
            r1.append(int(j+1))
        print(datacollector)
        
        # Create bars
        barWidth = 0.9
        #bars1 = [int(locs[0]),int(Acc[1]),int(Acc[2]),int(Acc[3]),int(Acc[4])]
        #bars1 = [int(cntop),int(cntdd),int(cnttb),int(cnttp)]
        bars4 = datacollector 

        # The X position of bars
        #r1 = [1,2,3,4]
        r4=r1

        # Create barplot
        lblr=accds[i]
        #lblr='Driver Fault'
        plt.bar(r1, bars4, width = barWidth, label=lblr)
        # Note: the barplot could be created easily. See the barplot section for other examples.

        # Create legend
        plt.legend()

        # Text below each barplot with a rotation at 90°
        plt.xticks([r + barWidth for r in range(len(r4))], locations, rotation=90)
        # Create labels
            
            
        label = []#[str(Acc[0]),str(Acc[1]),str(Acc[2]),str(Acc[3]),str(Acc[4])]
        for i in range(len(datacollector)):
            label.append(str(datacollector[i]))
        #label = [str(cntop),str(cntdd),str(cnttb),str(cnttp)]

        # Text on the top of each barplot
        for i in range(len(r4)):
            plt.text(x = r4[i]-0.5 , y = bars4[i]+0.1,  s = label[i], size = 6)

        # Adjust the margins
        plt.subplots_adjust(bottom= 0.4, top = 0.98)

        # Show graphic
        #plt.show(block=False)
        plt.show()
        #plt.pause(3)
        plt.close()




'''
hbs=datad.Rooms.unique()
print(hbs)


for i in range(len(hbs)):
    
    if(i<4):
        datacollector=[]
        locations=[]
        r1=[]
        for j in range(len(locs)):
            nalist=datad.query('State.str.contains("'+locs[j]+'")  and Rooms=="'+hbs[0]+'"', engine='python')
            datacollector.append(nalist.Rooms.count())
            locations.append(locs[j])
            r1.append(int(j+1))
        print(datacollector)
        
        # Create bars
        barWidth = 0.9
        #bars1 = [int(locs[0]),int(Acc[1]),int(Acc[2]),int(Acc[3]),int(Acc[4])]
        #bars1 = [int(cntop),int(cntdd),int(cnttb),int(cnttp)]
        bars4 = datacollector 

        # The X position of bars
        #r1 = [1,2,3,4]
        r4=r1

        # Create barplot
        lblr=hbs[i]
        #lblr='Driver Fault'
        plt.bar(r1, bars4, width = barWidth, label=lblr)
        # Note: the barplot could be created easily. See the barplot section for other examples.

        # Create legend
        plt.legend()

        # Text below each barplot with a rotation at 90°
        plt.xticks([r + barWidth for r in range(len(r4))], locations, rotation=90)
        # Create labels
            
            
        label = []#[str(Acc[0]),str(Acc[1]),str(Acc[2]),str(Acc[3]),str(Acc[4])]
        for i in range(len(datacollector)):
            label.append(str(datacollector[i]))
        #label = [str(cntop),str(cntdd),str(cnttb),str(cnttp)]

        # Text on the top of each barplot
        for i in range(len(r4)):
            plt.text(x = r4[i]-0.5 , y = bars4[i]+0.1,  s = label[i], size = 6)

        # Adjust the margins
        plt.subplots_adjust(bottom= 0.4, top = 0.98)

        # Show graphic
        plt.show(block=False)
        plt.pause(3)
        plt.close()



print("-------------------------SVM----------------------------")'''
from collections import Counter

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)
# Modeling step Test differents algorithms 
random_state = len(actype)
classifiers = []
classifiers.append(SVC(random_state=random_state))


### SVM classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

#gsSVMC.fit(train_X, train_y)
'''
print(train_X)
print(train_y)

svmpred= gsSVMC.predict(test_x)
print(svmpred)

SVMC_best = gsSVMC.best_estimator_

# Best score
gsSVMC.best_score_
svmacc=score.SVCaccuracy()'''
# Plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

try :
    g = plot_learning_curve(gsSVMC.best_estimator_,"SVM curve",train_X,test_y,cv=kfold)
    #plt.show()
except:
    print()
