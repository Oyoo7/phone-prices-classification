import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import numpy as np


dftr= pd.read_csv(r"D:\vscode projects\projects.py\train.csv")
dfte = pd.read_csv(r"D:\vscode projects\projects.py\test.csv")
dftee = dfte.drop("id",axis = 1)
dftee
print(dftr)
print("head is \n:" ,dftr.head())
print("description is \n:", dftr.describe())
print("info is \n:", dftr.info())
print("shape is \n:", dftr.shape)
print("The nulls are \n:", dftr.isnull().sum())
#plot the histogram and boxplots to visualize the datasets
column = dftr.columns
for col in column:
    #balnk fig
    blanfig, axes = plt.subplots(1,2,figsize= (10,4))
    #hist
    dftr[col].hist(bins=30, ax=axes[0])
    axes[0].set_title(f"histogram of {col}")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("frequency")
    #boxplot
    dftr[col].plot.box(ax= axes[1])
    axes[1].set_title(f"boxplot of {col}")
    axes[1].set_xlabel(col)
    axes[1].set_ylabel("values")
plt.tight_layout()
plt.show()
#Eda notes
eda_notes = {}
eda_notes["battery_power"]={
     "type": "continuous",
    "skew":None,
    "outliers":None,
    "correlation":0.200,
    "action":"scale"
}

eda_notes["blue"] ={
    "type":"binary",
    "skew": None,
    "outliers":None,
    "correlation":0.020,
    "action":"no preprocessing needed"
}
eda_notes["clock_speed"]={
    "type": "continuous",
    "skew":"right",
    "outliers":None,
    "correlation":-0.006,
    "action":"scale,log transform later"
}
eda_notes["dual_sim"] = {
    "type":"binary",
    "skew":None,
    "outliers": None,
    "correlation":0.0174,
    "action": "no preprocessing needed"
}
eda_notes["fc"]={
    "type":"continuous",
    "skew":"right",
    "outliers":"three high values(17-19 mp)",
    "correlation":0.0212,
    "action":"check outlier vs target,scale,log transform later"
}
eda_notes["four_g"]={
    "type":"binary",
    "skew":None,
    "outliers":None,
    "correlation":0.014,
    "action":"no preprocessing needed"
}
eda_notes["int_memory"]={
    "type":"continous",
    "skew":None,
    "outliers":None,
    "correlation":0.044,
    "action":"no preprocessing needed.flagged for bivariate phase"
}
eda_notes["m_dep"]={
    "type":"continuous",
    "skew":None,
    "outliers":None,
    "correlation":0.000853,
    "action":"no preprocessing needed.flagged for evaluation stage"
}
eda_notes["mobile_wt"]={
    "type":"continuous",
    "skew":None,
    "outliers":None,
    "correlation":-0.030302,
    "action":"no preprocessing needed.flagged for bivariate analysis and evaluation"
}
eda_notes["n_cores"]={
    "type":"discrete",
    "skew":None,
    "outliers":None,
    "correlation":0.004399,
    "action":"flag for scaling"
}
eda_notes["pc"]={
    "type":"continuous",
    "skew":None,
    "outliers":None,
    "correlation":0.0336,
    "action":"possible scaling for svm and logistic regression"
}
eda_notes["px_height"]={
    "type":"continuous",
    "skew":"right",
    "outliers":"present",
    "correlation":0.1488,
    "action":"check for outliers, flag for possible log transform,scale"
}
eda_notes["px_width"]={
    "type":"continuous",
    "skew":None,
    "outliers":None,
    "correlation":0.165,
    "action":"scale"
}
     
eda_notes["ram"]={
    "type":"continuous",
    "skew":None, 
    "outliers":None,
    "correlation":0.917,
    "action":"scale"
}
eda_notes["sc_h"]={
    "type":"discrete",
    "skew":"slight left",
    "outliers":"none",
    "correlation":0.022,
    "action":"flag for possible log transform,scale"
}
eda_notes["sc_w"]={
    "type":"continuous",
    "skew":"right",
    "outliers":"none",
    "correlation":0.0387,
    "action":"flag for possible log transform,scale"
}
eda_notes["talk_time"]={
    "type":"continuous",
    "skew":None,
    "outliers":None,
    "correlation":0.0218,
    "action":"scale"
}
eda_notes["three_g"]={
    "type":"binary",
    "skew":None,
    "outliers":None,
    "correlation":0.0236,
    "action":"No preprocessing needed"
}
eda_notes["touch_screen"]={
    "type":"binary",
    "skew":None,
    "outliers":None,
    "correlation":-0.0303,
    "action":"No preprocessing needed"
}
eda_notes["wifi"]={
    "type":"binary",
    "skew":None,
    "outliers":None,
    "correlation":0.018,
    "action":"No preprocessing needed"
}
eda_notes["price_range"]={
    "type":"discrete",
    "skew":None,
    "outliers":None,
    "correlation":1,
    "action":"No preprocessing needed,target variable"
}

eda_notes
#bivariate analytics.perfom a correlation test between the features and target variable
correlations = dftr.corr()["price_range"].sort_values(ascending=False)
print(f"The correlations betwwen the features and target variables are\n:{correlations}")
multicolleniarity = dftr.drop("price_range",axis = 1).corr()
print(f"the multicolleniarity check between features are\n:{multicolleniarity}")
#log important multicollinearity observations
multicolleniarity_notes={}
multicolleniarity_notes["three_g,four_g"]={
    "correlation":0.58,
    "action":"check if dropping either or combining will improve model"
}
multicolleniarity_notes["fc,pc"]={
    "correlation":0.64,
    "action":"check if dropping either or combining will improve model"
    }

multicolleniarity_notes["sc_h,sc_w"]={
    "correlation":0.5,
    "action":"check if dropping either or combining will improve model"
    }
dfte.columns
#split data into train and test
x = dftr.drop("price_range", axis =1)
y = dftr["price_range"]
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.20, random_state= 42)
#scale data for svc and logistic regression
scaler  =  StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled= scaler.transform(x_test)
#log transformation test
dftr_log = dftr.copy()
log_cols = ['clock_speed','fc','px_height','sc_h','sc_w']
for col in log_cols:
    dftr_log[col] = np.log1p(dftr_log[col])
#split log transformed data

x_log = dftr_log.drop("price_range",axis = 1)
y_log = dftr_log["price_range"]
x_train_log,x_test_log,y_train_log,y_test_log = train_test_split(x_log,y_log,test_size=0.20, random_state=42)
scaler_log = StandardScaler()
x_train_log_scaled = scaler_log.fit_transform(x_train_log)
x_test_log_scaled = scaler_log.transform(x_test_log)
#train randm forest log
model_logrf = RandomForestClassifier( random_state=42)
model_logrf.fit(x_train_log,y_train_log)
rfpred_log = model_logrf.predict(x_test_log)
#train SVC log scaled
svclogscmodel = SVC(random_state=42)
svclogscmodel.fit(x_train_log_scaled,y_train_log)
svclogscpred = svclogscmodel.predict(x_test_log_scaled)
#train Lr
Lrlogscmodel = LogisticRegression(random_state=42)
Lrlogscmodel.fit(x_train_log_scaled,y_train_log)
Lrlogscpred = Lrlogscmodel.predict(x_test_log_scaled)

#train random forest
rf_model= RandomForestClassifier(random_state=42)
rf_model.fit(x_train,y_train)
#train svc
svc_model= SVC()
svc_model.fit(x_train_scaled,y_train)
#logistic reg train
log_model = LogisticRegression(random_state=42)
log_model.fit(x_train_scaled,y_train)
#prediction
rf_pred = rf_model.predict(x_test)
svc_pred = svc_model.predict(x_test_scaled)
log_pred = log_model.predict(x_test_scaled)
#evaluation rf
rf_accuracy = accuracy_score(y_test,rf_pred)
rf_confusion = confusion_matrix(y_test,rf_pred)
rf_report = classification_report(y_test,rf_pred)
#evaluation svc
svc_accuracy = accuracy_score(y_test,svc_pred)
svc_confusion = confusion_matrix(y_test,svc_pred)
svc_report = classification_report(y_test,svc_pred)
#evaluate logistic reg
log_accuracy = accuracy_score(y_test,log_pred)
log_confusion= confusion_matrix(y_test,log_pred)
log_report= classification_report(y_test,log_pred)
#log transformation out put
rf_log_accuracy = accuracy_score(y_test_log,rfpred_log)
rf_log_confusion = confusion_matrix(y_test_log,rfpred_log)
rf_log_classificationr = classification_report(y_test_log,rfpred_log)
#svc log output
svc_log_accuracy = accuracy_score(y_test_log,svclogscpred)
svc_log_confusion = confusion_matrix(y_test_log,svclogscpred)
svc_log_classificationr = classification_report(y_test_log,svclogscpred)
#LR log output
LRlog_accuracy = accuracy_score(y_test_log,Lrlogscpred)
LRlog_confusion = confusion_matrix(y_test_log,Lrlogscpred)
Lrlog_report =classification_report(y_test_log,Lrlogscpred)




#print result first run
print(f"the accuracy of random forest is :\n{rf_accuracy}")
print(f"the confusion of random forest is :\n{rf_confusion}")
print(f"the repor of random forest is :\n{rf_report}")
print(f"the accuracy of svc is :\n {svc_accuracy}")
print(f"the confusion of svc is :\n{svc_confusion}")
print(f"the report of svc is :\n{svc_report}")
print(f"the accuracy of logistic reg is : \n{log_accuracy}")
print(f"the confusion of logistic reg is :\n{log_confusion}")
print(f"the report of logistic reg is : \n {log_report}")
print(f"the accuracy of random forest log transformed is : \n {rf_log_accuracy}")
print(f"the confusion matrix of random forest log transformed is : \n {rf_log_confusion}")
print(f"the clasification report of random forest log transformed is : \n {rf_log_classificationr}")

print(f"the accuracy of SVC log transformed is : \n {svc_log_accuracy}")
print(f"the confusion matrix of SVC log transformed is : \n {svc_log_confusion}")
print(f"the classification report of SVC log transformed is : \n {svc_log_classificationr}")

print(f"the accuracy of Logistic reg log transformed is : \n {LRlog_accuracy}")
print(f"the confusion matrix of Logistic reg log transformed is : \n {LRlog_confusion}")
print(f"the classification report of Logistic reg log transformed is : \n {Lrlog_report}")
dftee_scaled = scaler.transform(dftee)
dftee_predictions = log_model.predict(dftee_scaled)
dfte["predicted_range"] = dftee_predictions
dfte
import joblib
joblib.dump(log_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print(dftr.columns)