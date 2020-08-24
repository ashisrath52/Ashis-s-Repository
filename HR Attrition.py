# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 12:16:45 2020

@author: ashis
"""
import pandas as pd
dataset=pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
dataset1=dataset.isnull()
dataset2=dataset.dropna()
dataset3=dataset2.drop_duplicates()
dataset3.columns
dataset3[['Age','BusinessTravel','DistanceFromHome','Education', 'MonthlyIncome','NumCompaniesWorked','PercentSalaryHike','YearsWithCurrManager','TotalWorkingYears','YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion']]
dataset3[['Age','BusinessTravel','DistanceFromHome','Education', 'MonthlyIncome','NumCompaniesWorked','PercentSalaryHike','YearsWithCurrManager','TotalWorkingYears','YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion']].median()
dataset3[['Age','BusinessTravel','DistanceFromHome','Education', 'MonthlyIncome','NumCompaniesWorked','PercentSalaryHike','YearsWithCurrManager','TotalWorkingYears','YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion']].mode()
dataset3[['Age','BusinessTravel','DistanceFromHome','Education', 'MonthlyIncome','NumCompaniesWorked','PercentSalaryHike','YearsWithCurrManager','TotalWorkingYears','YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion']].std()
dataset3[['Age','BusinessTravel','DistanceFromHome','Education', 'MonthlyIncome','NumCompaniesWorked','PercentSalaryHike','YearsWithCurrManager','TotalWorkingYears','YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion']].skew()
dataset3[['Age','BusinessTravel','DistanceFromHome','Education', 'MonthlyIncome','NumCompaniesWorked','PercentSalaryHike','YearsWithCurrManager','TotalWorkingYears','YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion']].kurt()
import matplotlib.pyplot as plt
plt.boxplot(dataset.Age)
df=dataset3[['Age','BusinessTravel','DistanceFromHome','Education', 'MonthlyIncome','NumCompaniesWorked','PercentSalaryHike','YearsWithCurrManager','TotalWorkingYears','YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion']].describe()
df
plt.boxplot(dataset.YearsAtCompany)
attrition_yes=dataset[dataset['Attrition']=='Yes']
attrition_no=dataset[dataset['Attrition']=='No']
from scipy.stats import mannwhitneyu
stats,p=mannwhitneyu(attrition_yes.MonthlyIncome,attrition_no.MonthlyIncome)
print(stats,p)
stats,p=mannwhitneyu(attrition_yes.TotalWorkingYears,attrition_no.TotalWorkingYears)
print(stats,p)
stats,p=mannwhitneyu(attrition_yes.DistanceFromHome,attrition_no.DistanceFromHome)
print(stats,p)
stats,p=mannwhitneyu(attrition_yes.YearsAtCompany,attrition_no.YearsAtCompany)
print(stats,p)
stats,p=mannwhitneyu(attrition_yes.YearsWithCurrManager,attrition_no.YearsWithCurrManager)
print(stats,p)
stats,p=mannwhitneyu(attrition_yes.YearsSinceLastPromotion,attrition_no.YearsSinceLastPromotion)
print(stats,p)
stats,p=mannwhitneyu(attrition_yes.PercentSalaryHike,attrition_no.PercentSalaryHike)
print(stats,p)
stats,p=mannwhitneyu(attrition_yes.OverTime,attrition_no.OverTime)
print(stats,p)
stats,p=mannwhitneyu(attrition_yes.BusinessTravel,attrition_no.BusinessTravel)
print(stats,p)
stats,p=mannwhitneyu(attrition_yes.MaritalStatus,attrition_no.MaritalStatus)
print(stats,p)
from scipy.stats import pearsonr
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
dataset['Attrition']=label_encoder.fit_transform(dataset['Attrition'])
from scipy.stats import pearsonr
stats,p=pearsonr(dataset.Attrition,dataset.MonthlyIncome)
print(stats,p)
dataset4=dataset[['Age','DistanceFromHome','Education','MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike','TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany','YearsSinceLastPromotion', 'YearsWithCurrManager']]
dataset5=dataset4.corr()
import statsmodels.api as sm
dataset4=dataset[['Attrition','Age','DistanceFromHome','Education','MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike','TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany','YearsSinceLastPromotion', 'YearsWithCurrManager']]
Y=dataset['Attrition']

X=dataset['MonthlyIncome']
X1=sm.add_constant(X)
Singlepredictor=sm.Logit(Y,X1)
result=Singlepredictor.fit()
result.summary()





