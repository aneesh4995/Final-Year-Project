import numpy as np
import pandas as pd
import csv
import random
import math
from random import gauss


rafg1c = []
rafg2c = []
rafg1m = []
rafg2m = []
rafg1f = []
rafg2f = []
rafg2m = []


def g1c(digits):
 	rounded_number = round(random.uniform(0.20, 0.88), digits)
 	rafg1c.append(rounded_number)

def g1m(digits):
 	rounded_number = round(random.uniform(0.20, 0.88), digits)
 	rafg1f.append(rounded_number)

def g1f(digits):
 	rounded_number = round(random.uniform(0.33, 0.7), digits)
 	rafg1m.append(rounded_number)	

def g2c(digits):
 	rounded_number = round(random.uniform(0.23, 0.89), digits)
 	rafg2c.append(rounded_number)

def g2m(digits):
 	rounded_number = round(random.uniform(0.15, 0.83), digits)
 	rafg2f.append(rounded_number)

def g2f(digits):
 	rounded_number = round(random.uniform(0.36, 0.88), digits)
 	rafg2m.append(rounded_number)	
###################################################################################
def g1c(digits):
 	rounded_number = round(random.uniform(0.41, 0.81), digits)
 	rafg1c.append(rounded_number)

def g1m(digits):
 	rounded_number = round(random.uniform(0.30, 0.68), digits)
 	rafg1m.append(rounded_number)

def g1f(digits):
 	rounded_number = round(random.uniform(0.21, 0.71), digits)
 	rafg1f.append(rounded_number)	

def g2c(digits):
 	rounded_number = round(random.uniform(0.33, 0.74), digits)
 	rafg2c.append(rounded_number)

def g2m(digits):
 	rounded_number = round(random.uniform(0.35, 0.63), digits)
 	rafg2m.append(rounded_number)

def g2f(digits):
 	rounded_number = round(random.uniform(0.37, 0.84), digits)
 	rafg2f.append(rounded_number)	



for i in range(0,270):
	g1c(2)
	#g1m(2)
	#g1f(2)
	#g2c(2)
	#g2f(2)
	#g2m(2)

'''#print (rafg1c, rafg1f)	
for j in range(0,121):
	g1c(2)
	g1m(2)
	g1f(2)
	g2c(2)
	g2f(2)
	g2m(2)
'''

'''
data = pd.read_csv('FinalHearT.csv', sep=" ", header=None)
data.columns = ["age", "sex", "cpt", "rbp", "chol", "fbs", "ecg", "hr", "eia", "oldpeak", "slope", "nov", "thal", "rafg1C","rafg1M","rafg1F","rafg2C","rafg2M","rafg2F" ,"outcome"]
df = pd.DataFrame(data)
del df["rafg1C"]
print df
df.to_csv('HEART.csv', sep=' ', encoding='utf-8',index=False)
'''

#print (sorted_df)

# Splitting Dataset based on whether the outcome is presence or absence
def split_on_absence():
	absence = df[14]<2
	df[absence].to_csv('heart1.csv', sep=' ', encoding='utf-8',index=False)

def split_on_presence():
	presence = df[14]>1
	df[presence].to_csv('heart2.csv', sep=' ', encoding='utf-8',index=False)


se1 = pd.Series(rafg1c)
'''se2 = pd.Series(rafg1m)
se3 = pd.Series(rafg1f)
se4 = pd.Series(rafg2c)
se5 = pd.Series(rafg2m)
se6 = pd.Series(rafg2f)
'''
'''
data = pd.read_csv('HEART.csv', sep=" ", header=None)
data.columns = ["age", "sex", "cpt", "rbp", "chol", "fbs", "ecg", "hr", "eia", "oldpeak", "slope", "nov", "thal", "rafg1M","rafg1F","rafg2C","rafg2M","rafg2F", "outcome"]
df = pd.DataFrame(data)
	#print(df)
df['rafg1C'] = se1.values
	
print df
df.to_csv('hearT.csv', sep=' ', encoding='utf-8',index=False)
'''

data = pd.read_csv('hearT.csv', sep=" ", header=None)
df3 = pd.DataFrame(data)
#print df3
df3 = df3[[0,1,2,3,4,5,6,7,8,9,10,11,12,19,13,14,15,16,17,18]]
print df3
df3.to_csv('FinalHearT.csv',sep=' ', encoding='utf-8',index =False)


'''

data = pd.read_csv('heart2.csv', sep=" ", header=None)
data.columns = ["age", "sex", "cpt", "rbp", "chol", "fbs", "ecg", "hr", "eia", "oldpeak", "slope", "nov", "thal", "rafF1", "outcome"]
df = pd.DataFrame(data)
df['rafg1C'] = se1.values
df['rafg1M'] = se2.values
df['rafg1F'] = se3.values
df['rafg2C'] = se4.values
df['rafg2M'] = se5.values
df['rafg2F'] = se6.values 
#df.to_csv('heart2.csv', sep=' ', encoding='utf-8',index =False)
'''


# Concatenating both files as single file
def concate(df1,df2):
	frames = [df1,df2]
	result = pd.concat(frames)
	df3 = pd.DataFrame(result)
	df3 = df3[[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,13]]
	#cols = list(df3.columns.values)
	df3.to_csv('FinalHeart.csv',sep=' ', encoding='utf-8',index =False)

#data1 = pd.read_csv('heart1c.csv', sep=" ", header=None)
#df1 = pd.DataFrame(data1)


#data2 = pd.read_csv('heart2c.csv', sep=" ", header=None)
#df2 = pd.DataFrame(data2)

#concate(df1,df2)

