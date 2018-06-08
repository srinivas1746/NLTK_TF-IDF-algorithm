"""
It can takes the input from the text file and will apply TFIDF ALGORITHM to that file 


 """
from nltk.tokenize import sent_tokenize ,word_tokenize
import pandas as pd
from time import sleep
import math
import json
import operator
from nltk.corpus import stopwords
f = open('/home/seenu/Desktop/nltk_codes/hello.txt', 'r')  ## This is the text file where the resume information stores
lines = f.readlines()
mystr = '\t'.join([line.strip() for line in lines])   ## this line can arrange the information in one string irrespective of comma's and full stop
g = open('/home/seenu/Desktop/nltk_codes/hello1.txt', 'r')
lines1 = g.readlines()
mystr1 = ' '.join([line.strip() for line in lines1])
#print(mystr)
splt_0 =word_tokenize(mystr)
splt_1 = word_tokenize(mystr1)
print(splt_0)
stop = set(stopwords.words('english'))
fil_0 = []
for i in splt_0:
        if i not in stop:
               fil_0.append(i)
print(fil_0)
fil_1 = []
for i in splt_1:
        if i not in stop:
               fil_1.append(i)
print(fil_1)
sleep(2)
Doc = [mystr,mystr1]           ## combining the 2 strings

splitting_0 = mystr.split()    ## splitting the sentence
splitting_1 = mystr1.split()
print(splitting_0)
common = set(fil_0).union(set(fil_1))       ## using union to combining 

Dict_0 = dict.fromkeys(common,0)              ## Converting this into dictionary form          ###########
Dict_1 = dict.fromkeys(common,0)

for i in fil_0:
        Dict_0[i]+=1           ## If the word occur multiple times the value of that word will increase
for i in fil_1:
        Dict_1[i]+=1
#print(Dict_1)
print(pd.DataFrame([Dict_0,Dict_1]))  ## Converting it into matrix form for representation only
Final_doc = [Dict_0,Dict_1]
#print(Final_doc)



## computing TF for each word in corresponding document


## formula 

         #                        No. of times a word occur in document
 #                           TF =  -----------------------------------
  #                                Total no.of words in documents

def compute_tf(Dict_0,fil_0):
        word_count = len(fil_1)
        for key,value in Dict_0.items():
                Dict_0[key] = (value)/float(word_count)
        return Dict_0        
tf_Doc_0 = compute_tf(Dict_0,fil_0)
tf_Doc_1 = compute_tf(Dict_1,fil_1)



##             COMPUTING IDF
  
        ##                      Total no.of documents in corpus                        
        ##            IDF =   ----------------------------------
        ##                      No.of documents have that perticular word

        
def compute_idf(doclist):
        N= len(doclist)
        idfDict = {}
      
        idfDict =dict.fromkeys(doclist[0],0)     ## taking whole words into idfDict keyword
                       
        for doc in doclist:
                for word ,val in doc.items():
                        if val>0:
                                idfDict[word] +=1   ## If the word in 2 documents so the count will be 2
        for word ,val in idfDict.items():
                idfDict[word] =math.log( N/float(val))
        
        return idfDict
                       
idf = compute_idf([Dict_0,Dict_1])

## computing TFIDF 

        ##          TFIDF = TF*IDF

def compute_tfidf(tf_Doc,idf):
        tfidf = {}
        for word , val in tf_Doc.items():
                tfidf[word] = val*idf[word] 
        return tfidf
tfidf_0 = compute_tfidf(tf_Doc_0,idf)
tfidf_1 = compute_tfidf(tf_Doc_1,idf)
print(pd.DataFrame([tfidf_0,tfidf_1])) ##  Representing it in matrix form

sorted_tfidf_1 = sorted(tfidf_1.items(), key=operator.itemgetter(1),reverse = True)     
          
tfidf_1 = {'tfidf_0': sorted_tfidf_0}

with open('/home/seenu/Desktop/final.txt', 'w') as file:
    file.write(json.dumps(sorted_tfidf_0)) 
