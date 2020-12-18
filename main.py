#importing all the python modules 

import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
from tkinter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import tweepy 
import csv
warnings.filterwarnings("ignore", category=DeprecationWarning)
from tkinter import *
import tweepy 
import csv
  
#Twitter API_DETAILS
consumer_key = "XXXXXXXXXXXXXXXXXXX" 
consumer_secret = "XXXXXXXXXXXXXXXX"
access_key = "XXXXXXXXXXXXXXXXXXXXX"
access_secret = "XXXXXXXXXXXXXXXXX"

#Extract the Tweets
def get_tweets():
        # Authorization to consumer key and consumer secret 
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 
  
        # Access to user's access key and access secret 
        auth.set_access_token(access_key, access_secret) 
        # Calling api 
        api = tweepy.API(auth)

        alltweets = []
        screen_name=username_text.get()
        new_tweets = api.user_timeline(screen_name = screen_name,count=200)
        alltweets.extend(new_tweets)

        #save the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        while len(new_tweets) > 0:
                new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
                alltweets.extend(new_tweets)
                oldest = alltweets[-1].id - 1
        #REMOVING EMOGI'S        
        for i in range(len(alltweets)):
            ch=''
            for char in alltweets[i].text:
                if ord(char)<65535:
                    ch+=char
            alltweets[i].text = ch
                
        outtweets = [[tweet.id_str, tweet.text] for tweet in alltweets]
        #WRITE THE CSV FILE OF TWEETS
        with open(f'new_{screen_name}_tweets.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["id","tweet"])
                writer.writerows(outtweets)

      
        #TRAIN and TEST THE ML Model
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')

        #READ THE CSV FILE OF TWEETS
        csvFileName = 'new_' + screen_name + '_tweets.csv'
        new_tweet=pd.read_csv(csvFileName)

        test=test.append(new_tweet)
        combine = train.append(test,ignore_index=True,sort=True)
       #PreProcessing and Data Cleaning

        def remove_pattern(text,pattern):
            
            # re.findall() finds the pattern i.e @user and puts it in a list for further task
            r = re.findall(pattern,text)
            
            # re.sub() removes @user from the sentences in the dataset
            for i in r:
                text = re.sub(i,"",text)
            
            return text
        combine['Tidy_Tweets'] = np.vectorize(remove_pattern)(combine['tweet'], "@[\w]*")
        combine['Tidy_Tweets'] = combine['Tidy_Tweets'].str.replace("[^a-zA-Z#]", " ")
        tokenized_tweet = combine['Tidy_Tweets'].apply(lambda x: x.split())
        
        from nltk import PorterStemmer
        ps = PorterStemmer()
        tokenized_tweet = tokenized_tweet.apply(lambda x: [ps.stem(i) for i in x])

        for i in range(len(tokenized_tweet)):
            tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

        combine['Tidy_Tweets'] = tokenized_tweet

        def Hashtags_Extract(tokenized_tweet):
            hashtags=[]
            
            # Loop over the words in the tweet
            for i in tokenized_tweet:
                ht = re.findall(r'#(\w+)',i)
                hashtags.append(ht)
            
            return hashtags

        #TFID
        tfidf=TfidfVectorizer(max_df=0.90, min_df=2,max_features=1000,stop_words='english')

        tfidf_matrix=tfidf.fit_transform(combine['Tidy_Tweets'])

        df_tfidf = pd.DataFrame(tfidf_matrix.todense())

        train_tfidf_matrix = tfidf_matrix[:31962]
        train_tfidf_matrix.todense()

        from sklearn.model_selection import train_test_split

        x_train_tfidf,x_valid_tfidf,y_train_tfidf,y_valid_tfidf = train_test_split(train_tfidf_matrix,train['label'],test_size=0.3,random_state=17)

       #TEST ON LOGISTIC REGRESSION ML MODEL

        Log_Reg = LogisticRegression(random_state=0,solver='lbfgs')
        Log_Reg.fit(x_train_tfidf,y_train_tfidf)
        prediction_tfidf = Log_Reg.predict_proba(x_valid_tfidf)
        prediction_int = prediction_tfidf[:,1]>=0.3
        prediction_int = prediction_int.astype(np.int)

        # calculating f1 score
        log_tfidf = f1_score(y_valid_tfidf, prediction_int)

        test_tfidf = tfidf_matrix[31962:]

        test_pred = Log_Reg.predict_proba(test_tfidf)

        test_pred_int = test_pred[:,1] >= 0.3

        test_pred_int = test_pred_int.astype(np.int)

        test['label'] = test_pred_int

        submission = test[['id','label']]
        submission = submission[17197:]

        submission.to_csv('result.csv', index=False)


        tweetIdx = 0
        sentiment = ''
        displayTweet = []
        with open(r'C:\Users\rohit\Desktop\Twitter_Sentiment_Analysis\result.csv','r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['label'] == '0':
                    sentiment = " -- POSITIVE"
                else:
                    sentiment = " -- NEGATIVE"
                displayTweet.append(alltweets[tweetIdx].text+" "+sentiment)
                tweetIdx+=1
     
        list1.delete(0,END)
        for row in displayTweet:
            list1.insert(END,row)


# tkinter window 
window=Tk ()
window.geometry('350x400')
window.wm_title("Twitter Sentiment Analysis")
window['bg'] = '#2980b9'


l1=Label(window,text="Twitter Handle",font ="roboto 10")
l1.grid(row=0,column=7,pady=7)


username_text = StringVar()
e1=Entry(window,textvariable=username_text,font="lucida 10")
e1.grid(row=1,column=7,pady=7)


b1=Button(window,text="View Tweets", width=12,command=get_tweets,font ="lucida 10")
b1.grid(row=3,column=7,padx=10, pady=7)

list1=Listbox(window,width='45')
list1.grid(row=6,column=5,rowspan=15,columnspan=8, padx=30,pady=50)

if __name__ == "__main__":
    window.mainloop()
