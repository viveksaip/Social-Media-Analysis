# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 22:41:57 2018

@author: Lenovo
"""

import tweepy
from tweepy import OAuthHandler
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from IPython import get_ipython
import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split  
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from keras.layers import Bidirectional
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from wordcloud import WordCloud
from pylab import * 
import seaborn as sns
from scipy.stats import hmean 
from scipy.stats import norm


   
#SentimentAnalysis (training on 1.6M tweets)
    csv = 'clean_tweet_final.csv'
    my_df = pd.read_csv(csv,index_col=0)
    my_df.head()
    
# ti check null entity
    my_df[my_df.isnull().any(axis=1)].head()
    np.sum(my_df.isnull().any(axis=1))
    my_df.isnull().any(axis=0)
    

# creating unique id for tweets

     
    tokenizer = Tokenizer(nb_words=2500, lower=True,split=' ')
    tokenizer.fit_on_texts(my_df['token'].values)
    X = tokenizer.texts_to_sequences(my_df['token'].values)
    #tokenizer.fit_on_texts(my_df['token'].astype(str).values)
    X = pad_sequences(X)
    Y = pd.get_dummies(my_df['target']).values

    smote = SMOTE(ratio='minority')
    X_sm, y_sm = smote.fit_sample(X, Y[:,1])
    
    y_sm = pd.get_dummies(y_sm).values


    #bidirectional
    embed_dim = 128
    lstm_out = 200
    batch_size = 32
    
    model = Sequential()
    model.add(Embedding(2500, embed_dim,input_length = X.shape[1], dropout = 0.2))
    model.add(Bidirectional((LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())
    
    

    #Here we train the Network.

    model.fit(X_sm, y_sm, batch_size =batch_size, nb_epoch = 1,  verbose = 5)
   


#WebScraping 

def Social_Media_Analysis(kw,n):      
    
    consumer_key = 'hQSfuYsO9ybVNVKPCxURndKQ6'
    consumer_secret = 'BUbTUcCSOozzU3PzFSbt44A4nWDjWVccAwuyPTgzQo25ImhcsV'
    access_key = '1030700072755855361-1bl3CyvIAvr8VQ74j368130fq7odUT'
    access_secret = '1cw58KNSxH0VCs0bqZCbYdevCIm6sIpzQLkuBvglqirwQ'
 
          
        # Authorization to consumer key and consumer secret 
    auth = tweepy.OAuthHandler(consumer_key,consumer_secret) 
  
        # Access to user's access key and access secret 
    auth.set_access_token(access_key, access_secret) 
    
  
        # Calling api 
    api = tweepy.API(auth) 
  
        # n tweets to be extracted 
    tweets = api.user_timeline(screen_name= kw, count=n ) 
  
        # Empty Array 
    tmp=[] 
    tmp1=[]
        
  
        # create array of tweet information: username,  
        # tweet id, date/time, text 
    tweets_for_csv = [tweet.text for tweet in tweets]# CSV file created  
    tweets_for_csv1 = [tweet.created_at for tweet in tweets]
    csv= 'tweets_for_csv'
    for j in tweets_for_csv: 
            # Appending tweets to the empty array tmp 
        tmp.append(j)
            
    csv= 'tweets_for_csv1'
    for j in tweets_for_csv1: 
            # Appending tweets to the empty array tmp 
        tmp1.append(j.date())    
            
  
        # Printing the tweets 
        
    df1 = pd.DataFrame({'text':tmp})
    d = pd.DataFrame({'col':tmp1})
    df1.to_excel('ext5.xlsx')
    d.to_excel('tim1.xlsx')
    
    
#keyword = 'romansaini'
#n = 20    
#Social_Media_Analysis(keyword,n)    
    

#Cleaning Tweets

    get_ipython().run_line_magic('matplotlib', 'inline')
 #   %config InlineBackend.figure_format = 'retina'
    tok = WordPunctTokenizer()

    pat1 = r'@[A-Za-z0-9_]+'
    pat2 = r'https?://[^ ]+'
    combined_pat = r'|'.join((pat1, pat2))
    www_pat = r'www.[^ ]+'
    negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
    neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
    
    df1= pd.read_excel('ext5.xlsx')
    
    corpus = []
    for i in range(0,n):
        review = re.sub('[^a-zA-Z]', ' ', df1['text'][i])
        review = review.lower()
        review = review.split()# converting into list
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)

    def tweet_cleaner(text):
        soup = BeautifulSoup(text, 'lxml')
        souped = soup.get_text()
        try:
            bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
        except:
                bom_removed = souped
        stripped = re.sub(combined_pat, '', bom_removed)
        stripped = re.sub(www_pat, '', stripped)
        lower_case = stripped.lower()
        neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
        letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
        # During the letters_only process two lines above, it has created unnecessay white spaces,
        # I will tokenize and join together to remove unneccessary white spaces
        ps = PorterStemmer()
        wordsq = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
        review = [ps.stem(word) for word in wordsq if not word in set(stopwords.words('english'))]
        return (" ".join(review)).strip()


    testing = df1.text[:100]
    test_result = []
    for t in testing:
        test_result.append(tweet_cleaner(t))
    test_result

    #nums = [0,1600000]
    print ("Cleaning and parsing the tweets...\n")
    clean_tweet_texts = []
    for i in range(0,n):
        #if( (i+1)%10000 == 0 ):
         #   print ("Tweets %d of %d has been processed" % ( i+1, nums[1] )  )                                                                  
        clean_tweet_texts.append(tweet_cleaner(df1['text'][i]))
    
# saving cleaned data
    clean_df = pd.DataFrame(clean_tweet_texts,columns=['text'])
    clean_df.head()    
    
    
    clean_df['token']=clean_df['text'].apply(word_tokenize)
    clean_df.head()
    
    clean_df.to_csv('clean_tweet6.csv',encoding='utf-8')
    csv1 = 'clean_tweet6.csv'
    my_df1 = pd.read_csv(csv1,index_col=0)
    my_df1.head()
    
    # ti check null entity
    my_df1[my_df1.isnull().any(axis=1)].head()
    np.sum(my_df1.isnull().any(axis=1))
    my_df1.isnull().any(axis=0)

    
    
    X1 = tokenizer.texts_to_sequences(my_df1['token'].values)
#padding the tweet to have exactly the same shape as `embedding_2` input
    X1 = pad_sequences(X1, maxlen=27, dtype='int32', value=0)
    

    # output    
    yhat = model.predict_classes(X1)
    yhat
    



  

#Visualization 

    
    df2 = pd.DataFrame({'yhat':yhat})
    
    clean_df['yhat']=df2 
    
    clean_df.to_csv('clean_tweet.csv2',encoding='utf-8')
    csv1 = 'clean_tweet.csv2'
    my_df1 = pd.read_csv(csv1,index_col=0)
    my_df1.head()
    
    
    # ti check null entityI
    my_df1[my_df1.isnull().any(axis=1)].head()
    np.sum(my_df1.isnull().any(axis=1))
    my_df1.isnull().any(axis=0)

    # visualizing the output
    
    # word cloud
    neg_tweets = my_df1[my_df1.yhat== 0]
    neg_string = []
    for t in neg_tweets.text:
        neg_string.append(t)
    neg_string = pd.Series(neg_string).str.cat(sep=' ')
  

    wordcloud = WordCloud(width=1600, height=800,max_font_size=200,background_color="white").generate(neg_string)
    plt.figure(figsize=(12,10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Negative Tweets")
    plt.show()

    pos_tweets = my_df1[my_df1.yhat == 1]
    pos_string = []
    for t in pos_tweets.text:
        pos_string.append(t)
    pos_string = pd.Series(pos_string).str.cat(sep=' ')

    wordcloud = WordCloud(width=1600, height=800,max_font_size=200,colormap='plasma').generate(pos_string)
    plt.figure(figsize=(12,10)) 
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off") 
    plt.title("Positive Tweets")
    plt.show()

# count vectoriser
    from sklearn.feature_extraction.text import CountVectorizer
    cvec = CountVectorizer()
    cvec.fit(my_df1.text.values.astype('U'))
    len(cvec.get_feature_names())

    document_matrix = cvec.transform(my_df1.text.values.astype('U'))
    my_df1[my_df1.yhat == 0].tail()

    # statics
    
    neg_doc_matrix = cvec.transform(my_df1[my_df1.yhat== 0].text.values.astype('U'))
    pos_doc_matrix = cvec.transform(my_df1[my_df1.yhat== 1].text.values.astype('U'))
    neg_tf = np.sum(neg_doc_matrix,axis=0)
    pos_tf = np.sum(pos_doc_matrix,axis=0)
    neg = np.squeeze(np.asarray(neg_tf))
    pos = np.squeeze(np.asarray(pos_tf))
    term_freq_df = pd.DataFrame([neg,pos],columns=cvec.get_feature_names()).transpose()
    term_freq_df.columns = ['negative', 'positive']
    term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['positive']
    term_freq_df.sort_values(by='total', ascending=False).iloc[:10]

    


    # visualiztion using bar chart
    y_pos = np.arange(50)
    plt.figure(figsize=(12,10))
    plt.bar(y_pos, term_freq_df.sort_values(by='positive', ascending=False)['positive'][:50], align='center', alpha=0.5)
    plt.xticks(y_pos, term_freq_df.sort_values(by='positive', ascending=False)['positive'][:50].index,rotation='vertical')
    plt.ylabel('Frequency')
    plt.xlabel('Top 50 positive tokens')
    plt.title('Top 50 tokens in positive tweets')
    
    y_pos = np.arange(50)
    plt.figure(figsize=(12,10))
    plt.bar(y_pos, term_freq_df.sort_values(by='negative', ascending=False)['negative'][:50], align='center', alpha=0.5,color='red')
    plt.xticks(y_pos, term_freq_df.sort_values(by='negative', ascending=False)['negative'][:50].index,rotation='vertical')
    plt.ylabel('Frequency')
    plt.xlabel('Top 50 negative tokens')
    plt.title('Top 50 tokens in negative tweets')
  

   
    dy = pd.DataFrame({'negative': term_freq_df.negative,'positive': term_freq_df.positive})
    plt.figure(figsize=(10,6))
    
    
    ax = sns.regplot(x="negative", y="positive",fit_reg=True, scatter_kws={'alpha':0.01},data=term_freq_df,color='red')
    plt.ylabel('Positive Frequency')
    plt.xlabel('Negative Frequency')
    plt.title('Negative Frequency vs Positive Frequency')
    
    
    # Statistics
    # positive rate
    term_freq_df['pos_rate'] = term_freq_df['positive'] * 1./term_freq_df['total']
    term_freq_df.sort_values(by='pos_rate', ascending=False).iloc[:10]
    
    # positve freq
    term_freq_df['pos_freq_pct'] = term_freq_df['positive'] * 1./term_freq_df['positive'].sum()
    term_freq_df.sort_values(by='pos_freq_pct', ascending=False).iloc[:10]
    
    # harmonic mean
    
    term_freq_df['pos_hmean'] = term_freq_df.apply(lambda x: (hmean([x['pos_rate'], x['pos_freq_pct']])                                                               if x['pos_rate'] > 0 and x['pos_freq_pct'] > 0 else 0), axis=1)
                                                       
    term_freq_df.sort_values(by='pos_hmean', ascending=False).iloc[:10]




#cdf
    
    def normcdf(x):
        return norm.cdf(x, x.mean(), x.std())
    term_freq_df['pos_rate_normcdf'] = normcdf(term_freq_df['pos_rate'])
    term_freq_df['pos_freq_pct_normcdf'] = normcdf(term_freq_df['pos_freq_pct'])
    term_freq_df['pos_normcdf_hmean'] = hmean([term_freq_df['pos_rate_normcdf'], term_freq_df['pos_freq_pct_normcdf']])
    term_freq_df.sort_values(by='pos_normcdf_hmean',ascending=False).iloc[:10]
    
# for neagtive


    term_freq_df['neg_rate'] = term_freq_df['negative'] * 1./term_freq_df['total']
    term_freq_df['neg_freq_pct'] = term_freq_df['negative'] * 1./term_freq_df['negative'].sum()
    term_freq_df['neg_hmean'] = term_freq_df.apply(lambda x: (hmean([x['neg_rate'], x['neg_freq_pct']])     if x['neg_rate'] > 0 and x['neg_freq_pct'] > 0                                                                else 0), axis=1)
                                                          
    term_freq_df['neg_rate_normcdf'] = normcdf(term_freq_df['neg_rate'])
    term_freq_df['neg_freq_pct_normcdf'] = normcdf(term_freq_df['neg_freq_pct'])
    term_freq_df['neg_normcdf_hmean'] = hmean([term_freq_df['neg_rate_normcdf'], term_freq_df['neg_freq_pct_normcdf']])
    term_freq_df.sort_values(by='neg_normcdf_hmean', ascending=False).iloc[:10]


    # Pie chart
    
    
    yhat1=list(yhat)
    neg=yhat1.count(1)
    pos=yhat1.count(0)
    
    senti = pd.DataFrame({'sentiment': [neg,pos]})
    labels =  ['negative','Positive']
    colors = ['blue', 'red']
    plt.figure(figsize=(5,5))
    patches = plt.pie(senti,colors=colors )
    plt.legend(patches, labels)
    plt.pie(senti,labels=labels)
    plt.axis('equal')
    plt.title("Sentiments")
    plt.show() 
    
    #time series
    print("Time vs sentiment Visualization")
    dw = pd.DataFrame({'col':tmp1,'yhat':yhat})
    fig = plt.figure(figsize=(n,5))
    ax = fig.add_subplot(111)
    ax.plot_date(x=dw.col, y=dw.yhat, marker='o',color='red')
    ax.plot(dw.col, dw.yhat,'bo-', linewidth=0.3)
    

#kw = input("enter the keyword:")
#Social_Media_Analysis(kw,10)
