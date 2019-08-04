#Importing libraries
import tweepy
import os
import pandas as pd
import csv
import re
import string
from textblob import TextBlob
import preprocessor as p
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#Twitter app Credentials
consumer_key = '     '
consumer_secret = '    '
access_key= '    '
access_secret = '    '

#Twitter credentials to Tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

#CSV for output tweets
tweets_output = "output_blobText.csv"

#columns of the csv file
cols = ['id','created_at', 'source', 'original_text','clean_text', 'sentiment','polarity','subjectivity','favorite_count', 'retweet_count', 'possibly_sensitive', 'hashtags',
'user_mentions', 'place']

#set two date variables for date range
start_date = '2019-08-01'
end_date = '2019-08-04'

#Emoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])
emoticons = emoticons_happy.union(emoticons_sad)

#Emoji patterns
emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

#mrhod clean_tweets()
def clean_tweets(tweet):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
 
    #after tweepy preprocessing the colon left remain after removing mentions
    #or RT sign in the beginning of the tweet
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)
    #replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)
 
 
    #remove emojis from tweet
    tweet = emoji_pattern.sub(r'', tweet)
 
    #filter using NLTK library append it to a string
    filtered_tweet = [w for w in word_tokens if not w in stop_words]
    filtered_tweet = []
 
    #looping through conditions
    for w in word_tokens:
        #check tokens against stop words , emoticons and punctuations
        if w not in stop_words and w not in emoticons and w not in string.punctuation:
            filtered_tweet.append(w)
    return ' '.join(filtered_tweet)
#Extracting tweets

def extract_tweets(keyword, file):

    df = pd.DataFrame(columns = cols)
    
    for page in tweepy.Cursor(api.search, q=keyword, count = 5, include_rts=False, since=start_date).pages(50):
        for status in page:
            new_entry = []
            status = status._json
            if status['lang'] != 'en':
                continue
            if status['retweeted'] or 'RT @' in status['text']: #To prevent using retweets as original status
                continue
            #below lines updates RT and Fav count
            if status['created_at'] in df['created_at'].values:
                i = df.loc[df['created_at'] == status['created_at']].index[0]
                if status['favorite_count'] != df.at[i, 'favorite_count'] or \
                   status['retweet_count'] != df.at[i, 'retweet_count']:
                    df.at[i, 'favorite_count'] = status['favorite_count']
                    df.at[i, 'retweet_count'] = status['retweet_count']
                continue
            clean_text = clean_tweets(p.clean(status['text']))
            
            #textBlob for Sentiment Calculation
            blob = TextBlob(clean_text)
            sentiment = blob.sentiment
            polarity = sentiment.polarity
            if (polarity>0):
                polarity = 'positive'
            elif (polarity == 0):
                polarity = 'neutral'
            else:
                polarity = 'negative'
            subjectivity = sentiment.subjectivity
            
            new_entry += [status['id'], status['created_at'], status['source'], status['text'],
                          clean_text, sentiment, polarity, subjectivity,
                          status['favorite_count'], status['retweet_count']]
            try:
                is_sensitive = status['possibly_sensitive']
            except KeyError:
                is_sensitive = None
            new_entry.append(is_sensitive)
            
            #hashtags
            hashtags = ", ".join([hashtag_item['text'] for hashtag_item in status['entities']['hashtags']])
            new_entry.append(hashtags)
            #mentions
            mentions = ", ".join([mention['screen_name'] for mention in status['entities']['user_mentions']])
            new_entry.append(mentions)
            
            #location
            try:
                location = status['user']['location']
            except TypeError:
                location = ''
            new_entry.append(location)
            single_tweet_df = pd.DataFrame([new_entry], columns=cols)
            df = df.append(single_tweet_df, ignore_index=True)
            csvFile = open(file, 'a' ,encoding='utf-8', newline = '\n')
    df.to_csv(csvFile, mode='a', columns=cols, index=False, encoding="utf-8")

keywords = '#mcDonalds'

extract_tweets(keywords, tweets_output)
