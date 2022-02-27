import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import time
from datetime import date
from textblob import TextBlob
from feel_it import EmotionClassifier, SentimentClassifier
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from config import DefaultConfig
CONFIG = DefaultConfig()


def percentage(part, total):
    return 100 * float(part)/float(total)


def clean_tweet(text):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z'àèìòùé?!,&.:;%€()<>=$\" \t])| (\w+:\ / \ / \S+)", " ", text)
                    .split())


def searchTweets(path):
    #authentication
    consumerKey = CONFIG.CONS_KEY
    consumerSecret = CONFIG.CONS_SECR
    accessToken = CONFIG.ACCESS_TOKEN
    accessTokenSecret = CONFIG.ACCESS_TOKEN_SECRET

    auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
    auth.set_access_token(accessToken, accessTokenSecret)
    api = tweepy.API(auth)

    keywords = ['#greenpass', '#supergreenpass', '#terzadose', '#quartadose', '#vaccino', '#vaccinatevi_e_basta',
               '#vaccini', '#dittaturasanitaria', '#vaccinoobbligatorio', '#Covid19', '#omicron',
               '#nessunacorrelazione', '#pandemia', '#lockdown', '#terzadoseoniente',
               '#obbligovaccinale', '#quartaondata', '#vaccinokiller', '#RestiamoUmani', '#covid', '#greenpasspremium',
               '#greenpassobbligatorio', "#iostoconlascienza", '#novax', '#sivax', '#booster']
    noOfTweet = 100
    dataOggi = date.today().strftime("%d/%m/%Y")
    listOfLists =[[]]

    for keyword in keywords:
        tweets = tweepy.Cursor(api.search_tweets, q=keyword, tweet_mode='extended', lang='it').items(noOfTweet)

        for tweet in tweets:
            # check per controllare se è un retweet ed eventualmente accedere al
            # full_text dello stesso per evitare troncamenti
            if 'retweeted_status' in dir(tweet):
                text = tweet.retweeted_status.full_text
            else:
                text = tweet.full_text

            listOfLists.append([tweet.id_str, clean_tweet(text), tweet.user.location, tweet.user.id_str, dataOggi])

    df = pd.DataFrame(listOfLists, columns=['id', 'text', 'location', 'idUser', 'date'])
    df.drop_duplicates(inplace=True, subset=['text'])
    df.to_csv(path_or_buf=path, sep='^', encoding='utf-8', mode='a', index=False, header=False)


def sentimentAndEmotionAnalysis():
    dfIta = pd.read_csv(filepath_or_buffer='Tweets/TweetsSett3.csv', sep='^', encoding='utf-8')
    dfEng = pd.read_csv(filepath_or_buffer='Tweets/TweetsSett3-en.csv', sep='^', encoding='utf-8')
    emotionIta = EmotionClassifier()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    dfIta.drop_duplicates(inplace=True, subset=['text'])
    dfEng.drop_duplicates(inplace=True, subset=[' text '])

    positive = 0; negative = 0; neutral = 0
    tweet_list_ita = []; tweet_list_eng = []; neutral_list = []; negative_list = []; positive_list = []

    for i in range(dfIta.shape[0]):
        if dfIta.loc[i]['text'] != '^^^^':
            tweet_list_ita.append(dfIta.loc[i]['text'])

    for i in range(dfEng.shape[0]):
        if dfEng.loc[i][' text '] != '^^^^':
            tweet_list_eng.append(dfEng.loc[i][' text '])

    print('1. Liste ita ed eng aggiornate: ')

    # formattazione cella [nome_regione, #negativi, #neutrali, #positivi]
    regioni = [['Abruzzo',0,0,0],
               ['Basilicata',0,0,0],
               ['Calabria',0,0,0],
               ['Campania',0,0,0],
               ['Emilia Romagna',0,0,0],
               ['Friuli Venezia Giulia',0,0,0],
               ['Lazio',0,0,0],
               ['Liguria',0,0,0],
               ['Lombardia',0,0,0],
               ['Marche',0,0,0],
               ['Molise',0,0,0],
               ['Piemonte',0,0,0],
               ['Puglia',0,0,0],
               ['Sardegna',0,0,0],
               ['Sicilia',0,0,0],
               ['Toscana',0,0,0],
               ['Trentino Alto Adige',0,0,0],
               ['Umbria',0,0,0],
               ["Valle d'Aosta",0,0,0],
               ['Veneto',0,0,0]]

    start = time.time()
    emotionList = emotionIta.predict(tweet_list_ita)
    end = time.time()

    print('2. Predict effettuata: ' + str(end-start))

    start = time.time()
    j = int(0)
    for single_tweet in tweet_list_eng:
        analysis = TextBlob(single_tweet)

        tweetLocation = dfIta.loc[j]['location']

        if analysis.sentiment.polarity < 0:
            negative_list.append(single_tweet)
            negative += 1

            if type(tweetLocation) is str:
                if 'Rome' in tweetLocation or 'Roma' in tweetLocation:
                    regioni[6][1] = regioni[6][1] + 1
                elif 'Naples' in tweetLocation or 'Napoli' in tweetLocation:
                    regioni[3][1] = regioni[3][1] + 1
                elif 'Milan' in tweetLocation or 'Milano' in tweetLocation:
                    regioni[8][1] = regioni[8][1] + 1
                elif 'Turin' in tweetLocation or 'Torino' in tweetLocation:
                    regioni[11][1] = regioni[11][1] + 1
                else:
                    for loc in regioni:
                        if loc[0] in tweetLocation:
                            loc[1] += 1
        elif analysis.sentiment.polarity > 0:
            positive_list.append(single_tweet)
            positive += 1

            if type(tweetLocation) is str:
                if 'Rome' in tweetLocation or 'Roma' in tweetLocation:
                    regioni[6][3] = regioni[6][3] + 1
                elif 'Naples' in tweetLocation or 'Napoli' in tweetLocation:
                    regioni[3][3] = regioni[3][3] + 1
                elif 'Milan' in tweetLocation or 'Milano' in tweetLocation:
                    regioni[8][3] = regioni[8][3] + 1
                elif 'Turin' in tweetLocation or 'Torino' in tweetLocation:
                    regioni[11][3] = regioni[11][3] + 1
                else:
                    for loc in regioni:
                        if loc[0] in tweetLocation:
                            loc[3] += 1
        else:
            if emotionList[j] == 'joy':
                emotionList[j] = 'neutral'
                neutral_list.append(single_tweet)
                neutral += 1
                pos = 2
            else:
                negative_list.append(single_tweet)
                negative += 1
                pos = 1

            if type(tweetLocation) is str:
                if 'Rome' in tweetLocation or 'Roma' in tweetLocation:
                    regioni[6][pos] = regioni[6][pos] + 1
                elif 'Naples' in tweetLocation or 'Napoli' in tweetLocation:
                    regioni[3][pos] = regioni[3][pos] + 1
                elif 'Milan' in tweetLocation or 'Milano' in tweetLocation:
                    regioni[8][pos] = regioni[8][pos] + 1
                elif 'Turin' in tweetLocation or 'Torino' in tweetLocation:
                    regioni[11][pos] = regioni[11][pos] + 1
                else:
                    for loc in regioni:
                        if loc[0] in tweetLocation:
                            loc[pos] += 1
        j = j + 1

    end = time.time()
    print('3. for effettuato: ' + str(end - start))

    print('total  number: ', len(tweet_list_eng))
    print('positive number: ', len(positive_list))
    print('negative number: ', len(negative_list))
    print('neutral number: ', len(neutral_list))
    #print(positive_list)
    #print(neutral_list)
    #print(negative_list)
    createSentimentPiecart(percentage(len(positive_list), len(tweet_list_eng)),
                  percentage(len(neutral_list), len(tweet_list_eng)),
                  percentage(len(negative_list), len(tweet_list_eng)))
    createEmotionPiecart(emotionList)
    for i in range(len(regioni)):
        print('Regione ' + str(regioni[i][0]) + ' = negativi: ' + str(regioni[i][1]) + ', neutrali: ' + str(regioni[i][2]) + ', positivi: ' + str(regioni[i][3]))


def createSentimentPiecart(positive, neutral, negative):
    labels = ['Positive['+str(positive)[0:4] +' %]', 'Neutral['+str(neutral)[0:4] +' %]', 'Negative['+str(negative)[0:4] +' %]']
    sizes = [positive, neutral, negative]
    colors = ['orange', 'blue', 'red']
    patches, texts = plt.pie(sizes, colors=colors, startangle=90)
    plt.style.use('default')
    plt.legend(labels)
    plt.title('Sentiment Analysis Result')
    plt.axis('equal')
    plt.show()


def createEmotionPiecart(emotions):
    joy = 0; fear = 0; neutral = 0; sadness = 0; anger = 0;
    for i in range(len(emotions)):
        if emotions[i] == 'joy':
            joy+=1
        elif emotions[i] == 'fear':
            fear+=1
        elif emotions[i] == 'sadness':
            sadness+=1
        elif emotions[i] == 'anger':
            anger+=1
        else:
            neutral+=1

    labels = ['Joy['+str(joy) + ']', 'Neutral['+str(neutral) + ']', 'Fear['+str(fear) + ']', 'Sadness['+str(sadness) + ']', 'Anger['+str(anger) + ']']
    sizes = [joy, neutral, fear, sadness, anger]
    colors = ['orange', 'blue', 'red', 'green', 'yellow']
    patches, texts = plt.pie(sizes, colors=colors, startangle=90)
    plt.style.use('default')
    plt.legend(labels)
    plt.title('Emotion Analysis Result')
    plt.axis('equal')
    plt.show()


def create_wordtweet(list):
    text = ''
    for t in list:
        text += (' ' + t)
    mask = np.array(Image.open('wordcloud/twitter_mask.png'))
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color='white',
        mask=mask,
        max_words=3000,
        stopwords=stopwords,
        contour_width=2,
        contour_color='blue',
        colormap='gist_ncar',
        repeat=True).generate(text)

    plt.figure()
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    # store to file
    plt.savefig("wordcloud/twitter.png", format="png")


def test_accuracy():
    dfIta = pd.read_csv(filepath_or_buffer='Tweets/prova.csv', sep='^', encoding='utf-8')
    dfEng = pd.read_csv(filepath_or_buffer='Tweets/prova-en.csv', sep='^', encoding='utf-8')
    emotionIta = EmotionClassifier()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    contatoreTotale = 0; contatoreCorretti = 0

    positive = 0;
    negative = 0;
    neutral = 0
    tweet_list_ita = [];
    tweet_list_eng = [];
    neutral_list = [];
    negative_list = [];
    positive_list = []

    for i in range(dfIta.shape[0]):
        if dfIta.loc[i]['text'] != '^^^^':
            tweet_list_ita.append(dfIta.loc[i]['text'])

    for i in range(dfEng.shape[0]):
        if dfEng.loc[i][' text '] != '^^^^':
            tweet_list_eng.append(dfEng.loc[i][' text '])


    emotionList = emotionIta.predict(tweet_list_ita)

    j = int(0)
    for single_tweet in tweet_list_eng:
        analysis = TextBlob(single_tweet)

        if analysis.sentiment.polarity < 0:
            negative_list.append(single_tweet)
            negative += 1
        elif analysis.sentiment.polarity > 0:
            positive_list.append(single_tweet)
            positive += 1
        else:
            if emotionList[j] == 'joy':
                emotionList[j] = 'neutral'
                neutral_list.append(single_tweet)
                neutral += 1
            else:
                negative_list.append(single_tweet)
                negative += 1

        print(tweet_list_ita[j] + '\n||| emozione: ' + str(emotionList[j]) + '||| sentimento: ' + str(analysis.sentiment.polarity))
        if input('inserire si per conteggiare come corretto, no altrimenti') == 'si':
            contatoreCorretti += 1

        contatoreTotale += 1;

        j = j + 1
    print(contatoreCorretti)
    print(contatoreTotale)


if __name__ == '__main__':
    print("replace this instruction to run your script")

