import pandas as pd
import pyLDAvis.gensim_models
from gensim import corpora, models
from gensim.utils import SaveLoad
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from re import sub


wnl = WordNetLemmatizer()


def removePunc(myWord):
    """Function to remove punctuation from string inputs"""
    if myWord is None:
        return myWord
    else:
        return sub('[.:;()/!&-*@$,?^\d+]', '', myWord)


def removeAscii(myWord):
    """Function to remove ascii from string input"""
    if myWord is None:
        return myWord
    else:
        return str(sub(r'[^\x00-\x7F]+', '', myWord.strip()))


def lemmatize(myWord):
    """Function to lemmatize words"""
    if myWord is None:
        return myWord
    else:
        return str(wnl.lemmatize(myWord))


def removeStopWords(myWord):
    """Function to remove stop words"""
    if myWord is None:
        return myWord
    if myWord not in str(stopwords.words('english')):
        return myWord


def removeLinkUser(myWord):
    """Function to remove web addresses and twitter handles"""
    if not myWord.startswith('@') and not myWord.startswith('http'):
        return myWord


def prepText(myWord):
    """Final text pre-processing function"""
    return removeStopWords(
        lemmatize(
            removeAscii(
                removePunc(
                    removeLinkUser(
                        myWord.lower()
                    )
                )
            )
        )
    )


def filterTweetList(tweetList):
    """Remove stop words, lemmatize, and clean all tweets"""
    return [[prepText(word) for word
                in tweet.split()
                    if prepText(word) is not None]
                for tweet in tweetList]


def makeDict(myTweetList):
    """Create dictionary from list of tokenized documents"""
    return corpora.Dictionary(myTweetList)


def makeCorpus(myTweetList,myDict):
    """Create corpus from list of tokenized documents"""
    return [myDict.doc2bow(tweet) for tweet in myTweetList]


def createLDA(myCorpus, myDictionary, myTopics=50, myPasses=10, myIterations=50, myAlpha=0.001):
    """LDA model call function"""
    return models.LdaMulticore(myCorpus, id2word=myDictionary, num_topics=myTopics, passes=myPasses,
                               iterations=myIterations, alpha=myAlpha)


def topicModeling():
    df1 = pd.read_csv(filepath_or_buffer='Tweets/TweetsSett1-en.csv', sep='^', encoding='utf-8')
    df2 = pd.read_csv(filepath_or_buffer='Tweets/TweetsSett2-en.csv', sep='^', encoding='utf-8')
    df3 = pd.read_csv(filepath_or_buffer='Tweets/TweetsSett3-en.csv', sep='^', encoding='utf-8')

    tweet_list = []

    for i in range(df1.shape[0]):
        if df1.loc[i][' text '] != '^^^^':
            tweet_list.append(df1.loc[i][' text '].replace('https:', '').replace('t.co', ''))

    for i in range(df2.shape[0]):
        if df2.loc[i][' text '] != '^^^^':
            tweet_list.append(df2.loc[i][' text '].replace('https:', '').replace('t.co', ''))

    for i in range(df3.shape[0]):
        if df3.loc[i][' text '] != '^^^^':
            tweet_list.append(df3.loc[i][' text '].replace('https:', '').replace('t.co', ''))


    #print(tweet_list)
    cleanList = filterTweetList(tweet_list)

    '''
    #run this after the first run
    kagLda = SaveLoad.load('LDAmodel') #generazione del modello
    kagDict = corpora.Dictionary.load('Dictionary.dict') #generazione del dizionario
    kagCorpus = corpora.MmCorpus('Corpus.mm') #testo sotto analisi
    #end
    '''


    #run this at the first run
    # """Create model objects"""
    kagDict = makeDict(cleanList)
    kagCorpus = makeCorpus(cleanList, kagDict)
    kagLda = createLDA(kagCorpus, kagDict)

    # """Save model objects"""
    SaveLoad.save(kagLda, 'LDAmodel')
    corpora.MmCorpus.serialize('Corpus.mm', kagCorpus)
    kagDict.save('Dictionary.dict')
    #end


    ldaViz = pyLDAvis.gensim_models.prepare(kagLda, kagCorpus, kagDict, mds='mmds')
    pyLDAvis.save_html(ldaViz, 'TopicVisualization.html')


if __name__ == '__main__':
    topicModeling()