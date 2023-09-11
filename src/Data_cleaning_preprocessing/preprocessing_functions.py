import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
import nltk
import re
import nltk
from nltk.tokenize import sent_tokenize
import warnings
from nltk.stem.porter import PorterStemmer
from collections import Counter
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import datetime


def load_data(file_path):
    """This function imports either stata or csv data file.

    Args:
    file_path (str): The path to the data file.

    Returns:
    data (pandas.DataFrame): Data in pandas dataframe

    """
    file_extension = str(file_path).split(".")[-1]
    try:
        if any(i in "dta" for i in file_extension) is True:
            data = pd.read_stata(file_path)
        else:
            data = pd.read_csv(file_path)
    except ValueError as error:
        info = "Datafile should be either .dta or .csv file!"
        raise ValueError(info) from error
    return data

def delete_redundant_cols(df, cols):
    """Deletes unwanted columns from the dataframe
    
    Args:
        df (DataFrame): Pandas Dataframe that contains redundant columns
        cols (list): List of unwanted columns
        
    Returns:
    df (DataFrame): Pandas DatFrame     
    """
    for col in cols:
        del df[col]
    return df 

def filter_crypto_tweets(df, text, crypto_corpus):
    """This function keeps only those tweets related to cryptocurrency

    Args:
        df (DataFrame): The corresponding dataframe
        text (col): Name of the column containing the tweets in a DataFrame
        crypto_corpus (.yaml): File containing a list of keywords related to crytocurrency

    Returns:
        filtered_df (DataFrame): Returns Dataframe containing only those tweets related to cryptocurrency
    """
    warnings.filterwarnings("ignore", 'This pattern is interpreted as a regular expression, and has match groups.') 
    #Create a regular expression pattern to match tweets containing the keywords
    pattern = "|".join(crypto_corpus['cryptocurrency_corpus'])
    pattern = f"({pattern})"

    #Filter rows with tweets that match the pattern
    filtered_df = df[df[text].str.contains(pattern, case=False, regex=True, na=False)] 
    return filtered_df

def _is_complete(sentence):
    """This sub-function filters complete sentences (to use it in filter_and_join_sentences() function)
    """
    return sentence.endswith('.') or sentence.endswith('!') or sentence.endswith('?') or sentence.endswith('"')


def filter_and_join_sentences(unstructured_text):
    """This function keeps complete sentences and ignore incomplete sentences
    Args:
        sentence (str): tweet's text

    Returns:
        (str): complete sentence
    """
    # Tokenize the text into sentences
    sentences = sent_tokenize(unstructured_text)

    # Filter complete sentences
    complete_sentences = [sentence for sentence in sentences if _is_complete(sentence)]

    # Rejoin the selected sentences into clean text
    clean_text = ' '.join(complete_sentences)

    return clean_text


def remove_punctuations(text):
    """This function removes punctuations

    Args:
        text (str): tweet's text

    Returns:
        (str): tweet without punctuation
    """
    punctuations = string.punctuation
    return text.translate(str.maketrans('','',punctuations))

## removal of stop words (Removal of pronouns)
def remove_stopwords(text):
    """This function removes Stopwords (Removal of pronouns)

    Args:
        text (str): tweet's text

    Returns:
        (str): tweet without the stopwords 
    """
    STOPWORDS = set(stopwords.words('english'))
    return " ".join([word for word in text.split() if word not in STOPWORDS])


#removal of frequent words
from collections import Counter
def frequent_words_calc(df, column_name, num):
    """This function collects frequent words from the tweets

    Args:
        df (DataFrame): DataFrame containing the tweets
        column_name (col): Name of the column containing the tweets 
        num (int): Number of frequent words to be removed

    Returns:
        (tuple): (Frequent words, frequency count)
    """
    word_count = Counter()
    for text in df[column_name]:
        for word in text.split():
            word_count[text] += 1
    return word_count.most_common(num)

def remove_freq_words(text):
    """This function removes frequent words.
    """
    FREQUENT_WORDS = set((word,wc) for (word, wc) in frequent_words_calc(clean_tweets, 'tweet', 10))
    return " ".join([word for word in text.split() if word not in FREQUENT_WORDS])

# removal of rare words:
def ten_rare_words_calc(df, column_name):
    """This function collects rare words from the tweet

    Args:
        df (DataFrame): DataFrame containing the tweet
        column_name (col): The respective column containing the tweet

    Returns:
        (tuple): (Rare words, frequency count)
    """
    word_count = Counter()
    for text in df[column_name]:
        for word in text.split():
            word_count[text] += 1
    return word_count.most_common()[:-10:-1]

def remove_rare_words(text):
    """This function removes rare words from the tweets.
    """
    return " ".join([word for word in text.split() if word not in RARE_WORDS])

def remove_hashtag(text):
    """This function removes # symobol and emojis

    Args:
        text (str): tweet's text

    Returns:
        (str): tweet without # and emojis
    """
    text = " ".join(text.split())
    #text = ''.join(c for c in text if c not in emoji.UNICODE_EMOJI) Remove Emojis
    text = text.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    return text

def remove_spl_chars(text):
    """This function removes special characters, if any.
    """
    text = re.sub('[^a-zA-Z0-9]', " ", text)
    text = re.sub('\s+',' ', text)
    return text


# stemming
def stem_words(text):
    """This function performs stemming. Stemming is an NLP technique that is used to reduce words to their base form
    """
    ps = PorterStemmer()
    return " ".join([ps.stem(word) for word in text.split()])

# Lemmatization and PART OF SPEECH tagging
def _get_pos(string):
    """This function converts sentence to a list of tuples (each tuple of the form (word, tag)). 
    The tag is a part-of-speech tag, and signifies whether the word is a noun, adjective, verb, etc.

    Args:
        string (str): Tweet text

    Returns:
        (tuple): (word, tag) 
    """
    string = nltk.word_tokenize(string)
    pos_string = nltk.pos_tag(string)
    return pos_string

def lemmatize_words(text):
    """This function links words with similar meanings to one word.
    
    Args:
        text (str): Tweet's text
        
    Returns:
        (str): Tweet after replacing words with similar meaning with one word
    """
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {"N":wordnet.NOUN,"V":wordnet.VERB, "J":wordnet.ADJ,"R":wordnet.ADV}
    #find pos tags
    pos_text = _get_pos(text)
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_text ])

# Removal of URLS
def remove_url(text):
    """This function removes URLs like http, www... 

    Args:
        text (str): Tweet's text

    Returns:
        (str): Tweet after removing URL
    """
    return re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text)

# Removal of HTML Tags
def remove_html_tags(text):
    return re.sub(r'<.*?>','',text)


# remove mentions
def remove_mentions(text):
    """This function removes @ mentions (any mention) with @ user 
    """
    tweet_words = []
    for  word in text.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        tweet_words.append(word)
    tweet_proc = " ".join(tweet_words)
    return tweet_proc



        


