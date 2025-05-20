import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

nltk.download('stopwords')

abbreviations = {
    "u": "you",
    "r": "are",
    "ur": "your",
    "lol": "laughing out loud",
    "omg": "oh my god",
    "idk": "i don't know",
    "btw": "by the way",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "brb": "be right back",
    "b4": "before",
    "b/c": "because",
    "bc": "because",
    "lmk": "let me know",
    "smh": "shaking my head",
    "tbh": "to be honest",
    "thx": "thanks",
    "ty": "thank you",
    "np": "no problem",
    "plz": "please",
    "pls": "please",
    "gr8": "great",
    "2moro": "tomorrow",
    "2day": "today",
    "w/": "with",
    "w/o": "without",
    "ya": "you",
    "cuz": "because",
    "cos": "because",
    "afaik": "as far as i know",
    "fyi": "for your information",
    "rn": "right now",
    "ikr": "i know right",
    "gtg": "got to go",
    "g2g": "got to go",
    "dm": "direct message",
    "l8r": "later",
    "omw": "on my way",
    "yolo": "you only live once",
    "rofl": "rolling on the floor laughing",
    "bff": "best friends forever",
    "xoxo": "hugs and kisses",
    "ttyl": "talk to you later",
    "fml": "fuck my life",
    "tfw": "that feeling when",
    "nvm": "never mind",
    "idc": "i don't care",
    "ily": "i love you"
}

def replace_abbreviations(tweet, abbreviations):
    tokens = tweet.split()
    tokens = [abbreviations.get(token, token) for token in tokens]
    return ' '.join(tokens)



def preprocess_tweet(tweet, rejoin=False):
    """Fonction de traitement d'un tweet.
    Entrée :
        tweet : une chaîne de caractères contenant un tweet
    Sortie :
        tweet_nettoye : une liste de mots contenant le tweet traité
    """
    stemmer = PorterStemmer()
    stopwords_anglais = stopwords.words('english')

    # Suppression des anciens retweets "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    # Suppression des liens hypertextes
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    # Suppression du symbole #
    tweet = re.sub(r'#', '', tweet)
    
    # Remplacer les abréviations
    tweet = replace_abbreviations(tweet, abbreviations)

    # Tokenisation du tweet
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tokens_tweet = tokenizer.tokenize(tweet)

    tweet_nettoye = []
    for mot in tokens_tweet:
        if (mot not in stopwords_anglais and mot not in string.punctuation):
            mot_racine = stemmer.stem(mot)
            tweet_nettoye.append(mot_racine)

    if rejoin:
        tweet_nettoye = ' '.join(tweet_nettoye)

    return tweet_nettoye



def preprocess_tweet_bert(tweet):
    """Fonction de traitement d'un tweet pour BERT.
    Entrée :
        tweet : une chaîne de caractères contenant un tweet
    Sortie :
        tweet : une chaîne de caractères contenant le tweet traité
    """
    tweet = tweet.lower()

    # Remplacer URLs et nombres
    tweet = re.sub(r"http\S+|www.\S+|https\S+", "[URL]", tweet, flags=re.MULTILINE)
    tweet = re.sub(r"\d+", "[NUMBER]", tweet)

    # Remplacer les abréviations
    tweet = replace_abbreviations(tweet, abbreviations)

    # Enlever espaces multiples
    tweet = re.sub(r"\s+", " ", tweet).strip()

    return tweet