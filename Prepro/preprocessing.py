#SEGMENTATION

import nltk  # Natural Language Toolkit for text processing
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
import re  # Regular expressions for text cleaning
from nltk import ne_chunk

nltk.download('punkt_tab')

# Original Text
text = """Apple Inc. is an American multinational technology company headquartered in
        Cupertino, California that designs, develops,
        and sells consumer electronics, computer software,
        and online services. The company's hardware products include the iPhone smartphone,
        the iPad tablet computer, the Mac personal computer, the iPod portable media player,
        the Apple smartwatch, and the Apple TV digital media player."""

sentences = sent_tokenize(text)


#TOKENIZATION
# First sentence
print("\nFirst Sentence:")
print(sentences[0])

text = re.sub(r"[^a-zA-Z0-9]", " ", sentences[0])


# TOKENIZATION
words = word_tokenize(text)

#REMOVING STOP WORD
stop_words = set(stopwords.words("english"))
filtered_words = [w for w in words if w not in stop_words]

#STEMMING
stemmer = PorterStemmer()
stemmed = [stemmer.stem(w) for w in filtered_words]

#LEMMATIZATION
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(w) for w in filtered_words]

#PART OF SPEECH TAGGING
tagged_words = pos_tag(filtered_words)

#NAMED ENTITY RECOGNITION
ner_tree = ne_chunk(pos_tag(word_tokenize(sentences[0])))