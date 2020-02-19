import nltk
from nltk.corpus import stopwords

#nltk.download('stopwords')

#used to see which words were in prebuilt list by nltk
#last couple words had biases: lowered accuracy to .79. After removing, accuracy was .82
print(stopwords.words("english"))