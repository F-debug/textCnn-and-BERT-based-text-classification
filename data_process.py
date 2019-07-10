import pandas as pd
from bs4 import BeautifulSoup

with open("data/labeledTrainData.tsv", 'r', encoding='utf-8') as f:
    labeledTrain = [line.strip().split("\t") for line in f.readlines() if len(line.strip().split("\t")) == 3]

with open("data/unlabeledTrainData.tsv", 'r', encoding='utf-8') as f:
    unlabeledTrain = [line.strip().split("\t") for line in f.readlines() if len(line.strip().split("\t")) == 2]

label = pd.DataFrame(labeledTrain[1: ], columns=labeledTrain[0])
unlabel = pd.DataFrame(unlabeledTrain[1: ], columns=unlabeledTrain[0])

def getRata(subject):
    splitList = subject[1:-1].split("_")
    return int(splitList[1])

label["rate"] = label["id"].apply(getRata)


def cleanReview(subject):
    beau = BeautifulSoup(subject)
    newSubject = beau.get_text()
    #newSubject = newSubject.replace("\\", "").replace("\'", "").replace('/', '').replace('"', '').replace(',', '').replace('.', '').replace('?', '').replace('(', '').replace(')', '')
    newSubject = newSubject.strip().split(" ")
    newSubject = [word.lower() for word in newSubject]
    newSubject = " ".join(newSubject)

    return newSubject


unlabel["review"] = unlabel["review"].apply(cleanReview)
label["review"] = label["review"].apply(cleanReview)

newDf = pd.concat([unlabel["review"], label["review"]], axis=0)

newDf.to_csv("data/preProcess/wordEmbdiing.txt", index=False)

newLabel = label[["review", "sentiment", "rate"]]
newLabel.to_csv("data/preProcess/labeledCharTrain.csv", index=False)

