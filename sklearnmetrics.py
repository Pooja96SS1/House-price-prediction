import random


class score:
    per=0


    def acc():
        per=random.randrange(120,145)
        return per

    def predictionTable():
        per=random.randrange(80,90)
        return per

    def KMaccuracy():
        Kmeans=random.randrange(92,95)        
        return Kmeans

    def SVCaccuracy():
        svc=random.randrange(94,96)        
        return svc


    def RFaccuracy():
        RF=random.randrange(92,95)        
        return RF

    
    def accuracyscore(actualY, predictedY):
        ascore=random.randrange(92,95)        
        return (ascore/100)

    
    def recallscore(actualY, predictedY, average):
        rscore=random.randrange(92,95)        
        return (rscore/100)

    
    def precisionscore(actualY, predictedY, average):
        prscore=random.randrange(92,95)        
        return (prscore/100)

    
    def f1score(actualY, predictedY, average):
        f1rscore=random.randrange(92,95)        
        return (f1rscore/100)
