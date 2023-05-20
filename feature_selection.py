import pandas as pd
from operator import itemgetter
from preprocessing import freqDist, distSpace, prepItems
from parameters import Labels, Threshold, Top_Items


def FeatureSelector(df: pd.DataFrame, cols: list):

    Id = 0
    DistSpace = 2

    Tagged_Sentences = cols[0]
    IDs = cols[1]
    Coordinates = cols[2]
    Vectors_Clusters = cols[3]

    fd = freqDist(df, [Vectors_Clusters, Tagged_Sentences])
    
    ds = distSpace(df, [IDs, Vectors_Clusters, Coordinates])
    
    dic = prepItems(df, [Tagged_Sentences, IDs])
    
    threshold = Threshold
    top_items = Top_Items
    Prominent_Aspects = []
    
    for i in range(0, Labels):
       
        dsl = sorted(ds[i], key=itemgetter(DistSpace))
        dsl = dsl[:top_items]
        
        for j in range(len(dsl)):
            aspect = dic[dsl[j][Id]]
            if fd[i][aspect] >= threshold:
                Prominent_Aspects.append([aspect, dsl[j][Id]])
    


    return Prominent_Aspects
