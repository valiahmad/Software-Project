import pandas as pd
from operator import itemgetter
from preprocessing import freqDist, distSpace, prepItems
from clustering import Labels


def FeatureSelector(df: pd.DataFrame, cols: list):

    Id = 0
    DistSpace = 1

    Tagged_Sentences = cols[0]
    IDs = cols[1]
    Coordinates = cols[2]
    Vectors_Clusters = cols[3]

    fd = freqDist(df[[Vectors_Clusters, Tagged_Sentences]], 
             [Vectors_Clusters, Tagged_Sentences])
    
    ds = distSpace(df[[IDs, Vectors_Clusters, Coordinates]],
              [IDs, Vectors_Clusters, Coordinates])
    
    
    dic = prepItems(df[[Tagged_Sentences, IDs]], [Tagged_Sentences, IDs])
    threshold = 0
    top_items = 0
    Prominent_Aspects = []
    
    for i in range(0, Labels):
       
        dsl = sorted(ds[i], key=itemgetter(DistSpace))
        dsl = dsl[:top_items]
        
        for j in range(len(dsl)):
            aspect = dic[dsl[j][Id]]
            if fd[i][aspect] >= threshold:
                Prominent_Aspects.append([aspect, dsl[j][Id]])



    return Prominent_Aspects
