import pandas as pd
from sklearn.cluster import KMeans

Data = pd.read_csv("db.csv")
print(Data)
Data.drop(["Orden","Original Title","Company","Budget","Opening Weekend USA","GrossUSA","Gross Worldwide"],axis = 1, inplace = True)
print(Data)

#---------------KMeans----------------
km = KMeans(n_clusters = 7)
clus = km.fit_predict(Data)
print("Estas son las variables con los clusters:")
print(clus)
Data["Clusters"] = clus
print("Data integrada con cluster:")
print(Data)
print("--------------------------------------------------")
copia = pd.read_csv("db.csv")
Data["Original Title"] = copia["Original Title"]
print(Data)


