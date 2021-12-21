import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df=pd.read_csv("wine-quality-white-and-red.csv", index_col=0)
df.head()

fig=plt.figure(figsize=(10,8))
ax=fig.add_subplot()

ax.set_title('Ubicación de crímenes', pad=15)
ax.set_xlabel('alcohol')
ax.set_ylabel('fixed acidity')

sns.scatterplot(df['alcohol'], df['fixed acidity'], ax=ax)




k_means=KMeans(n_clusters=6,max_iter=2000)
k_means.fit(df[["alcohol","fixed acidity"]])
centers=k_means.cluster_centers_
centers

fig=plt.figure(figsize=(10,8))
ax = fig.add_subplot()

ax.set_title('Calidad de vinos por cantidad de alcohol  y acides', pad=15)
ax.set_xlabel('Alcohol')
ax.set_ylabel('fixed acidity')

sns.scatterplot(df['alcohol'], df['fixed acidity'], ax=ax, palette='rainbow');
sns.scatterplot(centers[:,0], centers[:,1], ax=ax, s=100, color='black');

clasificaciones = k_means.predict(df[['alcohol', 'fixed acidity']])

fig=plt.figure(figsize=(10,8))
ax = fig.add_subplot()

ax.set_title('Calidad de vinos por cantidad de alcohol  y acides', pad=15)
ax.set_xlabel('Alcohol')
ax.set_ylabel('fixed acidity')

sns.scatterplot(df['alcohol'], df['fixed acidity'], ax=ax, hue=clasificaciones, palette='rainbow');
sns.scatterplot(centers[:,0], centers[:,1], ax=ax, s=100, color='black');

ax.get_legend().remove()