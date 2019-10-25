import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import itertools
import nltk
nltk.download('punkt')
import wordcloud
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings("ignore")


data = pd.read_excel('r12mo_no_blk_pop.xlsx')
#print (data.info())
#print (data.head(3), data.shape)

info = pd.DataFrame(data = data.groupby(['Property'])['OrderID'].nunique(), index=data.groupby(['Property']).groups.keys()).T
#print (info)

# StockCode Feature ->
# We will see how many different products were sold in the year data was collected.
print(len(data['Item_Num'].value_counts()))

# Transanction feature
# We will see how many different transanctions were done.
print(len(data['OrderID'].value_counts()))

# Transanction feature
# We will see how many different Customers are there.
print(len(data['Cust_Num'].value_counts()))

pd.DataFrame({'products':len(data['Item_Num'].value_counts()),
              'transanctions':len(data['OrderID'].value_counts()),
              'Customers':len(data['Cust_Num'].value_counts())},
             index = ['Quantity'])

df = data.groupby(['Cust_Num', 'Item_Num'], as_index=False)['Sale_Date'].count()
df = df.rename(columns = {'Sale_Date':'Number of products'})
#df[:10].sort_values('Cust_Num')

temp = data.groupby(by=['Cust_Num', 'OrderID'], as_index=False)['Amount'].sum()
basket_price = temp.rename(columns = {'Amount': 'Basket Price'})

data['Sale_Date_int'] = data['Sale_Date'].astype('int64')
temp = data.groupby(by=['Cust_Num', 'OrderID'], as_index=False)['Sale_Date_int'].mean()
data.drop('Sale_Date_int', axis = 1, inplace=True)
basket_price.loc[:, 'Sale_Date'] = pd.to_datetime(temp['Sale_Date_int'])

basket_price = basket_price[basket_price['Basket Price'] > 0]
print (basket_price.sort_values('Cust_Num')[:6])

price_range = [0, 50, 100, 200, 500, 1000, 5000, 50000]
count_price = []
for i,price in enumerate(price_range):
    if i==0:continue
    val = basket_price[(basket_price['Basket Price'] < price)&
                       (basket_price['Basket Price'] > price_range[i-1])]['Basket Price'].count()
    count_price.append(val)

plt.rc('font', weight='bold')
f, ax = plt.subplots(figsize=(11, 6))
colors = ['yellowgreen', 'gold', 'wheat', 'c', 'violet', 'royalblue', 'firebrick']
labels = ["{}<.<{}".format(price_range[i-1], s) for i,s in enumerate(price_range) if i != 0]
sizes = count_price
explode = [0.0 if sizes[i] < 100 else 0.0 for i in range(len(sizes))]
ax.pie(sizes, explode = explode, labels = labels, colors = colors,
       autopct = lambda x:'{:1.0f}%'.format(x) if x > 1 else '',
       shadow = False, startangle = 0)
ax.axis('equal')
f.text(0.5, 1.01, "Distribution of order amounts", ha = 'center', fontsize = 18)
#plt.show()

is_noun = lambda pos:pos[:2] == 'NN'

def keywords_inventory(dataframe, colonne = 'Desc'):
    import nltk
    stemmer = nltk.stem.SnowballStemmer("english")
    keywords_roots = dict()
    keywords_select = dict()
    category_keys = []
    count_keywords = dict()
    icount = 0

    for s in dataframe[colonne]:
        if pd.isnull(s): continue
        lines = s.lower()
        tokenized = nltk.word_tokenize(lines)
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]

        for t in nouns:
            t = t.lower() ; racine = stemmer.stem(t)
            if racine in keywords_roots:
                keywords_roots[racine].add(t)
                count_keywords[racine] += 1
            else:
                keywords_roots[racine] = {t}
                count_keywords[racine] = 1


    for s in keywords_roots.keys():
        if len(keywords_roots[s]) > 1:
            min_length = 1000
            for k in keywords_roots[s]:
                if len(k) < min_length:
                    clef = k ; min_length = len(k)

            category_keys.append(clef)
            keywords_select[s] = clef

        else:
            category_keys.append(list(keywords_roots[s])[0])
            keywords_select[s] = list(keywords_roots[s])[0]

    print("Number of keywords in the variable '{}': {}".format(colonne, len(category_keys)))
    return category_keys, keywords_roots, keywords_select, count_keywords

nltk.download('averaged_perceptron_tagger')

df_produits = pd.DataFrame(data['Desc'].unique()).rename(columns = {0:"Desc"})
keywords, keywords_roots, keywords_select, count_keywords = keywords_inventory(df_produits)

list_products = []
for k, v in count_keywords.items():
    word = keywords_select[k]
    list_products.append([word, v])

liste = sorted(list_products, key = lambda x:x[1], reverse=True)

plt.rc('font', weight='normal')
fig, ax = plt.subplots(figsize=(7, 25))
y_axis = [i[1] for i in liste[:125]]
x_axis = [k for k,i in enumerate(liste[:125])]
x_label = [i[0] for i in liste[:125]]
plt.xticks(fontsize=15)
plt.yticks(fontsize=13)
plt.yticks(x_axis, x_label)
plt.xlabel("Number of occurance", fontsize = 18, labelpad = 10)
ax.barh(x_axis, y_axis, align='center')
ax = plt.gca()
ax.invert_yaxis()

plt.title("Word Occurance", bbox={'facecolor':'k', 'pad':5}, color='w', fontsize = 25)
#plt.show()

# Preserving important words :
list_products = []
for k, v in count_keywords.items():
    word = keywords_select[k]
    if word in ['pink', 'blue', 'tag', 'green', 'orange']: continue
    if len(word)<3 or v<13: continue
    list_products.append([word, v])

list_products.sort(key = lambda x:x[1], reverse=True)
print("Number of preserved words : ", len(list_products))


threshold = [0, 25, 75, 250, 750, 2500] #[0, 50, 500, 1000, 5000, 10000]

# Getting the description.
liste_produits = data['Desc'].unique()

# Creating the product and word matrix.
X = pd.DataFrame()
for key, occurence in list_products:
    X.loc[:, key] = list(map(lambda x:int(key.upper() in x), liste_produits))


label_col = []
for i in range(len(threshold)):
    if i == len(threshold) - 1:
        col = '.>{}'.format(threshold[i])
    else:
        col = '{}<.<{}'.format(threshold[i], threshold[i+1])

    label_col.append(col)
    X.loc[:, col] = 0

for i, prod in enumerate(liste_produits):
    prix = data[data['Desc'] == prod]['Amount'].mean()
    j = 0

    while prix > threshold[j]:
        j += 1
        if j == len(threshold):
            break
    X.loc[i, label_col[j-1]] = 1

print("{:<8} {:<20} \n".format('range', 'number of products') + 20*'-')
for i in range(len(threshold)):
    if i == len(threshold)-1:
        col = '.>{}'.format(threshold[i])
    else:
        col = '{}<.<{}'.format(threshold[i],threshold[i+1])
    print("{:<10}  {:<20}".format(col, X.loc[:, col].sum()))


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

matrix = X.as_matrix()


# Using optimal number of clusters using hyperparameter tuning:
# for n_clusters in range(3, 25):
#     kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init = 30)
#     kmeans.fit(matrix)
#     clusters = kmeans.predict(matrix)
#     sil_avg = silhouette_score(matrix, clusters)
#     print("For n_clusters : ", n_clusters, "The average silhouette_score is : ", sil_avg)

# Choosing number of clusters as 5:
# Trying Improving the silhouette_score :

n_clusters = 7
sil_avg = -1
while sil_avg < 0.6:
    kmeans = KMeans(init = 'k-means++', n_clusters = n_clusters, n_init = 30)
    kmeans.fit(matrix)
    clusters = kmeans.predict(matrix)
    sil_avg = silhouette_score(matrix, clusters)
    print("For n_clusters : ", n_clusters, "The average silhouette_score is : ", sil_avg)


# Printing number of elements in each cluster :
print (pd.Series(clusters).value_counts())

def graph_component_silhouette(n_clusters, lim_x, mat_size, sample_silhouette_values, clusters):
    import matplotlib as mpl
    mpl.rc('patch', edgecolor = 'dimgray', linewidth = 1)

    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(8, 8)
    ax1.set_xlim([lim_x[0], lim_x[1]])
    ax1.set_ylim([0, mat_size + (n_clusters + 1) * 10])
    y_lower = 10

    for i in range(n_clusters):
        ith_cluster_silhoutte_values = sample_silhouette_values[clusters == i]
        ith_cluster_silhoutte_values.sort()
        size_cluster_i = ith_cluster_silhoutte_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhoutte_values, alpha = 0.8)

        ax1.text(-0.03, y_lower + 0.5 * size_cluster_i, str(i), color = 'red', fontweight = 'bold',
                 bbox = dict(facecolor = 'white', edgecolor = 'black', boxstyle = 'round, pad = 0.3'))

        y_lower = y_upper + 10

# Plotting the intra cluster silhouette distances.
from sklearn.metrics import silhouette_samples
sample_silhouette_values = silhouette_samples(matrix, clusters)
graph_component_silhouette(n_clusters, [-0.07, 0.85], len(X), sample_silhouette_values, clusters)
#plt.show()

##################################################################################
# wordcloud
##################################################################################

liste = pd.DataFrame(liste_produits)
liste_words = [word for (word, occurance) in list_products]

occurance = [dict() for _ in range(n_clusters)]

# Creating data for printing word cloud.
for i in range(n_clusters):
    liste_cluster = liste.loc[clusters == i]
    for word in liste_words:
        if word in ['art', 'set', 'heart', 'pink', 'blue', 'tag']: continue
        occurance[i][word] = sum(liste_cluster.loc[:, 0].str.contains(word.upper()))

# Code for printing word cloud.
from random import randint
import random
def random_color_func(word=None, font_size=None, position=None,orientation=None, font_path=None, random_state=None):
    h = int(360.0 * tone / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)


def make_wordcloud(liste, increment):
    ax1 = fig.add_subplot(4, 2, increment)
    words = dict()
    trunc_occurances = liste[0:150]
    for s in trunc_occurances:
        words[s[0]] = s[1]

    wc = wordcloud.WordCloud(width=1000,height=400, background_color='lightgrey', max_words=1628,relative_scaling=1,
                             color_func = random_color_func, normalize_plurals=False)
    wc.generate_from_frequencies(words)
    ax1.imshow(wc, interpolation="bilinear")
    ax1.axis('off')
    plt.title('cluster n{}'.format(increment-1))


fig = plt.figure(1, figsize=(14,14))
color = [0, 160, 130, 95, 280, 40, 330, 110, 25]
for i in range(n_clusters):
    list_cluster_occurences = occurance[i]
    tone = color[i]
    liste = []
    for key, value in list_cluster_occurences.items():
        liste.append([key, value])
    liste.sort(key = lambda x:x[1], reverse = True)
    make_wordcloud(liste, i+1)
#plt.show()

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(matrix)
pca_samples = pca.transform(matrix)


# Checking the amount of variance explained :
fig, ax = plt.subplots(figsize=(14, 5))
sns.set(font_scale=1)
plt.step(range(matrix.shape[1]), pca.explained_variance_ratio_.cumsum(), where = 'mid', label = 'Cummulative Variance Explained')
sns.barplot(np.arange(1, matrix.shape[1] + 1), pca.explained_variance_ratio_, alpha = 0.5, color = 'g',
            label = 'Individual Variance Explained')
plt.xlim(0, 100)
plt.xticks(rotation = 45, fontsize = 14)
ax.set_xticklabels([s if int(s.get_text())%2 == 0 else '' for s in ax.get_xticklabels()])

plt.ylabel("Explained Variance", fontsize = 14)
plt.xlabel("Principal Components", fontsize = 14)
plt.legend(loc = 'upper left', fontsize = 13)
#plt.show()

corresp = dict()
for key, val in zip(liste_produits, clusters):
    corresp[key] = val

data['Category'] = data.loc[:, 'Desc'].map(corresp)
data[['OrderID', 'Desc', 'Category']][:10]

# Creating 5 new features that will contain the amount in a single transanction on different categories of product.
for i in range(5):
    col = 'categ_{}'.format(i)
    df_temp = data[data['Category'] == i]
    price_temp = df_temp['Amount'] #* (df_temp['Quantity'] - df_temp['QuantityCancelled'])
    price_temp = price_temp.apply(lambda x:x if x > 0 else 0)
    data.loc[:, col] = price_temp
    data[col].fillna(0, inplace = True)

data[['OrderID', 'Desc', 'Category', 'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4']][:10]


# sum of purchases by user and order.
temp = data.groupby(by=['Cust_Num', 'OrderID'], as_index = False)['Amount'].sum()
basket_price = temp.rename(columns={'Amount': 'Basket Price'})

# percentage spent on each product category
for i in range(5):
    col = "categ_{}".format(i)
    temp = data.groupby(by=['Cust_Num', 'OrderID'], as_index = False)[col].sum()
    basket_price.loc[:, col] = temp

# Dates of the order.
data['InvoiceDate_int'] = data['Sale_Date'].astype('int64')
temp = data.groupby(by=['Cust_Num', 'OrderID'], as_index = False)['InvoiceDate_int'].mean()
data.drop('InvoiceDate_int', axis = 1, inplace=True)
basket_price.loc[:, 'Sale_Date'] = pd.to_datetime(temp['InvoiceDate_int'])

# Selecting entries with basket price > 0.
basket_price = basket_price[basket_price['Basket Price'] > 0]
print (basket_price.sort_values('Cust_Num', ascending=True)[:10])

print (basket_price['Sale_Date'].min())
print (basket_price['Sale_Date'].max())

########################Datetime##########################
import datetime
set_entrainment = basket_price[basket_price['Sale_Date'] < datetime.date(2019, 8, 1)]
set_test = basket_price[basket_price['Sale_Date'] >= datetime.date(2019, 8, 1)]
basket_price = set_entrainment.copy(deep = True)


transanctions_per_user = basket_price.groupby(by=['Cust_Num'])['Basket Price'].agg(['count', 'min', 'max', 'mean', 'sum'])

for i in range(5):
    col = 'categ_{}'.format(i)
    transanctions_per_user.loc[:, col] = basket_price.groupby(by=['Cust_Num'])[col].sum() / transanctions_per_user['sum'] * 100

transanctions_per_user.reset_index(drop = False, inplace = True)
basket_price.groupby(by=['Cust_Num'])['categ_0'].sum()
print (transanctions_per_user.sort_values('Cust_Num', ascending = True)[:10])

# Generating two new variables - days since first puchase and days since last purchase.
last_date = basket_price['Sale_Date'].max().date()

first_registration = pd.DataFrame(basket_price.groupby(by=['Cust_Num'])['Sale_Date'].min())
last_purchase = pd.DataFrame(basket_price.groupby(by=['Cust_Num'])['Sale_Date'].max())

test = first_registration.applymap(lambda x:(last_date - x.date()).days)
test2 = last_purchase.applymap(lambda x:(last_date - x.date()).days)

transanctions_per_user.loc[:, 'LastPurchase'] = test2.reset_index(drop = False)['Sale_Date']
transanctions_per_user.loc[:, 'FirstPurchase'] = test.reset_index(drop = False)['Sale_Date']

print (transanctions_per_user[:10])

#### For finding & handling customers with only one purchase
# n1 = transanctions_per_user[transanctions_per_user['count'] == 1].shape[0]
# n2 = transanctions_per_user.shape[0]
# print("No. of Customers with single purchase : {:<2}/{:<5} ({:<2.2f}%)".format(n1, n2, n1/n2*100))


list_cols = ['count', 'min', 'max', 'mean', 'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4']
selected_customers = transanctions_per_user.copy(deep=True)
matrix = selected_customers[list_cols].as_matrix()


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(matrix)
print("Variable Mean Values: \n" + 90*'-' + '\n', scaler.mean_)
scaled_matrix = scaler.transform(matrix)

pca = PCA()
pca.fit(scaled_matrix)
pca_samples = pca.transform(scaled_matrix)

# Checking the amount of variance explained :
fig, ax = plt.subplots(figsize=(14, 5))
sns.set(font_scale=1)
plt.step(range(matrix.shape[1]), pca.explained_variance_ratio_.cumsum(), where = 'mid', label = 'Cummulative Variance Explained')
sns.barplot(np.arange(1, matrix.shape[1] + 1), pca.explained_variance_ratio_, alpha = 0.5, color = 'g',
            label = 'Individual Variance Explained')
plt.xlim(0, 10)
plt.xticks(rotation = 45, fontsize = 14)
ax.set_xticklabels([s if int(s.get_text())%2 == 0 else '' for s in ax.get_xticklabels()])

plt.ylabel("Explained Variance", fontsize = 14)
plt.xlabel("Principal Components", fontsize = 14)
plt.legend(loc = 'upper left', fontsize = 13)
#plt.show()

# Using optimal number of clusters using hyperparameter tuning:
# for n_clusters in range(3, 21):
#     kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init = 30)
#     kmeans.fit(scaled_matrix)
#     clusters = kmeans.predict(scaled_matrix)
#     sil_avg = silhouette_score(scaled_matrix, clusters)
#     print("For n_clusters : ", n_clusters, "The average silhouette_score is : ", sil_avg)


# Choosing number of clusters as 10:
# Trying Improving the silhouette_score :
n_clusters = 35
sil_avg = -1
while sil_avg < 0.35:
    kmeans = KMeans(init = 'k-means++', n_clusters = n_clusters, n_init = 30)
    kmeans.fit(scaled_matrix)
    clusters = kmeans.predict(scaled_matrix)
    sil_avg = silhouette_score(scaled_matrix, clusters)
    print("For n_clusters : ", n_clusters, "The average silhouette_score is : ", sil_avg)


n_clusters = 35
kmeans = KMeans(init = 'k-means++', n_clusters = n_clusters, n_init = 100)
kmeans.fit(scaled_matrix)
clusters_clients = kmeans.predict(scaled_matrix)
silhouette_avg = silhouette_score(scaled_matrix, clusters_clients)
print("Silhouette Score : {:<.30f}".format(silhouette_avg))

# Looking at clusters :
print (pd.DataFrame(pd.Series(clusters_clients).value_counts(), columns=['Number of Clients']).T)
#
sample_silhouette_values = silhouette_samples(scaled_matrix, clusters_clients)
#
graph_component_silhouette(n_clusters, [-0.15, 0.7], len(scaled_matrix), sample_silhouette_values, clusters_clients)
plt.show()
#
selected_customers.loc[:, 'cluster'] = clusters_clients
#
merged_df = pd.DataFrame()
for i in range(n_clusters):
    test = pd.DataFrame(selected_customers[selected_customers['cluster'] == i].mean())
    test = test.T.set_index('cluster', drop = True)
    test['size'] = selected_customers[selected_customers['cluster'] == i].shape[0]
    merged_df = pd.concat([merged_df, test])
#
#print (merged_df)
#merged_df.drop('Cust_Num', axis = 1, inplace = True)
print('Number of customers : ', merged_df['size'].sum())
#
merged_df = merged_df.sort_values('sum')
#
# Reorganizing the content of the dataframe.
liste_index = []
for i in range(5):
    column = 'categ_{}'.format(i)
    liste_index.append(merged_df[merged_df[column] > 20].index.values[0]) ###########

liste_index_reordered = liste_index
liste_index_reordered += [s for s in merged_df.index if s not in liste_index]

merged_df = merged_df.reindex(index = liste_index_reordered)
merged_df = merged_df.reset_index(drop = False)
print (merged_df.head())
#
selected_customers.to_csv("selected_customers.csv")
merged_df.to_csv("merged_df.csv")
plt.show()
