#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


# In[4]:


data = pd.read_csv('movie_bd_v5.csv')
data.sample(5)
display(data)


# In[5]:


data.describe()


# In[ ]:


#Все ответы верные 27/27


# # Предобработка

# In[ ]:


answers = {} # создадим словарь для ответов
#рассчитаем прибыль фильмов и добавим в качестве отдельного столбца в датафрейм
data['profit'] = data['revenue'] - data['budget'] 
#функция, собирающая словарь из требуемых элементов (режиссеры и их суммарная касса) 
#и возвращающая результат конвертация словаря в датафрейм.
def rev_dir(data):
    rev_sum = Counter()
    for i in range(len(data)):
        for j in data.iloc[i].director.split('|'):
            rev_sum[j] += data.iloc[i].revenue
    return pd.DataFrame.from_dict(rev_sum, orient = 'index', columns = ['Sum'])


# # 1. У какого фильма из списка самый большой бюджет?

# Использовать варианты ответов в коде решения запрещено.    
# Вы думаете и в жизни у вас будут варианты ответов?)

# In[ ]:


# в словарь вставляем номер вопроса и ваш ответ на него
# Пример: 
answers['1'] = '2. Spider-Man 3 (tt0413300)'
# запишите свой вариант ответа
answers['1'] = '5. Pirates of the Caribbean: On Stranger Tides (tt1298650)'
# если ответили верно, можете добавить комментарий со значком "+"


# In[ ]:


# тут пишем ваш код для решения данного вопроса:
data2 = pd.read_csv('movie_bd_v5.csv')
data2.groupby(['original_title'])['runtime'].max().sort_values(ascending=False)


# ВАРИАНТ 2

# In[ ]:


data.sort_values(by=['budget'], ascending=False).head(1)[['original_title','imdb_id']]


# # 2. Какой из фильмов самый длительный (в минутах)?

# In[ ]:


answers['2'] = '2. Gods and Generals (tt0279111)'


# In[ ]:


data.sort_values(by=['runtime'], ascending=False).head(1)[['original_title','imdb_id']]


# ВАРИАНТ 2

# In[ ]:


data[data['runtime'] == data['runtime'].max()].original_title


# # 3. Какой из фильмов самый короткий (в минутах)?
# 
# 
# 
# 

# In[ ]:


answers['3'] = '3. Winnie the Pooh (tt1449283)'


# In[ ]:


data['original_title'].loc[data['runtime'].idxmin()]


# ВАРИАНТ 2

# In[ ]:


data[data.runtime == data.runtime.min()].original_title


# # 4. Какова средняя длительность фильмов?
# 

# In[ ]:


answers['4'] = '2. 110'


# In[ ]:


data.describe().loc['mean']['runtime']


# ВАРИАНТ 2

# In[ ]:


round(data.runtime.mean())


# # 5. Каково медианное значение длительности фильмов? 

# In[ ]:


answers['5'] = '1. 107'


# In[ ]:


data.describe().loc['50%']['runtime']


# ВАРИАНТ 2

# In[ ]:


round(data.runtime.median())


# ВАРИАНТ 3

# In[ ]:


np.median(data.runtime)


# # 6. Какой самый прибыльный фильм?
# #### Внимание! Здесь и далее под «прибылью» или «убытками» понимается разность между сборами и бюджетом фильма. (прибыль = сборы - бюджет) в нашем датасете это будет (profit = revenue - budget) 

# In[ ]:


# лучше код получения столбца profit вынести в Предобработку что в начале


# In[ ]:


answers['6'] = '5. Avatar (tt0499549)'


# In[ ]:


data.groupby(['original_title'])['profit'].max().sort_values(ascending=False)


# ВАРИАНТ 2

# In[ ]:


data[data.profit == data.profit.max()].original_title


# ВАРИАНТ 3

# In[ ]:


data.sort_values(['profit'], ascending=False).head(1)


# # 7. Какой фильм самый убыточный? 

# In[ ]:


answers['7'] = '5. The Lone Ranger (tt1210819)'


# In[ ]:


data['profit'] = data['revenue'] - data['budget']
data.groupby(['original_title'])['profit'].min().sort_values()


# ВАРИАНТ 2

# In[ ]:


data.sort_values(by=['profit']).head(1)[['original_title','imdb_id']]


# # 8. У скольких фильмов из датасета объем сборов оказался выше бюджета?

# In[ ]:


answers['8'] = '1. 1478'


# In[ ]:


data.loc[data.profit > 0].imdb_id.count()


# ВАРИАНТ 2

# In[ ]:


data[data['revenue'] > data['budget']].original_title.count()


# ВАРИАНТ 3

# In[ ]:


data['imdb_id'][data['profit'] > 0].nunique()#Мы будем вычислять по imdb_id, ведь присутствуют фильмы с одинаковыми названиями


# ВАРИАНТ 4

# In[ ]:


len(data[data['revenue']>data['budget']])


# # 9. Какой фильм оказался самым кассовым в 2008 году?

# In[ ]:


answers['9'] = '4. The Dark Knight (tt0468569)'


# In[ ]:


data.query('release_year==2008').query('revenue==revenue.max()')


# ВАРИАНТ 2

# In[ ]:


data.loc[data.release_year == 2008].sort_values(by=['revenue'], ascending=False).head(1)[['original_title','imdb_id']]


# # 10. Самый убыточный фильм за период с 2012 по 2014 г. (включительно)?
# 

# In[ ]:


answers['10'] = '1245. The Lone Ranger (tt1210819)'


# In[ ]:


df = data[(2012 <= data.release_year) & (data.release_year <= 2014)]
display(df[df.profit == df.profit.min()].original_title)


# ВАРИАНТ 2

# In[ ]:


data.loc[(data.release_year >= 2012) & (data.release_year <=2014)].sort_values(by = 'profit').head(1).original_title


# # 11. Какого жанра фильмов больше всего?

# In[ ]:


answers['11'] = '3. Drama'


# In[ ]:


gen_count = []
for i in data['genres']:
    gen_count.append(i.split(sep='|'))
c = Counter(gen for gen_list in gen_count for gen in gen_list)
print(c)


# ВАРИАНТ 2

# In[ ]:


Counter(data.genres.str.cat(sep='|').split('|')).most_common(1)


# ВАРИАНТ 3

# In[314]:


data.genres.str.split('|', expand=True).stack().value_counts().index[0]


# ВАРИАНТ 4

# In[ ]:


pd.Series(data['genres'].str.cat(sep='|').split('|')).value_counts().head(1)


# # 12. Фильмы какого жанра чаще всего становятся прибыльными? 

# In[ ]:


answers['12'] = '1. Drama'


# In[ ]:


films_profit = data[data['profit'] > 0]
pd.Series(films_profit['genres'].str.cat(sep='|').split('|')).value_counts().head(1)


# ВАРИАНТ 2

# In[ ]:


Counter(data[data.profit>0].genres.str.cat(sep='|').split('|')).most_common(1)


# ВАРИАНТ 3

# In[ ]:


data[data.profit>0].genres.str.split('|', expand=True).stack().value_counts().index[0]


# # 13. У какого режиссера самые большие суммарные кассовые сбооры?

# In[ ]:


answers['13'] = '5. Peter Jackson'


# In[ ]:


test = data.groupby(['director']).sum()
test.revenue.sort_values(ascending = False).index[0]


# ВАРИАНТ 2

# In[ ]:


data.groupby(['director'])['revenue'].sum().sort_values(ascending=False).head()


# ВАРИАНТ 3

# In[ ]:


df = data.groupby(['director'])['revenue'].sum()
display(df[df == df.max()])


# ВАРИАНТ 4

# In[ ]:


display(rev_dir(data[data.revenue > 0]).sort_values(by = ['Sum'], ascending = False).head(1))


# # 14. Какой режисер снял больше всего фильмов в стиле Action?

# In[ ]:


answers['14'] = '3. Robert Rodriguez'


# In[ ]:


data_act = data[data.genres.str.contains('Action')]
data_act.director.str.split('|', expand=True).stack().value_counts()


# ВАРИАНТ 2

# In[ ]:


data[data['genres'].str.contains('action', case=False)]['director'].str.split('|').explode().value_counts().index[0]


# # 15. Фильмы с каким актером принесли самые высокие кассовые сборы в 2012 году? 

# In[ ]:


answers['15'] = '3. Chris Hemsworth'


# In[ ]:


data15 = pd.read_csv('movie_bd_v5.csv')
data15['cast'] = data15['cast'].str.split('|')
data15_cast = data15.explode('cast')
data15_cast[data15_cast['release_year'] == 2012].groupby(['cast'])['revenue'].sum().sort_values(ascending=False).head()


# ВАРИАНТ 2

# In[ ]:


films_2012 = data[data.release_year == 2012][['cast', 'revenue']]
films_2012['cast'] = films_2012.cast.apply(lambda x: x.split('|'))
films_2012.explode('cast').groupby(['cast'])['revenue'].sum().sort_values(ascending = False).head(1)


# # 16. Какой актер снялся в большем количестве высокобюджетных фильмов?

# In[ ]:


answers['16'] = '3. Matt Damon'


# In[ ]:


data16 = data15_cast.copy() #Берем с прошлого вопроса уже разбитый по актёрам фрейм
data16[data16['budget'] > data16['budget'].mean()].groupby(['cast'])['cast'].count().sort_values(ascending=False).head()


# ВАРИАНТ 2

# In[ ]:


df = data[data.budget > data.budget.mean()]
df.cast.str.split('|', expand=True).stack().value_counts()


# ВАРИАНТ 3

# In[ ]:


data2 = data[data['budget'] > data['budget'].mean()]
actors = pd.Series(data2.cast.str.cat(sep='|').split('|')).value_counts()
actors


# # 17. В фильмах какого жанра больше всего снимался Nicolas Cage? 

# In[ ]:


answers['17'] = '2. Action'


# In[ ]:


data_cage = data[data.cast.str.contains('Nicolas Cage')]
data_cage.genres.str.split('|', expand=True).stack().value_counts()


# ВАРИАНТ 2

# In[ ]:


data_cast = data.copy()
data_cast['genres'] = data_cast.genres.apply(lambda x: x.split('|'))
data_nic = data_cast.explode('genres')
data_nic[data_nic.cast.str.contains('Nicolas Cage', na = False)].genres.value_counts().head(1)


# # 18. Самый убыточный фильм от Paramount Pictures

# In[ ]:


answers['18'] = 'K-19: The Widowmaker (tt0267626)'


# In[ ]:


data_Paramaunt = data[data['production_companies'].str.contains('Paramount Pictures')]
data_Paramaunt[data_Paramaunt['profit'] == data_Paramaunt['profit'].min()].original_title


# ВАРИАНТ 2

# In[ ]:


data_Paramaunt = data[data.production_companies.str.contains("Paramount Pictures", na='')].copy()
data_Paramaunt.sort_values(by=['profit']).head(1)[['original_title','imdb_id']]


# # 19. Какой год стал самым успешным по суммарным кассовым сборам?

# In[ ]:


answers['19'] = '5. 2015'


# In[ ]:


data.groupby('release_year').revenue.agg('sum').sort_values(ascending = False).index[0]


# ВАРИАНТ 2

# In[ ]:


df = data.groupby(['release_year'])['revenue'].sum()
display(df[df == df.max()])


# ВАРИАНТ 3

# In[ ]:


data.groupby(['release_year'])['revenue'].sum().sort_values(ascending = False).head(1)


# # 20. Какой самый прибыльный год для студии Warner Bros?

# In[ ]:


answers['20'] = '2014'


# In[390]:


movies_Warner_Bros = data[data['production_companies'].str.contains('Warner Bros', case=False)]
movies_Warner_Bros.groupby('release_year', sort=False)['profit'].sum().sort_values(ascending=False).index[0]


# ВАРИАНТ 2

# In[ ]:


data_new = data[data.production_companies.str.contains('Warner Bros')]
pd.DataFrame(data_new.groupby(by='release_year').profit.sum()).sort_values('profit', ascending=False).head(1)


# # 21. В каком месяце за все годы суммарно вышло больше всего фильмов?

# In[ ]:


answers['21'] = 'Сентябрь'


# In[ ]:


res = data
res['month'] = pd.DatetimeIndex(res['release_date']).month
res[['imdb_id', 'month']].groupby(['month']).count().sort_values('imdb_id',  ascending=False).head()


# ВАРИАНТ 2

# In[ ]:


list_months = []
for months in data.release_date:
    list_months.append(months[:months.find('/')])
data_months = pd.Series(list_months)
data_months_sum = data_months.value_counts()
data_months_sum


# ВАРИАНТ 3

# In[ ]:


data['month'].value_counts()


# # 22. Сколько суммарно вышло фильмов летом? (за июнь, июль, август)

# In[ ]:


answers['22'] = '450'


# In[ ]:


data_months_sum[['6', '7', '8']].sum()


# ВАРИАНТ 2

# In[ ]:


len(data.loc[data['month'].isin([6,7,8])])


# ВАРИАНТ 3

# In[ ]:


res = data
res['month'] = pd.DatetimeIndex(res['release_date']).month
res[(res['month'] >= 6) & (res['month'] <= 8)][['imdb_id', 'month']].groupby(['month']).count().sum()


# # 23. Для какого режиссера зима – самое продуктивное время года? 

# In[ ]:


answers['23'] = '5. Peter Jackson'


# In[ ]:


data_winter = pd.read_csv('movie_bd_v5.csv')
data_winter['director'] = data_winter['director'].str.split('|')
data_winter = data_winter.explode('director')
data_winter['release_date'] = data_winter['release_date'].str.split('/')
month = []
for i in data_winter['release_date']:
    month.append(i[0])
data_winter['month'] = month
data_winter[data_winter['month'].isin(['1', '2', '12'])].groupby(['director'])['month'].count().sort_values(ascending=False)


# ВАРИАНТ 2

# In[ ]:


data_winter = data.loc[data['month'].isin([1, 2, 12])]
directors = pd.Series(data_winter.director.str.cat(sep='|').split('|')).value_counts()
directors.head()


# # 24. Какая студия дает самые длинные названия своим фильмам по количеству символов?

# In[ ]:


answers['24'] = 'Four By Two Productions'


# In[ ]:


data['length_title'] = data.original_title.apply(lambda x: len(x))
data.length_title.head()
companies = pd.Series(data['production_companies'].str.cat(sep=('|')).split('|')).value_counts()
companies.head()
for company in companies.index:
    companies[company] = data['length_title'][data['production_companies'].map(lambda x: True if company in x else False)].mean()
companies.sort_values(ascending = False)


# ВАРИАНТ 2

# In[ ]:


data_companies = data.copy()
data_companies['production_companies'] = data.production_companies.apply(lambda x: x.split('|'))
data_companies2 = data_companies.explode('production_companies')
data_companies2['len_films'] = data_companies2.original_title.apply(lambda x: len(x))
data_companies2.groupby(by = 'production_companies').len_films.mean().sort_values(ascending = False).head(1)


# # 25. Описание фильмов какой студии в среднем самые длинные по количеству слов?

# In[ ]:


answers['25'] = 'Midnight Picture Show'


# In[ ]:


data['overview_len'] = data.overview.apply(lambda x: len(x.split()))
data_overview = data.copy()
data_overview['production_companies'] = data.production_companies.apply(lambda x: x.split('|'))
summ_len_overview = data_overview.explode('production_companies').groupby(['production_companies'])['overview_len'].sum()
count_companies_overview = data_overview.explode('production_companies').groupby(['production_companies'])['imdb_id'].count()
c = summ_len_overview/count_companies_overview
c.sort_values(ascending = False).head(1)


# ВАРИАНТ 2

# In[ ]:


data['length_overview'] = data.overview.apply(lambda x: len(x.split()))
data['length_overview'].head()
companies = pd.Series(data['production_companies'].str.cat(sep=('|')).split('|')).value_counts()
for company in companies.index:
    companies[company] = data['length_overview'][data['production_companies'].map(lambda x: True if company in x else False)].mean()
companies.sort_values(ascending = False).head()


# # 26. Какие фильмы входят в 1 процент лучших по рейтингу? 
# по vote_average

# In[ ]:


answers['26'] = 'Inside Out, The Dark Knight, 12 Years a Slave'


# In[ ]:


data[data['vote_average'] > data['vote_average'].quantile(0.99)]['original_title'].str.cat(sep='\n')


# ВАРИАНТ 2

# In[ ]:


data[['vote_average', 'original_title']].sort_values(by='vote_average').tail(19)


# ВАРИАНТ 3

# In[ ]:


res = data[['original_title', 'vote_average']]
quant = res.sort_values('vote_average',  ascending=False).quantile(0.99)[0]
res = res[res['vote_average'] >= quant]
res.sort_values('vote_average',  ascending=False).head(7)


# ВАРИАНТ 4

# In[ ]:


df = data[['original_title', 'vote_average']].sort_values(['vote_average'], ascending=False)
df.index = list(df.original_title)
df = df['vote_average']
display(df.head(len(df)//100))


# # 27. Какие актеры чаще всего снимаются в одном фильме вместе?
# 

# In[ ]:


answers['27'] = 'Daniel Radcliffe & Rupert Grint'


# In[ ]:


from itertools import combinations
data_combo_list = pd.read_csv('movie_bd_v5.csv')
data_combo_list['cast'] = data_combo_list['cast'].str.split('|')
data_combo_list['combinations'] = data_combo_list['cast'].apply(lambda r: list(combinations(r, 2)))
data_combo_list = data_combo_list.explode('combinations')
data_combo_list.groupby(['combinations'])['combinations'].count().sort_values(ascending=False)


# ВАРИАНТ 2

# In[ ]:


actor_list=data.cast.str.split('|').tolist()
combo_list=[]
for i in actor_list:
    for j in combinations(i, 2):
        combo_list.append(' '.join(j))
# Произведем расчет, приведя все к DataFrame
combo_list=pd.DataFrame(combo_list)
combo_list.columns=['actor_combinations']
combo_list.actor_combinations.value_counts().head(2)


# # Submission

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




