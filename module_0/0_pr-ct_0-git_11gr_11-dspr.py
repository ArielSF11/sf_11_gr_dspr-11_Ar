#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np


# In[17]:


count = 0                            # счетчик попыток
number = np.random.randint(1,101)    # загадали число
print ("Загадано число от 1 до 100")

while True:                        # бесконечный цикл
    predict = int(input())         # предполагаемое число
    count += 1                     # плюсуем попытку
    if number == predict: break    # выход из цикла, если угадали
    elif number > predict: print (f"Угадываемое число больше {predict} ")
    elif number < predict: print (f"Угадываемое число меньше {predict} ")
        
print (f"Вы угадали число {number} за {count} попыток.")


# In[18]:


number = np.random.randint(1,100)    # загадали число
print ("Загадано число от 1 до 99")
for count in range(1,101):         # более компактный вариант счетчика
    if number == count: break    # выход из цикла, если угадали      
print (f"Вы угадали число {number} за {count} попыток.")


# In[19]:


def game_core_v1(number):
    '''Просто угадываем на random, никак не используя информацию о больше или меньше.
       Функция принимает загаданное число и возвращает число попыток'''
    count = 0
    while True:
        count+=1
        predict = np.random.randint(1,101) # предполагаемое число
        if number == predict: 
            return(count) # выход из цикла, если угадали
        
        
def score_game(game_core):
    '''Запускаем игру 1000 раз, чтобы узнать, как быстро игра угадывает число'''
    count_ls = []
    np.random.seed(1)  # фиксируем RANDOM SEED, чтобы ваш эксперимент был воспроизводим!
    random_array = np.random.randint(1,100, size=(1000))
    for number in random_array:
        count_ls.append(game_core(number))
    score = int(np.mean(count_ls))
    print(f"Ваш алгоритм угадывает число в среднем за {score} попыток")
    return(score)


# In[20]:


# запускаем
score_game(game_core_v1)


# In[21]:


def game_core_v2(number):
    '''Сначала устанавливаем любое random число, а потом уменьшаем или увеличиваем его в зависимости от того, больше оно или меньше нужного.
       Функция принимает загаданное число и возвращает число попыток'''
    count = 1
    predict = np.random.randint(1,101)
    while number != predict:
        count+=1
        if number > predict: 
            predict += 1
        elif number < predict: 
            predict -= 1
    return(count) # выход из цикла, если угадали


# In[22]:


# Проверяем
score_game(game_core_v2)


# In[23]:


def game_core_v3(number): #применим бинарный или, как еще называют двоичный поиск 
    count = 0 # счетчик попыток
    predict = 50 #берем в качестве первого предпологаемого числа - среднее значение из интервала от 0 до 100
    upper_limit = 100 #верхняя граница
    lower_limit = 0 #нижняя граница
    while number != predict:
        count+=1
        if number > predict:
            lower_limit = predict #так как загаданное число больше, то предположение становится новой нижней границей поиска
            predict = (lower_limit+upper_limit) // 2  #заново берём среднее значение
        elif number < predict:
            upper_limit = predict #так как загаданное число меньше, то предположение становится новой верхней границей поиска
            predict = (lower_limit+upper_limit) // 2 #заново берём среднее значение
    return(count)


# In[24]:


# Проверяем
score_game(game_core_v3)


# In[25]:


def game_core_v4(number):
    """Бинарный поиск с сокращением строк кода.
       Функция принимает загаданное число и возвращает число попыток"""
    count, possible_range = 0, [0,100]
    get_new_predict = lambda range_list: int(sum(range_list)/2)
    more_flag = lambda x: 1 if number > x else 0
    predict = get_new_predict(possible_range)
    while number != predict:
        possible_range[0] = possible_range[0]*(1-more_flag(predict)) + more_flag(predict)*predict
        possible_range[1] = possible_range[1]*more_flag(predict) + (1-more_flag(predict))*predict
        predict = get_new_predict(possible_range)
        count+=1
    return count


# In[26]:


# Проверяем
score_game(game_core_v4)


# In[ ]:





# In[ ]:





# In[27]:


import random
from random import randrange
number = randrange(101)+1                     
print ("Загадано число от 1 до 100")
predict = -1
count = 0                                    # счетчик попыток
while predict != number:
    predict = int(input('Введите число  '))  # предполагаемое число
    count +=1                                # плюсуем попытку
    if number == predict: break              # выход из цикла, если угадали
    elif predict > number:
        print(f"Угадываемое число меньше {predict} ")
    elif predict < number:
        print(f"Угадываемое число больше {predict} ")
print(f"Вы угадали число {number} за {count} попыток.")


# In[ ]:




