# импортируем необходимые библиотеки
import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt #Отрисовка графиков
import seaborn as sns
import numpy as np 
from PIL import Image
from datetime import datetime 

#задаем загаловок сайта
st.markdown('''<h1 style='text-align: center; color: black;'
            >Разведочный анализ данных</h1>''', 
            unsafe_allow_html=True)

#добавим фотографию пайплайн стримлита
image = Image.open('images/Pipeline.png')
st.image(image)

#краткое описание стримлита
st.write("""
Данный стримлит предназначен для наглядной демонтрации студентам простейших способов разведочного анализа данных (EDA - exploratory data analysis) для двух задач машинного обучения: 
классификация и регрессия.
""")
#-------------------------О проекте-------------------------
expander_bar = st.expander("Перед тем, как начать:")
expander_bar.markdown(
    """
\nЗадача **классификации** - предсказание категории объекта и разделение объектов согласно определенным и заданным заранее признакам. Таким образом можно сортировать данные по нужным категориям: 
одежду – по цветам или сезонам , книги – по жанрам или авторам, соусы – по степени остроты.
\nЗадача **регрессии** - предсказание целевой переменной по заданному набору признаков наблюдаемого объекта.
Таким образом можно прогнозировать цену недвижимости, капитализацию компании или стоимость акций. 
\n**Используемые библиотеки:** [streamlit](https://docs.streamlit.io/library/get-started), [pandas](https://pandas.pydata.org/docs/user_guide/index.html), [matplotlib](https://matplotlib.org/stable/api/index.html), 
[seaborn](https://seaborn.pydata.org).
\n **Полезно почитать:** [Про разведочный анализ данных](https://ru.wikipedia.org/wiki/Разведочный_анализ_данных), 
[Про классификацию](http://www.machinelearning.ru/wiki/index.php?title=Классификация), [Про регрессию](http://www.machinelearning.ru/wiki/index.php?title=Регрессия)
""")

#Даем студенту возможность выбрать самому задачу и данные (вместе с их описанием)
options = st.selectbox('Выберите направление задачи',
  ('Задача классификации', 'Задача регрессии'))

if options == 'Задача классификации':
  expander_bar = st.expander("Описание файлов:")
  expander_bar.markdown(
"""
\n**borrowers.csv**: исследование надежности заемщиков. Набор данных содержит личные сведения о каждом заемщике. Целевая переменная - была ли задолженность по возврату кредита (0 - задолженности не было; 
1 - задолженость была).
\n**wildfires.csv**: пожары в России. Набор данных содержит сведения МЧС России о географических точках, типах и датах природных пожаров, происходивших на территории России с 2012 по 2021 годы.
Целевая переменная - сколько часов было потрачено, чтобы потушить пожар полностью.
""")
  optionClass = st.selectbox(
  'Выберите фаил для классификации',
  ('borrowers.csv', 'wildfires.csv'))
  #чтобы правильно считывать датасеты
  if optionClass == 'wildfires.csv':
    input_Class = pd.read_csv(optionClass, parse_dates=['дата'])
  else:
    input_Class = pd.read_csv(optionClass)
  my_data = input_Class

if options == 'Задача регрессии':
  expander_bar = st.expander("Описание файлов:")
  expander_bar.markdown(
"""
\n**cars.csv**: исследование объявлений о продаже машин. Набор данных содержит информацию о марке машины, ее пробеге, объеме и типе двигателя и т.д.
Целевая переменная - стоимость машины
\n**SainP_houses.csv**: исследование объявлений с сервиса Яндекс.Недвижимость о продаже квартир в Санкт-Петербурге. Набор данных содержит информацию о самой квартире, ее расположении, наличии по близости торговых 
центров/аэропортов/прудов и т.д.
Целевая переменная - стоимость кваритры.
""")

  custom_date_parser = lambda x: datetime.strptime(x, "%Y")
  optionReg = st.selectbox(
  'Выберите фаил для регрессии',
  ('cars_price.csv', 'SaintP_houses.csv'))
  #чтобы правильно считывать датасеты
  if optionReg == 'SaintP_houses.csv':
    input_Reg = pd.read_csv(optionReg, parse_dates=['дата публикации'])  #, sep=','
  else:
    input_Reg = pd.read_csv(optionReg, parse_dates=['год'], date_parser=custom_date_parser)
  my_data = input_Reg

#---------------------Знакомимся с данными----------- 
st.subheader('Посмотрим на данные')

#покажем сам датасет, студент сможет сам котнролировать количество выводимых строк
if st.checkbox('Показать Датасет'):
  number = st.number_input('Сколько строк показать', min_value=1, max_value=my_data.shape[1])
  st.dataframe(my_data.head(number))

#выведем названия всех столбуов (фичей)
if st.checkbox('Название столбцов'):
  st.write(pd.DataFrame(my_data.columns, columns=['название столбцов']))

#покажем по отдельности кол-во строк и столбцов
if st.checkbox('Размер Датасета'):
  shape = st.radio(
    "Выбор данных",
     ('Строки', 'Столбцов'))
  if shape == 'Строки':
    st.write('Количество строк:', my_data.shape[0])
  elif shape == 'Столбцы':
    st.write('Количество столбцов:', my_data.shape[1])

#студент может выбрать один или несколько столбцов для просмотра
show_cols = st.checkbox('Выберите столбцы, на которые хотите посмотреть')
if show_cols:
  cols_to_show = st.multiselect('Столбцы', 
  my_data.columns.tolist())
  st.dataframe(my_data[cols_to_show])

#Посмотрим на уникальные значения 
n_unique = st.checkbox('Уникальные значения переменной')
if n_unique:
  n_cols = st.multiselect('Столбцы ', 
  my_data.columns.tolist())
  st.write(pd.DataFrame(my_data[n_cols].value_counts(), columns=['количество уникальных значений']))

#Познакомимся с типами данных
if st.checkbox('Типы данных'):
  st.write('**Тип данных** - внутреннее представление, которое язык программирования использует для понимания того, как данные хранить и как ими оперировать')
  expander_bar = st.expander('Информация об основных типах данных')
  expander_bar.info('''Object - текстовые или смешанные числовые и нечисловые значения 
  \nINT - целые числа 
  \nFLOAT - дробные числа 
  \nBOOL - значения True/False
  \nDATETIME - значения даты и времени
  ''')

  st.write(pd.DataFrame(my_data.dtypes.astype('str'), columns=['тип данных']))

#посмотрим на статистику 
if st.checkbox('Описательная статистика по всем числовым колонкам'):
  expander_bar = st.expander('Информация о данных, которые входят в описательную статистику')
  expander_bar.info('''Count - сколько всего было записей 
  \nMean - средняя велечина 
  \nStd - стандартное отклонение
  \nMin - минимальное значение
  \n25%/50%/70% - перцентили (показывают значение, ниже которого падает определенный процент наблюдений. Например, если число 5 - это 25% перцентиль, значит в наших данных 25% значений ниже 5)
  \nMax - максимальное значение
  ''')
  st.dataframe(my_data.describe())

#-----------------Строим графики---------------
st.set_option('deprecation.showPyplotGlobalUse', False) # чтобы убрать warning при построении графика
colors = ['indianred', 'steelblue', 'rosybrown', 'lightsteelblue','brown', 'darkgrey'] # Set project colors
st.subheader('Попробуем построить несколько базовых графиков')

#-----------------Гистограмма--------------------
vizHist = st.checkbox('Построить гистораму распределения HistPlot')
if vizHist:
  st.write('*HistPlot* - показывает распредление числовых значений объекта')
  option = st.selectbox('Выберите колонку', my_data.select_dtypes(exclude=['object']).columns)
  button_hist = st.button('Построить')
  if button_hist:
    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(20,10))
    plt.ticklabel_format(style='plain')
    ax = sns.histplot(data = my_data, x = my_data[option], kde = True)
    sns.set(style='darkgrid')
    st.pyplot(fig)

#------------------График корреляции--------------------
vizHeat = st.checkbox('Построить график корреляции Correlation Heatmap')
if vizHeat:
  st.write('*Correlation Heatmap* - графическое представление корреляционной матрицы, которая показывает зависимость между числовыми объектами')
  expander_bar = st.expander('Подробнее о корреляционной матрице')
  expander_bar.info(''' Корреляция - статистическая взаимосвязь двух или более переменных. Изменения значений одной переменной сопутствуют изменениям другой.
  \nВ нашем слуаем зависимость выражется на промежутке от -1 до 1. 
  \nЧем ближе значение к 1, тем сильнее прямая зависимость: увеличивая значения одной переменной, увеличивается значение и второй.
  \nЧем ближе значение к -1, тем сильнее обратная зависимость: увеличивая значение одной переменной, уменьшается значение второй и наоброт. 
  ''')
  fig, ax = plt.subplots(figsize=(20,10)) 
  ax = sns.heatmap(my_data.corr(),vmin=-1, vmax=1, annot=True, cmap='vlag',
                   center = 0, fmt='.1g', linewidths=1, linecolor='black')
  st.pyplot(fig)

#------------------Ящик с усами--------------------
vizBox = st.checkbox('Построить график формы распределения BoxPlot (Ящик с усами)')
if vizBox:
  st.write('*BoxPlot* - показывает медиану (линия внутри ящика), нижний (25%) и верхний квартили(75%), минимальное и максимальное значение выборки (усы) и ее выбросы')
  expander_bar = st.expander('Подробнее о квартиле')
  expander_bar.info(''' Квартили -  значения, которые делят данные на 4 группы (25%,50%,75%,100%), содержащие приблизительно равное количество наблюдений. 
  \nПо сути, это то же самое, что и перцентиль. То есть нижний квартиль - 25 перцентиль, а верхний квартиль - 75 перцентиль
  ''')
  image = Image.open('images/boxplot.png')
  st.image(image)
  fig, ax = plt.subplots() 
  fig = plt.figure(figsize=(20,10))
  plt.xticks(rotation=45)
  plt.ticklabel_format(style='plain')
  ax_x = st.selectbox('Ось Х', my_data.columns.tolist())
  ax_y = st.selectbox('Ось У', my_data.columns.tolist())
  button_box = st.button('Построить')
  if button_box:
    ax = sns.boxplot(x=my_data[ax_x], y=my_data[ax_y])
    st.pyplot(fig)

#------------------Диаграмма рассеяния---------------
vizScatter = st.checkbox('Построить диаграмму рассеяния ScatterPlot')
if vizScatter:
  st.write('**ScatterPlot** - помогает выявить взаимосвязи между переменными')
  fig, ax = plt.subplots() 
  fig = plt.figure(figsize=(30,15))
  plt.xticks(rotation=75)
  plt.ticklabel_format(style='plain')
  ax_x = st.selectbox('Ось Х', my_data.columns.tolist())
  ax_y = st.selectbox('Ось У', my_data.columns.tolist())
  button_scatter = st.button('Построить')
  if button_scatter:
    ax = sns.scatterplot(x=my_data[ax_x], y=my_data[ax_y])
    st.pyplot(fig)

#-------------------Задачи к лабораторной работе--------------------
st.subheader('Лабораторная работа по визуализации')

#для каждого датасета по 2-3 задачи 
options_2 = st.selectbox('Выберите датасет:',
  ('Borrowers.csv', 'Wildfires.csv', 'Cars.csv', 'SaintP.csv'))

if options_2 == 'Borrowers.csv':
  st.write("""
          1. Определите, какой средний возраст у заемщиков?
          2. Используя описательную статистику, посмотрите на максимальное значение уровня дохода. Может ли человек с таким доходом относится к должникам? Чтобы проверить гипотезу, постройте график?
          3. На что брали кредит 5 самых богатых по уровню дохода заемщика?
          """)
if options_2 == 'Wildfires.csv':
  st.write("""
           1. Сколько часов в среднем занимает тушение пожара? Посмотрите значение в описательной статистике, а затем визуализируете данные. 
           2. Какие типы пожаров устраняют дольше всего?
          """)
if options_2 == 'Cars.csv':
  st.write("""
          1. У какой модели автомобиля самая высокая средняя цена?
          2. Какой тип двигателя у самой старой машины?
          """)
if options_2 == 'SaintP.csv':
  st.write("""
          1. Какие 5 признаков сильнее всего влияют на стоимость квартир?
          2. Посмотрите описательную статистику по всем числовым колонкам. Сколько стоит самая дорогая квартира? Визуализируйте данные, чтобы узнать, сколько комнат в этой квартире.
          """)