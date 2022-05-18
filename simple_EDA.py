import streamlit as st
import pandas as pd #Пандас
# import matplotlib
import matplotlib.pyplot as plt #Отрисовка графиков
import seaborn as sns
import numpy as np #Numpy
#import pickle
from PIL import Image
#from tqdm import tqdm
import time

st.markdown('''<h1 style='text-align: center; color: black;'
            >Разведочный анализ данных</h1>''', 
            unsafe_allow_html=True)
# img = Image.open('2_RANEPA.png') #1_RANEPA.jpg or 2_RANEPA.png
# st.image(img, use_column_width='auto') #width=400

st.write("""
Данный сримлит предназначен для наглядной демонтрации студентам простейших способов разведочного анализа данных (EDA - exploratory data analysis) для двух задач машинного обучения: классификация и регрессия.

\nДанные подготовил Абдулвасиев Муминшо.
""")
#-------------------------О проекте-------------------------
expander_bar = st.expander("Перед тем, как начать:")
expander_bar.markdown(
    """
\nЗадача **классификации** - предсказание категории объекта и разделение объектов согласно определенным и заданным заранее признакам. Таким образом можно сортировать данные по нужным категориям: 
одежду – по цветам или сезонам , книги – по жанрам или авторам, соусы – по степени остроты.
\nЗадача **регресси** - предсказание целевой переменной по заданному набору признаков наблюдаемого объекта.
Таким образом можно прогнозировать цену недвижимости, капитализацию компании или стоимость акций. 
\n**Используемые библиотеки:** [streamlit](https://docs.streamlit.io/library/get-started), [pandas](https://pandas.pydata.org/docs/user_guide/index.html), [matplotlib](https://matplotlib.org/stable/api/index.html), [seaborn](https://seaborn.pydata.org).
\n **Полезно почитать:** [Про разведочный анализ данных](https://ru.wikipedia.org/wiki/Разведочный_анализ_данных), 
[Про классификацию](http://www.machinelearning.ru/wiki/index.php?title=Классификация), [Про регрессию](http://www.machinelearning.ru/wiki/index.php?title=Регрессия)
""")

options = st.selectbox('Выберите направление задачи',
  ('Задача классификации', 'Задача регрессии'))

if options == 'Задача классификации':
  expander_bar = st.expander("Описание файлов:")
  expander_bar.markdown(
"""
\n**wildfires.csv**: пожары в России. Набор данных содержит сведения МЧС России о географических точках, типах и датах природных пожаров, происходивших на территории России с 2012 по 2021 годы.
Целевая переменная - сколько часов было потрачено, чтобы потушить пожар полностью.
\n**borrowers.csv**: исследование надежности заемщиков. Набор данных содержит личные сведения о каждом заемщике. Целевая переменная - была ли задолжность по возврату кредита.
""")
  optionClass = st.selectbox(
  'Выберите фаил для классификации',
  ('wildfires.csv', 'borrowers.csv'))
  if optionClass == 'wildfires.csv':
    input_Class = pd.read_csv(optionClass, parse_dates=['дата']) #delimiter=','
  else:
    input_Class = pd.read_csv(optionClass)
  my_data = input_Class

if options == 'Задача регрессии':
  expander_bar = st.expander("Описание файлов:")
  expander_bar.markdown(
"""
\n**cars.csv**: исследование объявлений о продаже машин. Набор данных содержит информацию о марке машины, ее пробеге, объеме и типе двигателя и т.д.
Целевая переменная - стоимость машины
\n**SainP_houses.csv**: исследование объявлений с сервиса Яндекс.Недвижимость о продаже квартир в Санкт-Петербурге. Набор данных содержит информацию о самой квартире, ее расположении, наличии по близости торговых центров/аэропортов/прудов и т.д.
Целевая переменная - стоимость кваритры.
""")
# \n**mos_houses.csv**: исследование объявлений о продаже квартир в Москве. Набор данных содержит информацию о самой квартире, ее расположении, наличии метро и т.д.
#Целевая переменная - стоимость кваритры.

  optionReg = st.selectbox(
  'Выберите фаил для регрессии',
  ('cars_price.csv', 'SaintP_houses.csv'))
  if optionReg == 'SaintP_houses.csv':
    input_Reg = pd.read_csv(optionReg, parse_dates=['дата публикации'])  #, sep=','
  else:
    input_Reg = pd.read_csv(optionReg)
  my_data = input_Reg

# st.write('You selected:', option)

# if optionClass == 'wildfires.csv':
#   input_df = pd.read_csv(optionClass, delimiter=',', parse_dates=['dt'])
# else:
#   input_df = pd.read_csv(optionClass)

# if optionReg:
#   input_df = pd.read_csv(optionReg)

 
st.subheader('Посмотрим на данные')

if st.checkbox('Показать Датасет'):
  number = st.number_input('Сколько строк показать', min_value=1, max_value=my_data.shape[1])
  st.dataframe(my_data.head(number))

if st.checkbox('Название колонок'):
  st.write(pd.DataFrame(my_data.columns, columns=['название колонок']))

if st.checkbox('Размер Датасета'):
  shape = st.radio(
    "Выбор данных",
     ('Строки', 'Колонки'))
  if shape == 'Строки':
    st.write('Количество строк:', my_data.shape[0])
  elif shape == 'Колонки':
    st.write('Количество колонок:', my_data.shape[1])

if st.checkbox('Выберите колонки, на которые хотите посмотреть'):
  cols = st.multiselect('Колонки', 
  my_data.columns.tolist())
  st.dataframe(my_data[cols])

if st.checkbox('Уникальные значения целевой переменной'):
  st.write('*Целевая переменная* -  признак датасета, который предстоит предсказывать модели машинного обучения')
  st.write(pd.DataFrame(my_data.iloc[:, -1:].value_counts(), columns=['количество уникальных значений']))

if st.checkbox('Типы данных'):
  expander_bar = st.expander('Информация об основных типах данных')
  expander_bar.info('''Object - текстовые или смешанные числовые и нечисловые значения 
  \nINT - целые числа 
  \nFLOAT - дробные числа 
  \nBOOL - значения True/False
  \nDATETIME - значения даты и времени
  ''')
  st.write(pd.DataFrame(my_data.dtypes.astype('str'), columns=['тип данных']))

if st.checkbox('Описательная статистика по всем числовым колонкам'):
  expander_bar = st.expander('Информация о данных, которые входят в описательную статистику')
  expander_bar.info('''Count - сколько всего было записей 
  \nMean - средняя велечина 
  \nStd - стандартное отклонение
  \nMin - минимальное значение
  \n25%/50%/70% - перцентили (показывает значение, ниже которого падает определенный процент наблюдений)
  \nMax - максимальное значение
  ''')
  st.dataframe(my_data.describe())

#-----------------Visualization---------------
st.set_option('deprecation.showPyplotGlobalUse', False) # чтобы убрать warning при построении графика
st.subheader('Попробуем построить несколько базовых графиков')

#------------------PiePlot--------------------
vizPie  = st.checkbox('Построить график распределений по целевой переменной PiePlot') # придумать название
if vizPie:
  if options == 'Задача регрессии':
    labels = my_data.iloc[:, -1:].loc[:20].squeeze(axis=1).unique()
    sizes = my_data.iloc[:, -1:].loc[:20].squeeze(axis=1).value_counts()
  else:
    labels = my_data.iloc[:, -1:].squeeze(axis=1).unique()
    sizes = my_data.iloc[:, -1:].squeeze(axis=1).value_counts()
  fig1, ax1 = plt.subplots()
  ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
  ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
  plt.legend()
  show = plt.show()
  st.write('*PiePlot* - статистическая диаграмма, разделенная на срезы, которые иллюстрируют числовую пропорцию')
  st.pyplot(show)

#-----------------HistPlot--------------------
vizHist = st.checkbox('Построить гистораму распределения HistPlot')
if vizHist:
  st.write('*HistPlot* - показывает распредление числовых значений объекта')
  option = st.multiselect('Выберите одну переменную', my_data.select_dtypes(exclude=['object']).columns)
  fig, ax = plt.subplots()
  fig = plt.figure(figsize=(20,10))
  plt.ticklabel_format(style='plain')
  ax = sns.histplot(data = my_data, x = my_data[option[0]], kde = True)
  sns.set(style='darkgrid')
  st.pyplot(fig)

#------------------HeatMap--------------------
vizHeat = st.checkbox('Построить график корреляции Correlation Heatmap')
if vizHeat:
  st.write('*Correlation Heatmap* - графическое представление корреляционной матрицы, которая показывает зависимость между числовыми объектами')
  fig, ax = plt.subplots(figsize=(20,10)) 
  ax = sns.heatmap(my_data.corr(),vmin=-1, vmax=1, annot=True, cmap='RdBu',
                   center = 0, fmt='.1g', linewidths=1, linecolor='black')
  st.pyplot(fig)

# with sns.axes_style("white"):
#         f, ax = plt.subplots(figsize=(14, 10))
#         ax = sns.heatmap(corr, annot = True, vmin=-1, vmax=1, center= 0, 
#         cmap= 'coolwarm', fmt='.1g', linewidths=1, linecolor='black') # mask=mask, vmax=1, square=True
#     st.pyplot(f)

#------------------BoxPlot--------------------
vizBox = st.checkbox('Построить график формы распределения BoxPlot (Ящик с усами)') #Boxplot (Ящик с усами) — это график, отражающий форму распределение, медиану, квартили и выбросы.
if vizBox:
  st.write('*BoxPlot* - показывает медиану/среднее, нижний и верхний квартили, минимальное и максимальное значение выборки и ее выбросы')
  fig, ax = plt.subplots() 
  fig = plt.figure(figsize=(20,10))
  plt.xticks(rotation=45)
  plt.ticklabel_format(style='plain')
  if options == 'Задача классификации':
    ax_x = st.multiselect('Ось Х', my_data.iloc[:,-1:].columns.tolist())
    ax_y = st.multiselect('Ось У (выберите одну переменную)', my_data.iloc[:,:-1].columns.tolist())
    ax = sns.boxplot(x=my_data[ax_x[0]], y=my_data[ax_y[0]])
  else:
    ax_x = st.multiselect('Ось Х (выберите одну переменную)', my_data.iloc[:,:-1].columns.tolist())
    ax_y = st.multiselect('Ось У', my_data.iloc[:,-1:].columns.tolist()) 
    ax = sns.boxplot(x=my_data[ax_x[0]].iloc[:50], y=my_data[ax_y[0]].iloc[:50])
  st.pyplot(fig)
#------------------CatPlot--------------------
# vizCount = st.checkbox('Построить категориальный график CatPlot')
# if vizCount:
#   tar = st.multiselect('Целевая переменная', my_data.iloc[:,-1:].columns)
#   ax_x = st.multiselect('Выберите колонку (ось Х)',my_data.iloc[:,:-1].columns.tolist())
#   ax_y = st.multiselect('Выберите колонку (ось Y)', my_data.iloc[:,:-1].columns.tolist()) 
#   fig = sns.catplot(data = my_data, x=ax_x[0], y=ax_y[0], hue=tar[0], 
#                     height=10, aspect=1.5)
#   st.pyplot(fig)

#------------------OwnPlot--------------------
vizDiff = st.checkbox('Постройте свой график')
if vizDiff:
  plotType = st.selectbox('Выберите вид графика',
                       ('area chart', 'bar chart', 'line chart'))
  plotCols = st.multiselect('Выберите колонки по которым будем строить график',
                       my_data.columns.tolist())
  genPlot = st.button('Построить график')
  if plotType == 'area chart':
    if genPlot:
      st.write('*Area chart* - отображает количественные данные в графическом виде')
      st.area_chart(my_data[plotCols].iloc[:100])
  if plotType == 'bar chart':
    if genPlot:
      st.write('*Bar chart* - показывает изменения за определенный период времени. Также используется для сравнения значений данных нескольких объектов')
      st.bar_chart(my_data[plotCols].iloc[:100])
  if plotType == 'line chart':
    if genPlot:
      st.line_chart(my_data[plotCols].iloc[:100]) 

