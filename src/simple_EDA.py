# импортируем необходимые библиотеки
import streamlit as st
import pandas as pd 
import numpy as np 
from datetime import datetime 

import plotly.graph_objects as go
import plotly.express as px


#задаем загаловок сайта
st.markdown('''<h1 style='text-align: center; color: black;'
            >Разведочный анализ данных</h1>''', 
            unsafe_allow_html=True)


st.image(image='../images/Eda.png',use_column_width='auto')

st.header('Актуальность тематики', anchor='relevance') 
st.write("""
Разведочный анализ данных - один из первых этапов в машинном обучении. Благодаря такому анализу мы получаем предварительную информацию об исходном наборе данных, что позволяет нам находить зависимости и 
закономерности, оценить распределение данных, строить гипотезы, выявлять пропущенные значениы, выбросы или просто ошибки в данных, представлять информацию в понятным виде с помощью графиков, 
продумывать логику дальнейшего построения алгоритма обучения модели. Это настолько важный этап, что есть целая профессия под такую задачу - Дата-аналитика. Специалисты из этой области занимаются как раз 
разведочным анализом данных с помощью различных инструментов визуализации в BI-системах (Business intelegence). Дата-аналитики - востребованные специалисты в самых разных сферах (как в гумманитарных, 
так и в технических). Именно поэтому лабораторная работа **"Разведочный анализ данных"** будет актуальна для всех, кто хочет познакомиться с основами машинного обучения и возможностями дата-аналитики.
""")

st.write("""Цель лабораторной работы - познакомить студента с базовыми инструментами разведочного анализа данных.
\nОна состоит из **3 блоков**:
\n* **Анализ данных**: знакомимся с данными с помощью базовых методов библиотеки pandas
\n* **Визуализация**: знакомимся с базовыми графиками и их построением   
\n* **Лабораторная работа**: отвечаем на вопросы по данным, используя новые знания
\n **Полезно почитать про разведочный анализ данных:** [1](https://wiki.loginom.ru/articles/exploratory-analysis.html), [2](https://www.helenkapatsa.ru/razvedochnyy-analiz-dannykh-chast-1/), 
[3](https://www.helenkapatsa.ru/razvedochnyy-analiz-dannykh-chast-2/)
""")

#-------------------------О проекте-------------------------


#добавим фотографию пайплайн стримлита

st.header('Этапы разработки лабораторной работы', anchor='pipeline') 

st.image(image= '../images/Pipeline.png', use_column_width='auto', caption='Схема (пайплайн) лабораторной работы')


pipeline_bar = st.expander("Описание пайплайна лабораторной работы")
pipeline_bar.markdown(
  """
  \n**Этапы:**
  \n*(зелёным обозначены этапы, работа с которыми доступна студенту, красным - этапы, доступные для корректировки сотрудникам ЛИА)*
  \n**1. Сбор данных:** были использованы 2 набора данных от [(данные для классификации)](https://github.com/Kaushik-Varma/Marketing_Data_Analysis), [данные для регрессии]();
  \n**2. Обработка данных:** обработка пропущенных значений и ошибок 
  \n**3. Выбор данных для визуализации:** поиск и отбор наиболее популярных видов графиков
	\n**4. Визуализация данных:** построение графиков с помощью инструментов визуализации
	\n**5. Задачи для лабораторной работы:** решение задач к лабораторной работе для усвоение материала
  \n**6. Веб-приложение Streamlit:** работа с данными
  \nИспользуемые библиотеки [streamlit](https://docs.streamlit.io/library/get-started), [pandas](https://pandas.pydata.org/docs/user_guide/index.html), [plotly](https://plotly.com)
  """)

#Даем студенту возможность выбрать самому задачу и данные (вместе с их описанием)
options = st.radio('Выберите тип задачи',
  ('Задача классификации', 'Задача регрессии'))

type_info = st.expander('Информация о типах задач')
type_info.markdown("""
Классификация и регрессия - две задачи машинного обучения которые относятся к разделу [обучение с учителем](http://www.machinelearning.ru/wiki/index.php?title=Обучение_с_учителем)
(когда у нас есть данные с конкретными ответами, мы пытаемся найти зависимость между ними, а затем построить алгоритм, который будет давать ответ для нового набора данных.)
\n**Задача классификации** - предсказание дискретного значения по заданному набору признаков, например прогноз оттока банковских клиентов (бинарнрая классификация, так как только 2 возможных варианта исхода: 
уйдет или нет) или  классификация фруктов на изображении (многоклассовая классиификация, так как у нас может быть много разных фруктов). Число допустимых ответов ограничено. 
\n**Задача регрессии** - предсказание непрерывного значения по заданному набору признаков, например прогноз погоды или стоимости домов. Число допустимых ответов неограниченно 
""")


if options == 'Задача классификации':
  expander_info = st.expander("Информация о данных:")
  expander_info.markdown(
"""
\n**marketing.csv**: информация о маркетинговом исследовании компании в рамках рекламной акции. Набор данных содержит личные сведения о каждом респонденте, продолжительность разговора и прошлый опыт взаимодейтсвия
Целевая переменная - была принята рекламная акция (столбец "ответ").
""")
  col_expander = st.expander('Описание столбцов:')
  col_expander.markdown("""
  \n**возраст** - Возраст клиента
  \n**зарплата** - Зарплата клиента в долларах 
  \n**работа** - Место работы клиента
  \n**образование** - Уровень образования клиента
  \n**баланс** - Количество денег на счету в долларах
  \n**статус** - Семейное положение клиента 
  \n**таргетировали** - Проводили ли ранее таргетированную рекламу на клиенте  
  \n**квартира** - Наличие квартиры у клиента 
  \n**кредит** - Наличие кредита у клиента
  \n**Дата** - Когда был произведен звонок клиенту   
  \n**тип_связи** - Как связывались с клиентом 
  \n**продолжительность** - Продолжительность звонка 
  \n**предложение** - Тип предлагаемой акции
  \n**прошлый_звонок** - Сколько дней назад был совершен прошлый звонок (-1 если не был ни разу)
  \n**прошлый_исход** - Как прошел прошлый звонок 
  \n**ответ** - Принял ли предложение клиент
  """)


  my_data = pd.read_csv('data/marketing.csv')

if options == 'Задача регрессии':
  expander_bar = st.expander("Информация о данных:")
  expander_bar.markdown(
"""
\n**house_prices.csv**: исследование объявлений с сервиса Яндекс.Недвижимость о продаже квартир в Санкт-Петербурге. Набор данных содержит информацию о самой квартире, ее расположении, наличии по близости аэропортов/прудов/парков
и т.д. Целевая переменная - стоимость кваритры.
""")

  col_expander = st.expander('Описание столбцов:')
  col_expander.markdown("""
  \n**фотки** - Количество фоток в объявлении
  \n**площадь_квартиры** - Площадь квартиры в квадратных метрах
  \n**дата публикации** - Когда выложили объявление
  \n**команты** - Число комнат в квартире
  \n**потолки** - Высота потолков в метрах
  \n**этажность** - Всего этажей в доме
  \n**жил_площадь** - Жилая площадь в квадратных метрах
  \n**этаж** -  На каком этаже расположена квартира
  \n**апартаменты** - Является ли квартира апартаментом (1 - да, 0 - нет)
  \n**студия** -  Является ли квартира студией (1 - да, 0 - нет)
  \n**свободная_планировка** - Наличие свободной планировки (1- да, 0 - нет)
  \n**площадь_кухни** - Площадь кухни в метрах квадратных
  \n**балконы** - Число балконов
  \n**населенный_пункт** - Название населённого пункта
  \n**до_аэропорта** - Расстояние до ближайшего аэропорта в метрах
  \n**до_центра_города** - Расстояние до центра города в метрах
  \n**парки** - Число парков в радиусе 3 километров
  \n**до_парка** - Расстояние до ближайшего парка в метрах
  \n**водоемы** - Число водоёмов в радиусе 3 километров
  \n**до_водоема** - Расстояние до ближайшего водоёма в метрах
  \n**количество_дней_публикации** - Сколько дней размещено объявление
  \n**стоимость** - Стоимость квартиры в рублях
  """)

  my_data = pd.read_csv('data/house_prices.csv') 


#---------------------Знакомимся с данными----------- 
st.subheader('Анализ данных')

#покажем сам датасет, студент сможет сам котнролировать количество выводимых строк
if st.checkbox('Показать набор данных'):
  # number = st.number_input('Сколько строк показать', min_value=1, max_value=my_data.shape[1])
  st.dataframe(my_data)


#покажем по отдельности кол-во строк и столбцов
if st.checkbox('Размер набора данных'):
  shape = st.radio(
    "Выбор данных",
     ('Строки', 'Столбцы'))
  if shape == 'Строки':
    st.write('Количество строк:', my_data.shape[0])
  elif shape == 'Столбцы':
    st.write('Количество столбцов:', my_data.shape[1])

#студент может выбрать один или несколько столбцов для просмотра
show_cols = st.checkbox('Выберите столбцы, на которые хотите посмотреть')
if show_cols:
  cols_to_show = st.multiselect('Столбцы', 
  my_data.columns.tolist())
  if not cols_to_show:
    pass
  else:
    st.dataframe(my_data[cols_to_show])

#Посмотрим на уникальные значения 
n_unique = st.checkbox('Уникальные значения столбца')
if n_unique:
  n_cols = st.multiselect('Столбцы ', 
  my_data.columns.tolist())
  if not n_cols:
    pass
  else:
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

non_val = st.checkbox('Пропущенные значения')
if non_val:
  st.write(pd.DataFrame(my_data.isnull().sum().sort_values(ascending=False), columns=['количество пропущенных значений']))


#-----------------Строим графики---------------
st.subheader('Визуализация данных')
st.write('Ознакомьтесь с 5 разными видами графиков')

#-----------------ПайЧарт--------------------
with st.expander('Круговая диаграмма PieChart'):
  st.write("""Круговая диаграмма, которая разделена на срезы, иллюстрирующие числовую пропорцию. Длина дуги каждого среза пропорциональна величине, которую она представляет, 
  то есть показывет долю от целого (пропорцию или процентное соотношение)""")
  st.image(image='../images/pie_chart.png')
  with st.form(key='pie'):
    col_option = st.selectbox('Выберите колонку', my_data.columns)
    pie_df = my_data[col_option].value_counts()
    pie_fig = go.Figure(data=[go.Pie(labels=pie_df.index, values=pie_df.tolist(), textinfo='label+percent',
                    insidetextorientation='radial')])
    pie_fig.update_traces(textposition='inside', textinfo='percent+label')
    if st.form_submit_button('Построить'):
      st.plotly_chart(pie_fig, use_container_width=True)

#-----------------Гистограмма--------------------
with st.expander('Гисторамма распределения Histogram'):
  st.write("""Статистический график, который показывает распредление величины. Визуализирует распределение данных в рамках непрерывного интервала или ограниченного периода времени. 
  Каждая полоса на гистограмме представляет в табличной форме частотность за определенный интервал. Общая площадь гистограммы равна количеству данных.""")
  st.image(image='../images/hist.png', use_column_width=True)
  with st.form(key='hist'):
    option = st.selectbox('Выберите колонку', my_data.columns)
    if st.form_submit_button('Построить'):
      hist = px.histogram(my_data, x=option)
      st.plotly_chart(hist, use_container_width=True)


#------------------График корреляции--------------------
with st.expander('График корреляции Correlation Heatmap'):

  st.write('Графическое представление корреляционной матрицы, которая показывает зависимость между числовыми объектами')
  st.info(''' Корреляция - статистическая взаимосвязь двух или более переменных. Изменения значений одной переменной сопутствуют изменениям другой.
  \nВ нашем случае зависимость выражется на промежутке от -1 до 1. 
  \nЧем ближе значение к 1, тем сильнее прямая зависимость: увеличивая значения одной переменной, увеличивается значение и второй.
  \nЧем ближе значение к -1, тем сильнее обратная зависимость: увеличивая значение одной переменной, уменьшается значение второй и наоброт. 
  \n Чтобы узнать значение корреляции на нашем графике, нужно навести курсор на интересующий прямоугольник и посмотреть на значение 'z'
  ''')
  st.image(image='../images/heat.png')
  with st.form(key='corr_matrix'):
    df_corr = my_data.corr()
    fig_corr = go.Figure([go.Heatmap(z=df_corr.values,
                                 y=df_corr.index.values,
                                 x=df_corr.columns.values, 
                                 zmin=-1, zmax=1)])
    fig_corr.update_layout(width=640, height=340, margin=dict(b=0, l=0, r=2, t=0)) # bottom, left, right и top - отступы     title='Наша таблица', title_x=0.5, title_y=1,
    # fig = go.Figure(my_data.corr(), aspect="aut)
    if st.form_submit_button('Построить'):
      st.plotly_chart(fig_corr)

#------------------Ящик с усами--------------------
with st.expander('График формы распределения BoxPlot (Ящик с усами)'):
  st.write(""" Диаграмма показывает распределение значений в выборке и основные статистические показатели: медиану (линия внутри ящика), верхний(75%) и нижний(25%) квартили, наблюдаемые минимумы и максимумы (усы), а также выбросы""")
  st.info(''' Квартили -  значения, которые делят данные на 4 группы (25%,50%,75%,100%), содержащие приблизительно равное количество наблюдений. 
  \nПо сути, это то же самое, что и перцентиль. То есть нижний квартиль - 25 перцентиль, а верхний квартиль - 75 перцентиль
  ''')

  st.image(image='../images/boxplot.png', use_column_width='auto')

  with st.form(key='box_plot'):

    ax_x = st.selectbox('Выберите, что будет по горизонтали', my_data.columns.tolist())
    ax_y = st.selectbox('Выберите, что будет по вертикали', my_data.columns.tolist())
    button_box = st.form_submit_button('Построить')
    if button_box:
      box = px.box(my_data, x=ax_x, y=ax_y)
      st.plotly_chart(box, use_container_width=True)

#------------------Диаграмма рассеяния---------------
with st.expander('Диаграмма рассеяния ScatterPlot'):
  st.write("""Диаграмма показывает распределение элементов множества в плоскости между двумя переменными. Значения одного независимого параметра откладываются по оси X, значения второго 
  зависимого параметра – по оси Y. Это статистическая диаграмма и ее также используют для нахождения корреляции или выбросов""")
  st.image(image='../images/scatter.png')
  with st.form(key='scatter'):
    ax_x = st.selectbox('Выберите, что будет по горизонтали', my_data.columns.tolist())
    ax_y = st.selectbox('Выберите, что будет по вертикали', my_data.columns.tolist())
    scat = px.scatter(my_data, x=ax_x, y=ax_y)#, trendline="ols")
    if st.form_submit_button('Построить'):
      st.plotly_chart(scat, use_container_width=True)

#-------------------Задачи к лабораторной работе--------------------
st.subheader('Практическая часть')
st.write('Ответьте на вопросы о данных с помощью методов разведочного анализа данных')

#для каждого датасета по 2-3 задачи 
options_2 = st.radio('Выберите тип задачи :',
  ('Классификация', 'Регрессия'))

if options_2 == 'Классификация':
  st.write("""
          1. Какой тип образования самый распространенный среди клиентов? Сколько людей им обладает и какая это доля от целого?
          2. Клиентов какого типа профессии больше: офисный сотрудник или сфера услуг? В ответе укажите точное число обоих типов.
          3. У каких двух столбцов самая сильная прямая корреляция? А у каких обратная? укажите пары столбцоы и их корреляцию.
          4. Посмотрите на статус клиента. У какого типа из представленных есть выбросы в значениях по зарплате? В ответе укажите тип и все значения выбросов.
          5. У трех клиентов на балансе есть больше 80.000 долларов. Сколько лет этим клиентам?
          """)
if options_2 == 'Регрессия':
  st.write("""
          1. Какое максимальное количество балконо в квартире? Сколько есть таких объявлений и какую доли они занимают от всех объявлений? 
          2. Публикаций с каким количеством фотографий больше: 10 или 20? В ответе укажите точное количество публикаций обоих видов.
          3. Какие 5 признаков сильнее всего влияют на стоимость квартир?
          4. Только 6 квартир стоят больше 140 миллионов. На каких этажах они находятся? 
          5. Правда ли, что квартира с самой большой площадью самая дорогая? В ответе укажите площадь и стоимость этой квартиры.
          """)

st.subheader('Подведем итоги')
summary = st.expander('Какие выводы можно сделать?')
summary.markdown("""
Мы с вами познакомились с важным этапом в машинном обучении - разведочным анализом данных. Это довольно наглядный способ, так как с помощью визуализации можно легко понять, что происходит внутри
данных. Тем не менее такой анализ подразумевает сильную экспертизу и хорошую интуицию, так как грамотно искать полезную информацию в данных довольно сложно. Именно поэтому это довольно долгий 
этап.
""")