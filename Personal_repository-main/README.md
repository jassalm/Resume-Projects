## Проект "Анализ ваканский на hh.ru по трем ключевым направлениям IT: Data. Frontend, Backend

## Цель проекта:
Проект направлен на анализ данных о заработных платах. Основная цель – выявить текущие тенденции на рынке труда, определить средний уровень заработной платы для различных позиций, а также изучить зависимости заработных плат от различных факторов (например, опыта работы, места расположения, требуемых навыков и технологий).

## Задачи проекта:

### Сбор данных:

> Сбор данных о вакансиях с сайта [hh.ru](https://hh.ru/) для data-related специальностей (data scientist, data analyst, data engineer и т.д.) и программистов (backend, frontend, fullstack, mobile development и т.д.).

Сначала мы [спарсили ссылки на вакансии с API hh.ru](https://github.com/jassalm/Resume-Projects/blob/main/Personal_repository-main/API%20parsing.ipynb). Данные сложили в [файл](https://github.com/jassalm/Resume-Projects/blob/main/Personal_repository-main/Parsed%20data/from_api.xls). Внутри - названия, некоторые даные и ссылки на страницы вакансий. 

Далее мы проитерировались по ссылкам и достали больше данных с страниц, вот наш [парсер](https://github.com/jassalm/Resume-Projects/blob/main/Personal_repository-main/Parser.ipynb). Данные сохранили в [файлик](https://github.com/jassalm/Resume-Projects/blob/main/Personal_repository-main/Parsed%20data/for_cleaning2.xls).


### Очистка данных:

Cделали очистку данных [здесь](https://github.com/jassalm/Resume-Projects/blob/main/Personal_repository-main/Data_cleaner.ipynb). Сохранили данные в [файл](https://github.com/jassalm/Resume-Projects/blob/main/Personal_repository-main/Parsed%20data/final.xls)
[]()

Дополнительно мы сохранили закодированные данные о вакансиях в [файл](https://github.com/jassalm/Resume-Projects/blob/main/Personal_repository-main/Parsed%20data/role_decoder.csv)

### Предобработка данных, визуализация и новые признаки

Сделали предобработку и визуализации [здесь](https://github.com/jassalm/Resume-Projects/blob/main/Personal_repository-main/EDA_project%20(2).ipynb).



Анализ данных:

>Выявление средней и медианной заработной платы для различных позиций.
>Анализ зависимости заработной платы от опыта работы, региона, требуемых навыков и технологий.
>Сравнение уровня заработной платы между различными специальностями в области data-related и программирования.
>Анализ навыков, требуемых на разные направления работы
>Построение графиков и диаграмм для наглядного представления результатов анализа.

Также на этом этапе мы создали новые признаки, связанные с ЗП, направлением вакансии и ее локацией. Эти призаки оказались очень важны на этапе машинного обучения и гипотез: мы либо их предсказывали, либо формулировали гипотезы на их основе.

### Гипотезы

Мы проверили некоторые гипотезы [здесь](https://github.com/jassalm/Resume-Projects/blob/main/Personal_repository-main/Hypothesis%20(2).ipynb)

### Машинное и глубинное обучение

Предсказание:
1) ЗП по признакам
2) Сфера по текстовым признакам (NLP)
3) Кластеризация

Мы сделали еще одну предобработку и обучили множество моделей для предсказания ЗП (задача регрессии) [здесь](https://github.com/jassalm/Resume-Projects/blob/main/Personal_repository-main/ML.ipynb) . 

Текстовый анализ, и предсказание сферы (задача классификации) мы делаем [тут](https://github.com/jassalm/Resume-Projects/blob/main/Personal_repository-main/NLP_part.ipynb). 

Далее [здесь](https://github.com/jassalm/Resume-Projects/blob/main/Personal_repository-main/кластеризация.ipynb) мы сделали кластеризацию и написали небольшую нейросеть вот [тут](https://github.com/jassalm/Resume-Projects/blob/main/Personal_repository-main/нейросеть_проект_python_2_ipynb_.ipynb).
