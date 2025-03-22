1. Файл [Parser_and_Surface_Finale](https://github.com/jassalm/Resume-Projects/blob/main/Поверхность/Parser_and_Surface_Finale.ipynb) 

Содержит в себе применение парсера и построение поверхности. Мы воспользовались специализированным API (https://iss.moex.com/iss/apps/option-calc/v1/docs#/). Парсер выгружает данные по опционам за текущий момент.

Также мы учли что опционы на валютную пару не слишком ликвидны, поэтому собрали данные по опционам на акции сбера.

2. Файл [pars](https://github.com/jassalm/Resume-Projects/blob/main/Поверхность/pars.py)

Модуль с функциями для парсера, который используется в файле выше.

3. Файл [Options](https://github.com/jassalm/Resume-Projects/blob/main/Поверхность/Options.ipynb)

Содержит в себе реализацию трех вариантов парсера исторических данных по опционам.

4. Файл [parsing](https://github.com/jassalm/Resume-Projects/blob/main/Поверхность/parsing.py)

Модуль с функциями для парсеров в блокноте Options.
