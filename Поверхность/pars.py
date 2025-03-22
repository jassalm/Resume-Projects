import requests
import pandas as pd
from tqdm import tqdm 
import time 

def get_options_list(asset_code: str) -> list:
    """
    Получает список опционов для указанного актива с MOEX API.

    :param asset_code: Код актива (например, 'SBER').
    :return: Список опционов в формате JSON.
    :raises Exception: Если запрос к API завершился ошибкой.
    """
    url = f"https://iss.moex.com/iss/apps/option-calc/v1/assets/{asset_code}/options"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Ошибка при получении списка опционов: {response.status_code}")


def get_option_details(asset_code: str, secid: str) -> dict:
    """
    Получает детальную информацию по конкретному опциону с MOEX API.

    :param asset_code: Код актива (например, 'SBER').
    :param secid: Идентификатор опциона (например, 'SR260CC5A').
    :return: Детальная информация по опциону в формате JSON.
    :raises Exception: Если запрос к API завершился ошибкой.
    """
    url = f"https://iss.moex.com/iss/apps/option-calc/v1/assets/{asset_code}/options/{secid}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Ошибка при получении деталей опциона {secid}: {response.status_code}")


def collect_option_data(asset_code: str, delay: float = 0.1) -> pd.DataFrame:
    """
    Собирает данные по всем опционам для указанного актива.

    :param asset_code: Код актива (например, 'SBER').
    :param delay: Задержка между запросами (в секундах) для избежания блокировки.
    :return: DataFrame с данными по опционам.
    """

    options_list = get_options_list(asset_code)
    
    data = []
    
    for option in tqdm(options_list, desc="Сбор данных по опционам", unit="опцион"):
        secid = option['secid']
        details = get_option_details(asset_code, secid)
        
        # Объединяем базовую информацию и детали
        combined_info = {**option, **details}
        
        data.append(combined_info)
        
        time.sleep(delay)

    df = pd.DataFrame(data)
    
    return df