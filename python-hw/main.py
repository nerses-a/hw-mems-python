from RISnaEVM.config import first_time
import os

if first_time == "yes":
    # Список необходимых директорий
    required_dirs = ["Data", "Plots"]

    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"Директория '{dir_name}' уже существует")
        else:
            os.makedirs(dir_name)
            print(f"Директория '{dir_name}' создана")

    print("Проверка директорий завершена")

    from RISnaEVM import dreif
    from RISnaEVM import mk

print('\n\nПРОАНАЛИЗИРУЕМ МАСШТАБНЫЙ КОЭФФИЦИЕНТ\n')
from RISnaEVM import proc_dreif

print('\n\nПРОАНАЛИЗИРУЕМ ДРЕЙФ ГИРОСКОПА\n')
from RISnaEVM import proc_mk