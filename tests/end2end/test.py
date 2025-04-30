import os
import sys
import subprocess
import numpy as np
import re
from lxml import etree

def str_to_tensor(s):
    # Удаляем все пробелы и переносы
    s = re.sub(r'\s+', '', s)
    # Заменяем фигурные скобки на квадратные
    s = s.replace('{', '[').replace('}', ']')
    # Преобразуем строку в список через eval (осторожно!)
    data = eval(s)
    return np.array(data, dtype=np.float32)


def compare_svg_structure(file1, file2):
    """Сравнивает структуру SVG, игнорируя пробелы и порядок атрибутов"""
    parser = etree.XMLParser(remove_blank_text=True)

    try:
        # Парсинг файлов
        tree1 = etree.parse(file1, parser)
        tree2 = etree.parse(file2, parser)

        # Функция для нормализации элементов
        def normalize_tree(tree):
            for elem in tree.iter():
                # Создаем новый упорядоченный словарь атрибутов
                if elem.attrib:
                    sorted_attrib = dict(sorted(elem.attrib.items()))
                    elem.attrib.clear()
                    elem.attrib.update(sorted_attrib)
                # Нормализация текстового содержимого
                if elem.text and isinstance(elem.text, str):
                    elem.text = elem.text.strip()
            return tree

        # Нормализация деревьев
        normalize_tree(tree1)
        normalize_tree(tree2)

        # Сравнение
        return etree.tostring(tree1) == etree.tostring(tree2)

    except Exception as e:
        print(f"Ошибка при сравнении SVG: {str(e)}")
        return False

if len(sys.argv) != 4:
    print("Usage: python script.py <dir> <file>")
    sys.exit(1)

dir = sys.argv[1]
binary_dir = sys.argv[2]
file = sys.argv[3]
exec_file = os.path.join(binary_dir, file)
real = os.path.join(dir, "test.txt")
real_svg = os.path.join(dir, "test.svg")
dot = os.path.join(dir, "test.dot")
ref = os.path.join(dir, file + ".ans")
ref_svg = os.path.join(dir, file + ".svg")

# Запуск исполняемого файла и перенаправление ввода/вывода
with open(real, 'w') as outfile, open(dot, 'w') as errfile:
    subprocess.run([exec_file], stdout=outfile, stderr=errfile)

with open(real, 'r') as realf, open(ref, 'r') as reff:

    real_str = realf.read()
    tensor_real = str_to_tensor(real_str)

    ref_str = reff.read()
    tensor_ref = str_to_tensor(ref_str)

print("bark")
subprocess.run(['dot', '-Tsvg', dot, '-o', real_svg])
svg_cmp = compare_svg_structure(real_svg, ref_svg)
# Удаление временного файла
os.remove(real)
os.remove(dot)
os.remove(real_svg)

# Сравнение содержимого
if np.array_equal(tensor_ref, tensor_real) and svg_cmp:
    print("Files match.")
    sys.exit(0)
else:
    print("Files do not match.")
    print("real\n", tensor_real)
    print("ref\n", tensor_ref)
    sys.exit(1)
