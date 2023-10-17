import cv2 # библиотека компьютерного зрения
import numpy as np
import matplotlib.pyplot as plt
import pytesseract # Модуль работы с Tesseract – ПО для распознавания текста
from PIL import Image # Работа с изображениями
import glob, os # Библиотеки для поиска файлов в директории, работы с ОС
#для обучения модели
import tensorflow as tf
from sklearn.metrics import f1_score 
from keras import optimizers
from keras.models import Sequential
from keras.models import Sequential,model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
import keras.backend as K

def find_chars(contour_list):
    MAX_DIAG_MULTIPLYER = 5 # 5
    MAX_ANGLE_DIFF = 12.0 # 12.0
    MAX_AREA_DIFF = 0.5 # 0.5
    MAX_WIDTH_DIFF = 0.8
    MAX_HEIGHT_DIFF = 0.2
    MIN_N_MATCHED = 3 # 3    
    matched_result_idx = []
    
    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue

            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])

        # добавляем этот контур
        matched_contours_idx.append(d1['idx'])

        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        matched_result_idx.append(matched_contours_idx)

        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        unmatched_contour = np.take(contour_list, unmatched_contour_idx)
        
        # вызов ф-ции find_chars
        recursive_contour_list = find_chars(unmatched_contour)
        
        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    return matched_result_idx


def find_contours(dimensions, img) :

#Найти все контуры на изображении
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#Получим потенциальные размеры
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
#Проверьте самые большие  15 контуров для номерного знака или символа 
#соответственно
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    ii = cv2.imread('contour.jpg')
    
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        #Определим контур в бинарном изображении и возвращает координаты 	
#окружающего его прямоугольника
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        #Проверка размеров контура для фильтрации символов по размеру контура
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) #сохраняет координату x контура персонажа, чтобы использовать ее позже для индексации контуров

            char_copy = np.zeros((44,24))
            #извлечение символов с помощью координат окружающего прямоугольника.
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
            plt.imshow(ii, cmap='gray')
            plt.axis('off')

                   # Отформатируем результат для классификации: инвертировать цвета
            char = cv2.subtract(255, char)

# Изменяем размер изображения до 24x44 с черной рамкой
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) # Список, в котором хранится бинарное изображение (несортированное)           
# Возвращает символы в порядке возрастания относительно координаты x 
    plt.title("Отдельные символы номера")
    #plt.show()
# функция, хранящая отсортированный список индексов символов 
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])#сохраняет изображения персонажей по их индексу
    img_res = np.array(img_res_copy)
    return img_res

def segment_characters(image, save_path) :

    # Предварительно обработать обрезанное изображение номерного знака
    img_lp = cv2.resize(image, (333, 75))

    LP_WIDTH = img_lp.shape[0]
    LP_HEIGHT = img_lp.shape[1]

    # Сделаем границы белыми
    img_lp[0:3,:] = 0#255
    img_lp[:,0:3] = 0#255
    img_lp[72:75,:] = 0#255
    img_lp[:,330:333] = 0#255

    # Оценка размеров контуров символов обрезанных номеров

    dimensions = [LP_WIDTH/6,
                  LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
    plt.imshow(img_lp, cmap='gray')
    plt.axis('off')
    cv2.imwrite(save_path, img_lp)
    plt.title("Контур номера")
    #plt.show()
    
    cv2.imwrite('contour.jpg', img_lp)


    # Получить контуры в обрезанном номерном знаке
    char_list = find_contours(dimensions, img_lp)

    return char_list

def example(path, save_path, is_demo=False):
    img_ori = cv2.imread(path) # Считаем изображение по пути path
    
    height, width, channel = img_ori.shape # Получим его параметры
    
    plt.figure(figsize=(12, 10))
    plt.imshow(img_ori, cmap='gray') # Отрисуем его
    plt.axis('off')
    if is_demo: plt.savefig('Car.png',bbox_inches = 'tight')
    plt.title("Загружаем картинку")
    #plt.show()    
    
    # Переведем изображение из цветного в серое
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(gray, cmap='gray')
    plt.axis('off')
    plt.title("Переводим её в серый")
    if is_demo: plt.savefig('1_Car-GrayScale.png',bbox_inches = 'tight')
    #plt.show()    
    
    # Увеличим контрастность
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)
    
    imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
    gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(gray, cmap='gray')
    plt.axis('off')
    plt.title("Увеличиваем контраст")
    if is_demo: plt.savefig('2_Car-Contrast.png',bbox_inches = 'tight')
    #plt.show()    
     
    # Применим размытие по Гауссу
    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
    
    # Применим threshold
    img_thresh = cv2.adaptiveThreshold(
        img_blurred, 
        maxValue=255.0, 
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        thresholdType=cv2.THRESH_BINARY_INV, 
        blockSize=19, 
        C=9
    )
    
    plt.figure(figsize=(12, 10))
    plt.imshow(img_thresh, cmap='gray')
    plt.axis('off')
    plt.title("Adaptive Thresholding")
    if is_demo: plt.savefig('3_Car-Adaptive-Thresholding.png',bbox_inches = 'tight')
    #plt.show()    
    
    # Найдем контуры на изображении
    contours, _= cv2.findContours(
        img_thresh, 
        mode=cv2.RETR_LIST, 
        method=cv2.CHAIN_APPROX_SIMPLE
    )
    
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    
    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))
    
    plt.figure(figsize=(12, 10))
    plt.imshow(temp_result)
    plt.axis('off')
    plt.title("Находим контуры")
    if is_demo: plt.savefig('4_Car-Contours.png',bbox_inches = 'tight')
    #plt.show()    
    
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    
    contours_dict = []
    
    for contour in contours:
        # Пройдем по каждому контуру и впишем их в прямоугольники
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
        
        # insert to dict
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })
    
    plt.figure(figsize=(12, 10))
    plt.imshow(temp_result, cmap='gray')
    plt.axis('off')
    plt.title("Загоняем контуры в \"Прямоугольники\"")
    if is_demo: plt.savefig('5_Car-Boxes.png',bbox_inches = 'tight')
    #plt.show()    
    
    MIN_AREA = 80
    MIN_WIDTH, MIN_HEIGHT = 2, 8
    MIN_RATIO, MAX_RATIO = 0.25, 1.0
    
    possible_contours = []
    
    cnt = 0
    # Отсеем ненужные контуры
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']
        
        if area > MIN_AREA \
        and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
        and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)
            
    # визуализируем возможные контуры
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    
    for d in possible_contours:
        #cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(temp_result, cmap='gray')
    plt.axis('off')
    plt.title("Фильтруем полдученные прямоугольники по размеру")
    if is_demo: plt.savefig('6_Car-Boxes-byCharSize.png',bbox_inches = 'tight')
    #plt.show()    
    
    # попробуем найти симвыолы на том, что осталось после фильтрации
    result_idx = find_chars(possible_contours)
    
    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))
    
    # визуализируем возможные контуры
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    
    for r in matched_result:
        for d in r:
            #cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(temp_result, cmap='gray')
    plt.axis('off')
    plt.title("Находим прямоугольники, содержащие символы номера")
    if is_demo: plt.savefig('7_Car-Boxes-byContourArrangement.png',bbox_inches = 'tight')
    #plt.show()    
    
    result_idx = find_chars(possible_contours)
    
    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    
    for r in matched_result:
        for d in r:
            #cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
            cv2.rectangle(img_ori, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(0, 0, 255), thickness=2)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(img_ori, cmap='gray')
    plt.axis('off')
    plt.title("Накладываем найденные прямоугольники на нашу картинку")
    if is_demo: plt.savefig('8_Car-OverlappingBoxes.png',bbox_inches = 'tight')
    #plt.show()    
    
    PLATE_WIDTH_PADDING = 1.3 # 1.3
    PLATE_HEIGHT_PADDING = 1.5 # 1.5
    MIN_PLATE_RATIO = 3
    MAX_PLATE_RATIO = 10
    
    plate_imgs = []
    plate_infos = []
    # Обрежем и повернем потенциальные символы, чтобы их можно было распознать
    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])
    
        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
        
        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
        
        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h']
    
        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
        
        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )
        
        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
        
        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
        
        img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))
        
        img_cropped = cv2.getRectSubPix(
            img_rotated, 
            patchSize=(int(plate_width), int(plate_height)), 
            center=(int(plate_cx), int(plate_cy))
        )
        
        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
            continue
        
        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })
        
        plt.subplot(len(matched_result), 1, i+1)
        plt.imshow(img_cropped, cmap='gray')
        plt.axis('off')
        plt.title("Обрезаем номер и поворачиваем при необходимости")
        if is_demo: plt.savefig('9_Car-Plates(Rotated).png',bbox_inches = 'tight')
        #plt.show()    
    
    longest_idx, longest_text = -1, 0
    plate_chars = []
    
    for i, plate_img in enumerate(plate_imgs):
        plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
        _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
        #найдем контуры снова (так же, как выше)
        contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    
        plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
        plate_max_x, plate_max_y = 0, 0
    
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
    
            area = w * h
            ratio = w / h
    
            if area > MIN_AREA \
               and w > MIN_WIDTH and h > MIN_HEIGHT \
            and MIN_RATIO < ratio < MAX_RATIO:
                if x < plate_min_x:
                    plate_min_x = x
                if y < plate_min_y:
                    plate_min_y = y
                if x + w > plate_max_x:
                    plate_max_x = x + w
                if y + h > plate_max_y:
                    plate_max_y = y + h
    
        img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]
    
        img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
        _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
        plt.subplot(len(plate_imgs), 1, i+1)
        plt.imshow(img_result, cmap='gray')
        plt.axis('off')
        plt.title("Снова применяем thresholding")
        if is_demo: plt.savefig('10_Car-Plates(Thresholding).png',bbox_inches = 'tight')
        #plt.show()
        break    
    
    img = 255-img_result
    plt.imshow(img, 'gray')
    plt.axis('off')
    plt.title("Берем негатив от полученного на предыдущем шаге номера")
    if is_demo: plt.savefig('11_Car-Plates(Negative).png',bbox_inches = 'tight')
    #plt.show()
     # Отделим символы (для использования в дальнейшем)
    char = segment_characters(img, save_path)
    for i in range(len(char)):
        plt.subplot(1, len(char), i+1)
        plt.imshow(char[i], cmap='gray')
        plt.axis('off')
    #plt.savefig(save_path, bbox_inches = 'tight')
    plt.title("Сохраняем номер с отделенными символами")
    if is_demo: plt.savefig('12_Car-Plates-Char(Seperated).png',bbox_inches = 'tight')    
    if is_demo: print("Далее необходимо разбить таким образом все номера и нарезать их в выборку, на которой мы будем обучать модель")
    
if __name__ == "__main__":    
         # Если папки dataset не существует
    if not os.path.isdir('dataset'):
                  # Просим пользователя скачать датасет
        url = "https://www.kaggle.com/datasets/andrewmvd/car-plate-detection?datasetId=686454&sortBy=voteCount"
        print(f"Скачайте датасет по ссылке {url}\nРаспакуйте в папку dataset")
        os.mkdir('dataset') # Создаем папку для датасета
          # Аналогично, если нет папки для номеров – создаем ее
    if not os.path.isdir('plates'): 
        os.mkdir('plates')

    #Пример, который покажет/сохранит картинки, где будет пошагово описан #процесс работы
    example('dataset\\images\\Cars1.png', "0.png", True)

    # Инструкция по установке Tesseract
# # https://medium.com/nuances-of-programming/обнаружение-и-извлечение-текста-из-изображения-с-помощью-python-ae178219e8b9

    # Путь к tesseract.exe необходимо поменять на свой (если Вы не Кирилл, конечно 😊 )
    pytesseract.pytesseract.tesseract_cmd = r'"C:\Program Files\Tesseract-OCR\tesseract.exe"' 
          # Пройдем по всем падающим под маску «что-то, заканчивающееся на .png»
    for i, file in enumerate(glob.glob('dataset\\images\\*.png')):
        try: # попробуем выполнить кусок кода ниже
                           # Обрабатываем картинку, скормив путь к ней функции example
            example(file, f"plates\\{i}.png") 
            print("[V]",i, file, sep='\t')
            # Извлекаем номер с помощью Tesseract и выводим его на экран
            txt = pytesseract.image_to_string(f"plates\\{i}.png")
            print("\t",txt)
        except Exception: # Если не вышло – напишем об этом пользователю
            print("[X]",i, file, sep='\t')
