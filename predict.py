import joblib
import pandas as pd
import data_preprocessing as dp
import img_2_csv
import os


GIF_SAVE_PATH = 'D:/Test_GIF'
RAW_IMG_PATH = 'D:/Test_img/'
SEG_IMG_PATH = 'D:/Test_dataset'
test_gif_num = 10
CSV_SAVE_PATH = 'D:/test'


def predict():

    # 爬取动图
    dp.data_spider(GIF_SAVE_PATH, test_gif_num, RAW_IMG_PATH)

    # 灰度处理 & 二值化
    dp.binary_image(RAW_IMG_PATH)

    # 去噪
    crack_result = dp.ClearNoise(RAW_IMG_PATH)
    crack_result.clear_noise(SEG_IMG_PATH, 1)

    # img --> csv
    img_2_csv.convert_img_to_csv(SEG_IMG_PATH, CSV_SAVE_PATH, 1)

    # 预测, 拼接, 重命名
    data = pd.read_csv(f'{CSV_SAVE_PATH}.csv')
    # data = np.array(data).reshape(1, -1)
    model = joblib.load('model/model.pkl')
    predict_result = model.predict(data)

    # 预测
    for i in range(test_gif_num):

        crack_result = {'1': '', '2': '', '3': '', '4': ''}
        count = 0
        temp = ''
        while count < 16:
            img_no = i * 16 + count
            flag = 0
            for j in range(len(predict_result[img_no])):
                if predict_result[img_no][j] == 1:
                    temp += str(j)
                    flag = 1
            if flag == 0:  # 空缺值用'a'代替
                temp += 'a'
            if (count + 1) % 4 == 0:
                crack_result[str(int((count+1)/4))] = temp
                temp = ''
            count += 1

        # 拼接

        # 前两位
        if crack_result.get('4')[:2] == crack_result.get('3')[:2] or crack_result.get('2')[:2] == crack_result.get('3')[:2]:
            final_result = crack_result.get('3')[:2]
        else:
            final_result = crack_result.get('4')[:2]
        # 后两位
        final_result += crack_result.get('1')[-2:]
        print(f'final_result: {final_result}')

        # 根据预测结果重命名GIF
        raw_name = f'{GIF_SAVE_PATH}/{i}.gif'
        dst_name = f'{GIF_SAVE_PATH}/{final_result}.gif'
        try:
            os.rename(raw_name, dst_name)
        except Exception as e:
            print(e)
            print(f'{final_result} rename file fail\r\n')
        else:
            print(f'{final_result} rename file success\r\n')
