import captcha_train as ct
import data_preprocessing as dp
import img_2_csv
import predict


def main():
    """
    raw_path = 'D:/闪烁验证码集/'
    binary_path = 'D:/闪烁验证码集/'
    gif_save_path = 'D:/GIF'
    seg_img_path = 'D:/dataSet'
    gif_num = 100

    # 数据爬取
    dp.data_spider(gif_save_path, gif_num, raw_path)

    # 二值化
    dp.binary_image(binary_path)

    # 去噪
    a = dp.ClearNoise(binary_path)
    a.clear_noise(seg_img_path, 0)

    # img --> csv
    img_2_csv.convert_img_to_csv(img_dir='D:/trainSet/', csv_save_path='trainSet', mode=0)
    img_2_csv.convert_img_to_csv(img_dir='D:/testSet/', csv_save_path='testSet', mode=0)

    # 训练
    ct.captcha_train('model', train_path='trainSet.csv', test_path='testSet.csv')
    """
    # 预测
    predict.predict()


if __name__ == '__main__':
    main()
