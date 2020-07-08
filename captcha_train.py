import os
import joblib
import pandas as pd
from sklearn.neural_network import MLPClassifier


def captcha_train(model_save_path, train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    x_train = train_data.drop(columns=['label'])
    y_train = to_categorical(train_data['label'])
    x_test = test_data.drop(columns=['label'])
    y_test = to_categorical(test_data['label'])

    mlp = MLPClassifier(solver='sgd', activation='relu', alpha=1e-5, hidden_layer_sizes=(32, 32), tol=1e-5,
                        random_state=1, max_iter=100, verbose=10, learning_rate_init=0.1)
    mlp.fit(x_train, y_train)

    # 查看模型结果
    print(f'score = {mlp.score(x_test, y_test)}')
    print(f'mlp_layers = {mlp.n_layers_}')
    print(f'iter_times = {mlp.n_iter_}')
    print(f'loss = {mlp.loss_}')
    print(f'out_activation = {mlp.out_activation_}')

    # 保存模型
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        joblib.dump(mlp, model_save_path + '/model.pkl')


def to_categorical(labels):

    one_hot_labels = []
    for num in labels:
        one_hot = [0] * 10
        one_hot[num] = 1
        one_hot_labels.append(one_hot)
    return one_hot_labels
