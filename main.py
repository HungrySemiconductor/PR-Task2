import numpy as np
from data_loader import prepare_data, split_data


class NeuralNetwork():
    # 网络参数
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        pass

    # 权重初始化
    def init_weights(self):
        pass
    
    # 激活函数及其求导
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    
    def tanh_derivation(self, x):
        return 1.0 - np.tanh(x)**2


    # 前向传播
    def forward(self, X):
        pass

    # 损失函数计算
    def compute_loss(self, y_true, y_pred):
        pass

    # 单样本更新
    def train_signle_sample(self, X, y, verbose=False):
        pass

    # 批量更新
    def train_batch(self, X_batch, y_batch, verbose=False):
        pass

    # 训练
    def train(self, X_train, y_train, X_val, y_val, epochs=1000, batch_type='batch', verbose=100):
        pass

    # 预测
    def predict(self, X):
        pass

    # 计算准确率
    def evaluate(self, X, y):
        pass



