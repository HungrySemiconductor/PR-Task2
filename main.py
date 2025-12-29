import numpy as np
from data_loader import prepare_data, split_data


class NeuralNetwork():
    # 网络参数
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # 初始化权重
        self.init_weights()

    # ==================== 权重初始化 ====================
    def init_weights(self):
        # 输入层到隐含层
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))

        # 隐含层到输出层
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))

    # ==================== 激活函数及其求导 ====================
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1.0 - np.tanh(x)**2


    # ==================== 前向传播 ====================
    def forward(self, X):
        # 输入层到隐含层
        self.net_h = np.dot(X, self.W1) + self.b1  
        self.y_h  = self.tanh(self.net_h)          
        
        # 隐含层到输出层
        self.net_j = np.dot(self.y_h , self.W2) + self.b2   
        self.z_j = self.sigmoid(self.net_j)    

        return self.z_j 


    # ==================== 损失函数计算 MSE ====================
    def compute_loss(self, y_true, y_pred):
        batch_size = y_true.shape[0]
        loss = 0.5 * np.sum((y_true - y_pred) ** 2) / batch_size
        return loss


    # ==================== 反向传播 ====================

    # 单样本更新
    def train_single_sample(self, X, y, verbose=False):
        pass

    # 批量更新
    def train_batch(self, X_batch, y_batch, verbose=False):
        pass



    # ==================== 训练循环 ====================
    def train(self, X_train, y_train, X_val, y_val, epochs=1000, batch_type='batch', verbose=100):
        pass

    # ==================== 预测评估 ====================
    # 预测
    def predict(self, X):
        pass

    # 计算准确率
    def evaluate(self, X, y):
        pass



