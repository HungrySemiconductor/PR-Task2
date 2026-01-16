import numpy as np

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
    def train_single_sample(self, X, y, verbose=False): # 最后一个参数是用来控制是否打印损失的
        y_pred = self.forward(X)
        delta_j = self.sigmoid_derivative(self.net_j) * (y - y_pred)
        delta_h = self.tanh_derivative(self.net_h) * np.dot(delta_j, self.W2.T)  # 为什么是转置，为什么不是求和？？

        # 更新权重和偏置
        self.W2 -= self.learning_rate * np.dot(self.y_h.T, delta_j)
        self.b2 -= self.learning_rate * np.sum(delta_j, axis=0, keepdims=True)
        self.W1 -= self.learning_rate * np.dot(X.T, delta_h)
        self.b1 -= self.learning_rate * np.sum(delta_h, axis=0, keepdims=True)
        
        if verbose:
            loss = self.compute_loss(y, y_pred)
            print(f"单样本损失: {loss:.6f}")

    # 批量更新
    def train_batch(self, X_batch, y_batch, verbose=False):
        batch_size = X_batch.shape[0]
        
        y_pred = self.forward(X_batch)
        delta_j = (y_pred - y_batch) * self.sigmoid_derivative(self.net_j)
        delta_h = np.dot(delta_j, self.W2.T) * self.tanh_derivative(self.net_h)
        
        # 计算梯度（批量平均）
        dW2 = np.dot(self.y_h.T, delta_j) / batch_size
        db2 = np.sum(delta_j, axis=0, keepdims=True) / batch_size
        dW1 = np.dot(X_batch.T, delta_h) / batch_size
        db1 = np.sum(delta_h, axis=0, keepdims=True) / batch_size
        
        # 更新权重和偏置
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        
        if verbose:
            loss = self.compute_loss(y_batch, y_pred)
            print(f"批量损失: {loss:.6f}")



    # ==================== 训练循环 ====================
    def train(self, X_train, y_train, X_val, y_val, epochs=1000, batch_type='batch', verbose=100):
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            if batch_type == 'single':
                # 单样本训练
                for i in range(X_train.shape[0]):
                    X_sample = X_train[i:i+1]   # 保持二维形状
                    y_sample = y_train[i:i+1]
                    self.train_single_sample(X_sample, y_sample)
            else:
                # 批量训练
                self.train_batch(X_train, y_train)

            # 计算训练集和验证集损失
            train_pred = self.forward(X_train)
            train_loss = self.compute_loss(y_train, train_pred)
            train_losses.append(train_loss)
            
            val_pred = self.forward(X_val)
            val_loss = self.compute_loss(y_val, val_pred)
            val_losses.append(val_loss)
            
            if verbose and epoch % verbose == 0:
                print(f"Epoch {epoch}: 训练损失={train_loss:.6f}, 验证损失={val_loss:.6f}")
        
        return train_losses, val_losses

    # ==================== 预测评估 ====================
    # 预测
    def predict(self, X):
        y_pred = self.forward(X)
        return (y_pred > 0.5).astype(int)  # 二分类阈值

    # 计算准确率
    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy



if __name__ == "__main__":
    pass