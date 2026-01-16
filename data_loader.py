import numpy as np

def standardize(X):
    """均值归一化：(x - mean) / std"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # 防止除以0
    return (X - mean) / (std + 1e-8)

def train_test_split(X, y, test_size=0.3, random_seed=42):
    """手动实现数据集划分"""
    np.random.seed(random_seed)
    # 生成随机索引
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    # 计算划分界限
    val_count = int(X.shape[0] * test_size)
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

def prepare_data(test_size=0.3, random_seed=42):
    # 1. 原始数据 (基于实验报告要求)
    raw_data = {
        'class1': [
            [1.58, 2.32, -5.8], [0.67, 1.58, -4.78], [1.04, 1.01, -3.63],
            [-1.49, 2.18, -3.39], [-0.41, 1.21, -4.73], [1.39, 3.16, 2.87],
            [1.20, 1.40, -1.89], [-0.92, 1.44, -3.22], [0.45, 1.33, -4.38],
            [-0.76, 0.84, -1.96]
        ],
        'class2': [
            [0.21, 0.03, -2.21], [0.37, 0.28, -1.8], [0.18, 1.22, 0.16],
            [-0.24, 0.93, -1.01], [-1.18, 0.39, -0.39], [0.74, 0.96, -1.16],
            [-0.38, 1.94, -0.48], [0.02, 0.72, -0.17], [0.44, 1.31, -0.14],
            [0.46, 1.49, 0.68]
        ],
        'class3': [
            [-1.54, 1.17, 0.64], [5.41, 3.45, -1.33], [1.55, 0.99, 2.69],
            [1.86, 3.19, 1.51], [1.68, 1.79, -0.87], [3.51, -0.22, -1.39],
            [1.40, -0.44, -0.92], [0.44, 0.83, 1.97], [0.25, 0.68, -0.99],
            [0.66, -0.45, 0.08]
        ]
    }  
    
    X_list = []
    y_list = []
    
    # 2. 标签编码：One-Hot 编码
    for i, (class_name, samples) in enumerate(raw_data.items()):
        X_list.extend(samples)
        for _ in range(len(samples)):
            label = [0, 0, 0]
            label[i] = 1 # 根据类别索引设置1
            y_list.append(label)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    # 3. 数据预处理
    X = standardize(X)
    
    # 4. 划分数据集
    X_train, y_train, X_val, y_val = train_test_split(
        X, y, test_size=test_size, random_seed=random_seed
    )
    
    return X_train, y_train, X_val, y_val

if __name__ == "__main__":
    # 测试代码
    X_train, y_train, X_val, y_val = prepare_data(test_size=0.3)
    
    print(f"====================数据加载测试====================")
    print(f"训练集特征形状: {X_train.shape}") # 预期应为 (21, 3)
    print(f"训练集标签形状: {y_train.shape}") # 预期应为 (21, 3)
    print(f"验证集特征形状: {X_val.shape}")   # 预期应为 (9, 3)
    print(f"验证集标签形状: {y_val.shape}")   # 预期应为 (9, 3)
    print(f"\n前3个训练样本特征 (标准化后):\n{X_train[:3]}")
    print(f"\n前3个训练样本标签 (One-Hot):\n{y_train[:3]}")