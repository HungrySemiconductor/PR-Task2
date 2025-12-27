import numpy as np

def prepare_data(test_size=0.2, random_seed=42):
    # 1. 加载3个类别的数据集
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
    
    # 2. 合并数据并创建标签
    X_list = []
    y_list = []
    
    for i, (class_name, samples) in enumerate(raw_data.items()):
        X_list.extend(samples)
        # one-hot标签
        for _ in range(len(samples)):
            label = [0, 0, 0]
            label[i] = 1
            y_list.append(label)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    # 3. 数据标准化（可选）
    X = raw_data.standardize(X)
    
    # 4. 划分数据集
    X_train, y_train, X_val, y_val = raw_data.split_data(
        X, y, test_size=test_size, random_seed=random_seed
    )
    
    return X_train, y_train, X_val, y_val
