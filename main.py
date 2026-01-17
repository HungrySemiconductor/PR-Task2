import matplotlib.pyplot as plt
from data_loader import prepare_data
from network import NeuralNetwork

def run(update_type, epochs):
    X_train, y_train, X_val, y_val = prepare_data()
    
    # 实验配置
    hid_nodes = [5, 10, 20]     # 不同的隐含层节点数
    lrs = [0.01, 0.1, 0.5]      # 不同的学习率
    results = {}                # 记录最佳组合

    for h in hid_nodes:
        for lr in lrs:
            print("-" * 50)
            print(f"当前训练参数：hidden_nodes = {h}，learning_rate = {lr}")
            nn = NeuralNetwork(input_size=3, hidden_size=h, output_size=3, learning_rate=lr)
            
            train_losses, val_losses = nn.train(X_train, y_train, X_val, y_val, update_type=update_type, epochs=epochs, verbose=100)
            
            train_acc = nn.evaluate(X_train, y_train)
            val_acc = nn.evaluate(X_val, y_val)

            results[(h, lr)] = val_acc
            print(f"hidden_nodes={h:<10}leaning_rate={lr:<10}train_acc={train_acc:<15.4f}val_acc={val_acc:<15.4f}")

            
            # 固定学习率，比较不同隐含节点数
            if lr == 0.1:
                plt.figure(1)
                p = plt.plot(train_losses, label=f'Train (H={h})') # 训练损失
                color = p[0].get_color()
                plt.plot(val_losses, label=f'Val (H={h})', color=color, linestyle='--') # 验证损失

            if h == 5:
                plt.figure(2)
                p = plt.plot(train_losses, label=f'Train (LR={lrs})') # 训练损失
                color = p[0].get_color()
                plt.plot(val_losses, label=f'Val (LR={lrs})', color=color, linestyle='--') # 验证损失
    
    best_config = max(results, key=results.get)
    print(f"\n[最佳配置] 节点数: {best_config[0]}, 学习率: {best_config[1]}, 验证集最高准确率: {results[best_config]:.4f}")

    
    # 不同学习率对比
    plt.figure(1)
    plt.title('Figure 1: Impact of Hidden Nodes (Fixed LR=0.1)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend(fontsize='small')
    plt.grid(True, alpha=0.3)

    # 不同隐含节点数对比
    plt.figure(2)
    plt.title('Figure 2: Impact of Learning Rate (Fixed Hidden=10)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend(fontsize='small')
    plt.grid(True, alpha=0.3)

    plt.show() # 同时显示两个窗口

if __name__ == "__main__":
    print("================ 单样本更新实验 ================")
    run(update_type='single_update', epochs=200)
    print("================ 批量更新实验 ================")
    run(update_type='batch_update', epochs=500)

