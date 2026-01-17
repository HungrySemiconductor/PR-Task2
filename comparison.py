import matplotlib.pyplot as plt
from data_loader import prepare_data
from network import NeuralNetwork

def Comparion():
    X_train, y_train, X_val, y_val = prepare_data()
    
    hidden_nodes, learning_rate = 10, 0.1  
    epochs = 200   
    
    plt.figure(figsize=(10, 6))

    # 单样本更新
    nn_single = NeuralNetwork(3, hidden_nodes, 3, learning_rate)
    s_train_loss, s_val_loss = nn_single.train(X_train, y_train, X_val, y_val, update_type='single_update', epochs=epochs, verbose=0)
    plt.plot(s_train_loss, label='Single Train Loss', color='red')
    plt.plot(s_val_loss, label='Single Val Loss', color='red', linestyle='--')

    # 批量更新
    nn_batch = NeuralNetwork(3, hidden_nodes, 3, learning_rate)
    b_train_loss, b_val_loss = nn_batch.train(X_train, y_train, X_val, y_val, update_type='batch_update', epochs=epochs, verbose=0)
    plt.plot(b_train_loss, label='Batch Train Loss', color='blue')
    plt.plot(b_val_loss, label='Batch Val Loss', color='blue', linestyle='--')

    plt.title('Comparison: Single Update vs Batch Update')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
   Comparion()