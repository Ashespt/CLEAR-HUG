import matplotlib.pyplot as plt
import numpy as np

# Create a sample array with shape (1, 1024)
def draw_plot(data):
    colors1 = plt.cm.get_cmap('tab10', 10)
    colors2 = plt.cm.get_cmap('Dark2', 2)
    plt.figure(figsize=(30, 6))

# 每隔96个点画一条线
    for i in range(0, len(data), 96):
        
        if i//96 <= 9:
            color = colors1(i // 96)
            plt.plot(range(i, min(i+96, len(data))), data[i:i+96],linewidth=4,color=color)
        else:
            color = colors2(i // 96-10)
            plt.plot(range(i, min(i+96, len(data))), data[i:i+96],linewidth=4,color=color)

    plt.xticks(np.arange(0, len(data), 96))
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.show()
    plt.savefig('cls_token.png')
