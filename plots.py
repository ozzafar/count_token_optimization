import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# --- Amount VS AME Plot ---

# MAE values were taken by evaluating instaflow with YOLO on CIFAR10 classes
data = {
    'Number of Objects': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
    'MAE': [7.142857, 0.714286, 2.428571, 3.0, 5.142857, 7.428571, 7.857143, 9.857143,14.428571, 8.285714, 5.571429, 9.142857,11.285714,10.714286, 8.142857, 7.714286, 9.428571, 6.166667,10.285714,11.285714,13.142857,14.428571,     11.0,     12.0,11.857143,13.285714,17.142857,14.166667,17.285714,21.14285]
}
df = pd.DataFrame(data)

sns.set(style="whitegrid")
plt.figure(figsize=(20, 6))

# Create the line plot
sns.lineplot(x='Number of Objects', y='MAE', data=df, marker='o', color='#270fd9')
# plt.title("MAE per amount")
plt.savefig('plots/amount_vs_mae.png', dpi=300, bbox_inches='tight')
plt.show()
