import tensorflow as tf
import pandas as pd

from keras import layers, models
from sklearn.model_selection import train_test_split


df = pd.read_excel('train_data.xlsx')
# 假设第一列为特征，第二列为标签
X = df.iloc[:, 0].values  # 特征
y = df.iloc[:, 1].values  # 标签

# 数据预处理
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, 1))  # 变为二维数组

# # 创建TensorFlow数据集
# dataset = tf.data.Dataset.from_tensor_slices((X, y))
#
# # 进行批处理
# batch_size = 32
# dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建模型
model = models.Sequential([
    layers.InputLayer(input_shape=(X.shape[1],)),  # 输入层
    layers.Dense(64, activation='relu'),  # 隐藏层
    layers.Dense(3, activation='softmax')  # 输出层 (分类数为3)
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型

history = model.fit(X_train, y_train, epochs=100)

# 评估模型
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")
model.save('classify.keras')

import matplotlib.pyplot as plt

# 绘制损失曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

