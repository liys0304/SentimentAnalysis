import tensorflow as tf
import pandas as pd
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from keras import regularizers
# 读取数据
df = pd.read_csv('./data/train_with_scores.tsv', sep='\t', encoding='utf-8')
data = df[['label', 'score', 'pos_freq', 'neg_freq', 'test_len']]
X = data.iloc[:, 1:].values  # 特征
y = data.iloc[:, 0].values  # 标签

# 数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)  # 变为二维数组

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.InputLayer(input_shape=(X.shape[1],)),  # 输入层
    layers.Dense(128, activation='leaky_relu', kernel_regularizer=regularizers.l2(0.001)),  # 隐藏层 + 正则化
    layers.Dropout(0.5),  # Dropout防止过拟合
    layers.Dense(64, activation='leaky_relu', kernel_regularizer=regularizers.l2(0.001)),  # 添加另一隐藏层
    layers.Dense(3, activation='softmax')  # 输出层 (分类数为3)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 添加 EarlyStopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping])

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")



#
# # 构建模型
# model = models.Sequential([
#     layers.InputLayer(input_shape=(1,)),
#     layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.1)),  # 增加L2正则化
#     layers.Dropout(0.3),  # 增加Dropout比例
#     layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.1)),
#     layers.Dropout(0.3),
#     layers.Dense(3, activation='softmax')
# ])
#
# # 编译模型
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# from keras.callbacks import EarlyStopping
#
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#
# # 训练模型
# history = model.fit(X_train, y_train, epochs=100,
#                     class_weight=class_weight_dict,
#                     validation_data=(X_test, y_test),
#                     callbacks=[early_stopping])


# 使用其他评价指标
from sklearn.metrics import classification_report

y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1', 'Class 2']))

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

