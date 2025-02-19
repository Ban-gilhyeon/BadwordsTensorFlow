import os
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # TensorFlow 메시지 최소화
logging.getLogger("datasets").setLevel(logging.ERROR)  # datasets 로그 최소화

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt  # 학습 그래프 출력
from datasets import load_dataset, DownloadConfig

# 1. 데이터셋 준비 및 다운로드 진행 (진행바 비활성화)
down_config = DownloadConfig(disable_tqdm=True)
ds = load_dataset("smilegate-ai/kor_unsmile", download_config=down_config)

# 2. 데이터 전처리: 텍스트 벡터화
max_tokens = 10000
max_len = 20

vectorize_layer = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=max_len
)
# "train" 분할의 "문장" 열을 사용하여 어휘 사전 구축
vectorize_layer.adapt(ds["train"]["문장"])

# 3. 모델 구성
embedding_dim = 16
num_classes = 10  # 출력 클래스 수 (데이터셋의 레이블 길이가 10인 것으로 가정)

model = models.Sequential([
    layers.Input(shape=(1,), dtype=tf.string),
    vectorize_layer,
    layers.Embedding(input_dim=max_tokens, output_dim=embedding_dim, mask_zero=True),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # 다중 분류를 위한 출력 레이어
])

# 타깃 레이블이 원-핫 인코딩되어 있으므로 categorical_crossentropy 사용
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 4. 학습 및 검증 데이터 준비
# 입력 텍스트를 tf.constant()로 변환해 tf.string 타입으로 지정하고, (n, 1) 모양으로 재조정합니다.
train_texts = tf.constant([str(x) for x in ds["train"]["문장"]])
train_texts = tf.reshape(train_texts, (-1, 1))
train_labels = tf.constant(ds["train"]["labels"])  # 원-핫 인코딩된 레이블
valid_texts = tf.constant([str(x) for x in ds["valid"]["문장"]])
valid_texts = tf.reshape(valid_texts, (-1, 1))
valid_labels = tf.constant(ds["valid"]["labels"])

# 5. 모델 학습
history = model.fit(
    train_texts,
    train_labels,
    epochs=10,
    verbose=1,
    validation_data=(valid_texts, valid_labels)
)

# 6. 검증 데이터 평가
eval_results = model.evaluate(valid_texts, valid_labels, verbose=0)
print("Validation results:", eval_results)

# 7. 모델 저장 (Keras 네이티브 형식: .keras 확장자 사용)
model.save("profanity_filter_model.keras")

# 8. 학습 그래프 그리기 및 저장
plt.figure(figsize=(12, 5))

# 손실 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()