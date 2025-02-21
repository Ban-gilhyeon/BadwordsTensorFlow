import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import label_ranking_average_precision_score, classification_report

# 설정 값들
MODEL_NAME = 'beomi/kcbert-base'
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 3  # 빠른 테스트를 위해 에포크 수를 1로 설정
LEARNING_RATE = 1e-5
# MODEL_SAVE_PATH를 절대 경로로 설정 (또는 "saved_model"과 같이 상대 경로 대신 사용)
MODEL_SAVE_PATH = os.path.abspath("./saved_model")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# PyTorch Dataset 정의
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.encodings.items()}
        item['labels'] = self.labels[index]
        return item

    def __len__(self):
        return len(self.labels)


# 데이터셋 로드 및 준비 함수
def load_data():
    dataset = load_dataset('smilegate-ai/kor_unsmile')
    data_train = dataset["train"]
    data_valid = dataset["valid"]
    return data_train, data_valid


def prepare_datasets(tokenizer):
    data_train, data_valid = load_data()
    # 컬럼명이 "문장"과 "labels"라고 가정
    train_texts = data_train["문장"]
    train_labels = data_train["labels"]
    val_texts = data_valid["문장"]
    val_labels = data_valid["labels"]

    train_dataset = CustomDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


# 모델 및 토크나이저 초기화 함수
def initialize_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_train, _ = load_data()
    # 첫 샘플의 labels 길이로 num_labels 결정
    num_labels = len(data_train[0]['labels'])
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    model.to(device)
    return model, tokenizer


# 학습 함수
def train(model, tokenizer):
    train_loader, _ = prepare_datasets(tokenizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)
        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} 완료. 평균 Loss: {avg_loss:.4f}")

    # 학습 후 모델 저장
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print("모델 저장 완료.")


# 평가 함수
def evaluate(model, tokenizer):
    _, val_loader = prepare_datasets(tokenizer)
    model.eval()
    scores = []
    all_preds = []
    all_labels = []

    progress_bar = tqdm(val_loader, desc="Evaluation", leave=False)
    for batch in progress_bar:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            score = label_ranking_average_precision_score(
                labels.cpu().numpy(), logits.cpu().numpy()
            )
            scores.append(score)

            preds = torch.sigmoid(logits).cpu().numpy()  # 확률로 변환
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds) > 0.5  # 임계값 적용
    report = classification_report(np.array(all_labels), all_preds)
    print("Classification Report:")
    print(report)


# 테스트 함수
def test(model, tokenizer, test_text):
    test_encoding = tokenizer(
        test_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )
    test_encoding = {key: val.to(device) for key, val in test_encoding.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**test_encoding)
        probabilities = torch.sigmoid(outputs.logits)
        predictions = (probabilities > 0.5).int()

    print("Test 문장:", test_text)
    print("Probabilities:", probabilities.cpu().numpy())
    print("Predictions:", predictions.cpu().numpy())


if __name__ == '__main__':
    mode = input("Enter 'train' to train model, 'test' to test model: ").strip().lower()

    if mode == 'train':
        model, tokenizer = initialize_model_and_tokenizer()
        train(model, tokenizer)
        evaluate(model, tokenizer)
    elif mode == 'test':
        # 테스트 시 저장된 모델 불러오기
        model = BertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
        model.to(device)
        test_text = "너는 최고야"
        test(model, tokenizer,test_text)
    else:
        print("Invalid mode. Please enter 'train' or 'test'.")