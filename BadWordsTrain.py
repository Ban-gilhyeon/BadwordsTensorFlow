import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm  # 진행 상황 모니터링용
import numpy as np
from sklearn.metrics import label_ranking_average_precision_score, classification_report

# 1. 데이터셋 로드 및 확인
dataset = load_dataset('smilegate-ai/kor_unsmile')
data_train = dataset["train"]
data_valid = dataset["valid"]

print("첫 샘플:", data_train[0])

# 2. 모델 및 토크나이저 초기화
model_name = 'beomi/kcbert-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 첫 샘플의 'labels' 길이를 이용해 num_labels 결정
num_labels = len(data_train[0]['labels'])
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


# 3. CustomDataset 클래스 정의 (PyTorch Dataset 사용)
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


# 4. 데이터셋 전처리: 텍스트와 라벨 추출 (컬럼명이 "문장"과 "labels"로 가정)
train_texts = data_train["문장"]
train_labels = data_train["labels"]
val_texts = data_valid["문장"]
val_labels = data_valid["labels"]

max_length = 128  # 최대 길이 설정
train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)

# 5. DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 6. 옵티마이저와 학습 파라미터 설정
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
epochs = 1

# 7. 학습 루프 (tqdm으로 진행 상황 모니터링)
for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
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

# 8. 평가 단계
model.eval()
scores = []
all_preds = []
all_labels = []

eval_bar = tqdm(val_loader, desc="Evaluation", leave=False)
for batch in eval_bar:
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits

        # 배치별 평가 점수 계산
        score = label_ranking_average_precision_score(labels.cpu().numpy(), logits.cpu().numpy())
        scores.append(score)

        preds = torch.sigmoid(logits).cpu().numpy()  # 로짓을 시그모이드 함수로 변환
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

# 9. 최종 평가 결과 출력 (임계값 0.5 적용)
all_preds = np.array(all_preds) > 0.5
report = classification_report(np.array(all_labels), all_preds)
print("Classification Report:")
print(report)