import torch
from transformers import BertTokenizer, BertForTokenClassification
from torch.utils.data import DataLoader
from dataset import PrivacyDataset
import time
from tqdm import tqdm


def train(model, dataloader, device, num_epochs=200):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    for epoch in range(num_epochs):
        total_loss = 0
        start_time = time.time()  # 记录每个epoch开始的时间
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}",
                            unit="batch")

        for i, batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            model.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 更新进度条的描述信息
            progress_bar.set_postfix(loss=total_loss / (i + 1))

        end_time = time.time()  # 记录每个epoch结束的时间
        elapsed_time = end_time - start_time  # 计算消耗的时间
        print(f"Epoch {epoch + 1}/{num_epochs} completed in {elapsed_time:.2f} seconds")

    torch.save(model.state_dict(), 'privacy_model.pth')
    print('Model saved to privacy_model.pth')


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('./tokenizer/bert-base-chinese')
    dataset = PrivacyDataset('./train/data', './train/label', tokenizer, max_len=128, category_map_path='./categories.yaml')
    dataloader = DataLoader(dataset, batch_size=512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForTokenClassification.from_pretrained('./tokenizer/bert-base-chinese', num_labels=15)
    model.to(device)

    train(model, dataloader, device)
