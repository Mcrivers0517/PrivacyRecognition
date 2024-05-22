from transformers import BertTokenizerFast, BertForTokenClassification
import torch
import yaml


def load_model(model_path, num_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForTokenClassification.from_pretrained('./tokenizer/bert-base-chinese', num_labels=num_labels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_category_map(filepath):
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    # 反转映射：从类别ID映射到类别名称
    id_to_category = {v: k for k, v in data['categories'].items()}
    return id_to_category


def detect(model, text, tokenizer, category_map):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoding = tokenizer.encode_plus(text, return_tensors='pt', return_offsets_mapping=True)
    input_ids = encoding['input_ids'].to(device)
    offsets = encoding['offset_mapping'].detach().cpu().numpy()[0]

    with torch.no_grad():
        outputs = model(input_ids)
        predictions = torch.argmax(outputs.logits, dim=-1).detach().cpu().numpy()[0]

    print(predictions)

    # 从文本中重构隐私信息
    privacy_fragments = []
    last_pos = None
    last_token_id = None
    for idx, token_id in enumerate(predictions):
        if token_id != 0:  # 不是非隐私标签
            char_start, char_end = offsets[idx]
            if last_token_id != token_id or last_pos != char_start:
                if last_token_id is not None:
                    privacy_fragments.append('\n')
                privacy_fragments.append(f"{category_map[token_id]}: ")  # 添加类别名称
            privacy_fragments.append(text[char_start:char_end])
            last_pos = char_end
            last_token_id = token_id

    return ''.join(privacy_fragments)


if __name__ == '__main__':
    model_path = './privacy_model.pth'
    num_labels = 15
    tokenizer = BertTokenizerFast.from_pretrained('./tokenizer/bert-base-chinese')
    category_map_path = './categories.yaml'
    category_map = load_category_map(category_map_path)

    model = load_model(model_path, num_labels)

    text = ""
    privacy_info = detect(model, text, tokenizer, category_map)
    print(privacy_info)

