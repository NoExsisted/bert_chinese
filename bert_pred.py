import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_PATH = "bert_chinese_classifier"  # 目录

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def predict(text, model, tokenizer, device, max_length=128):
    # 预处理文本
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    # 将输入移到设备上
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    # 获取预测结果
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    _, predicted_idx = torch.max(probabilities, dim=1)
    
    predicted_idx = predicted_idx.item()
    probabilities = probabilities.flatten().cpu().numpy().tolist()
    
    return predicted_idx, probabilities

if __name__ == "__main__":
    label_to_phrase = {
        0: "住房修缮",
        1: "供暖服务",
        2: "供气服务",
        3: "供水服务",
        4: "供电服务",
        5: "公交客运",
        6: "公积金服务",
        7: "养老保险",
        8: "占地补偿",
        9: "城市更新(老旧小区改造)",
        10: "工作作风",
        11: "市政道路",
        12: "房屋交易",
        13: "拖欠工资",
        14: "村容村貌",
        15: "燃气管线申报",
        16: "物流快递",
        17: "运营商服务",
        18: "违法建设",
        19: "违章停车",
        20: "道路修缮"
    }

    # 标签 6, 13, 14, 14
    texts = [
        "杨先生反映：他在北京缴纳的公积金，晋城贷款买房，联系晋城市银行被告知联系支行人员解决，但是周末联系不到，望相关部门告知如何再次扣款。",
        "某先生反映：他在陵川县中学干活，受私人雇佣，未签订劳动合同，老板拖欠几人一共2万7千元未支付。望有关部门予以解决。",
        "李先生反映：他曾于2021年1月5日下午向晋城市12345政务服务热线反映陵川县附城镇川里村书记态度差的问题，因迟迟未收到答复，便于1月8日下午5点15分拨打附城镇政府0356-6866016进行咨询，接电话的工作人员说不清楚此事，然后就给他讲大道理，随后他称“你们这是附城镇政府吗”，接电话的工作人员说“我们这是香港，拜拜”，随即挂断电话。现他认为政府工作人员对待市民服务态度差，望相关部门调查处理，予以答复。",
        "刘先生反映：他是泽州县北石店镇徐家岭村11号住户，女儿刘洁，户口在徐家岭村。2018年在确认村内股权时，因刘洁在上学，保留了股权分配权利，2019年毕业回村，2020年6月入职祥达后勤部。近期村内公布2020年股权分配名单时，因没有女儿的名字，向村委咨询告知，需用人单位开具用人证明才可恢复股权，但村委告知女儿在单位缴纳五险，不符合村内股权分配规定，现因单位缴纳五险导致村内没有股权分配权利是否合理，望相关部门予以告知。"
    ]
    
    # 预测并显示结果
    for text in texts:
        label, probs = predict(text, model, tokenizer, device)
        print(f"文本: {text}")
        print(f"预测类别: {label, label_to_phrase[label]}")
        print(f"各类别概率: {[round(p, 4) for p in probs]}")
        print("-" * 50)