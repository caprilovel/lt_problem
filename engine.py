import torch 
from sklearn.metrics import precision_recall_fscore_support


def train(model, dataloader, criterion, optimizer, test_loader, device, epochs=10, lora=False):
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}, Accuracy: {100.*correct/total:.2f}%")
        evaluate(model, test_loader, device)
    if lora:
        pass 
        

def evaluate(model, dataloader, device):
    model.eval()
    class_correct = torch.zeros(10).to(device)
    class_total = torch.zeros(10).to(device)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            for i in range(len(labels)):
                class_total[labels[i]] += 1
                class_correct[labels[i]] += (predicted[i] == labels[i]).item()

    prf_macro = precision_recall_fscore_support(y_true, y_pred, average='macro',)
    prf_micro = precision_recall_fscore_support(y_true, y_pred, average='micro')
    prf_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print(f"Macro Precision: {prf_macro[0]:.4f}, Recall: {prf_macro[1]:.4f}, F1-Score: {prf_macro[2]:.4f}")
    print(f"Micro Precision: {prf_micro[0]:.4f}, Recall: {prf_micro[1]:.4f}, F1-Score: {prf_micro[2]:.4f}")
    print(f"Weighted Precision: {prf_weighted[0]:.4f}, Recall: {prf_weighted[1]:.4f}, F1-Score: {prf_weighted[2]:.4f}")
    acc_weighted = sum(class_correct) / sum(class_total)
    print(f"Weighted Accuracy: {acc_weighted:.4f}")
    
    print("Class-wise Accuracy:", class_correct / class_total)