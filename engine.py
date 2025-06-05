import torch 
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from model.lora import CALora
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def class_vectors(model, dataloader, device):
    '''
    calculate the mean vector of each class, in the last layer of the model before linear layer

    '''
    return 
    
def class_vectors(model, dataloader, device, num_classes=10):
    
    class_features = [[] for _ in range(num_classes)]
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        features = model.get_features(images)
        for i in range(len(labels)):
            class_features[labels[i].item()].append(features[i].detach().cpu())
    class_means = [torch.stack(f).mean(dim=0) if len(f) > 0 else torch.zeros_like(features[0].cpu()) for f in class_features]
    return torch.stack(class_means).to(device)  # [num_classes, feature_dim]

def cosine_angle_loss(features, labels, class_centers):
    features = F.normalize(features, dim=1)
    centers = F.normalize(class_centers[labels], dim=1)
    cosine_sim = (features * centers)
    # print(cosine_sim.shape)
    cosine_sim = cosine_sim.sum(dim=1) 
    return 1 - cosine_sim.mean()

def cosine_similarity_matrix(features, class_centers):
    """
    Compute cosine similarity between each feature and all class centers.
    
    Args:
        features: Tensor of shape [batch_size, feature_dim]
        class_centers: Tensor of shape [num_classes, feature_dim]
    
    Returns:
        Tensor of shape [batch_size, num_classes] with cosine similarities
    """
    # Normalize both features and centers
    features = F.normalize(features, dim=1)  # [B, D]
    centers = F.normalize(class_centers, dim=1)  # [C, D]

    # Compute cosine similarity: [B, C] = [B, D] @ [D, C]
    cosine_sim = features @ centers.T
    # softmax cosine similarity to get probabilities
    cosine_sim = F.softmax(cosine_sim, dim=1)  # Optional: if you want probabilities
    return cosine_sim  # shape: [batch_size, num_classes]



@torch.no_grad()
def extract_features(net, dataloader, device):
    net.eval()
    all_features = []
    all_labels = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        _ = net(inputs)
        feats = net.get_features(inputs)
        all_features.append(feats)
        all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return features, labels

def get_plot(
    net, train_loader, index=0
):
    features, labels = extract_features(net, train_loader, device=next(net.parameters()).device)

    # Compute class means
    class_means = []
    for i in range(10):
        class_feat = features[labels == i]
        class_mean = class_feat.mean(dim=0)
        class_means.append(class_mean)

    class_means = torch.stack(class_means)
    normalized_means = F.normalize(class_means, p=2, dim=1)  # [10, D]

    # Compute cosine similarity matrix
    cos_sim_matrix = torch.matmul(normalized_means, normalized_means.T)  # [10, 10]

    # Convert to numpy for visualization
    cos_sim_np = cos_sim_matrix.cpu().numpy()

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cos_sim_np, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Cosine Similarity between Class Mean Vectors")
    plt.xlabel("Class")
    plt.ylabel("Class")
    plt.show()
    plt.savefig(f"pics/cosine_similarity_class_means{index}.png")

    weights = net.fc.weight.data  

    device = next(net.parameters()).device

    means_norm = F.normalize(class_means.to(device), p=2, dim=1)
    weights_norm = F.normalize(net.fc.weight.data.to(device), p=2, dim=1)


    sim_matrix = torch.matmul(means_norm, weights_norm.T)



    plt.figure(figsize=(8, 6))
    
    sns.heatmap(sim_matrix.cpu().numpy(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)


    plt.title("Cosine Similarity between Class Means and Classifier Weights")
    plt.xlabel("Classifier Weights")
    plt.ylabel("Class Means")
    plt.show()
    plt.savefig(f"pics/cosine_similarity_class_means_weights{index}.png")

    
    device = next(net.parameters()).device
    means_norm = F.normalize(class_means.to(device), p=2, dim=1)
    weights_norm = F.normalize(net.fc.weight.data.to(device), p=2, dim=1)

    
    sim_matrix = torch.matmul(means_norm, weights_norm.T)

    
    np.set_printoptions(precision=4, suppress=True)
    print("Cosine similarity matrix between class means and classifier weights:")
    print(sim_matrix.cpu().numpy())
    
    # features_norm = F.normalize(features.to(device), p=2, dim=1)  # [N, D]
    # sim_to_weights = torch.matmul(features_norm, weights_norm.T)  # [N, 10]

    # pred_labels = sim_to_weights.argmax(dim=1)
    # correct = (pred_labels == labels.to(device)).sum().item()
    # total = labels.size(0)
    # accuracy = correct / total

    # print(f"Cosine Similarity Classification Accuracy: {accuracy * 100:.2f}%")
    features_norm = F.normalize(features.to(device), p=2, dim=1)  # [N, D]
    sim_to_weights = torch.matmul(features_norm, weights_norm.T)  # [N, 10]

    top1_preds = sim_to_weights.argmax(dim=1)
    correct_top1 = (top1_preds == labels.to(device)).sum().item()

    # Top-3 prediction
    top3_preds = sim_to_weights.topk(3, dim=1).indices  # [N, 3]
    correct_top3 = (top3_preds == labels.unsqueeze(1).to(device)).any(dim=1).sum().item()

    total = labels.size(0)
    acc_top1 = correct_top1 / total
    acc_top3 = correct_top3 / total

    print(f"Top-1 Cosine Similarity Accuracy: {acc_top1 * 100:.2f}%")
    print(f"Top-3 Cosine Similarity Accuracy: {acc_top3 * 100:.2f}%")




def train(model, dataloader, criterion, optimizer, test_loader, device, epochs=10, parser=None):
    use_lora = parser.use_lora if parser else False
    
    if use_lora:
        # train from scratch in first n epochs 
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
            get_plot(model, test_loader, index=epoch)
        
        model = CALora(model, rank=8, alpha=16).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # fine-tune the model
        for epoch in range(epochs):
            model.train()
            class_centers = class_vectors(model, dataloader, device)
            total_loss = 0
            correct = 0
            total = 0
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                
                # loss = criterion(outputs, labels)
                features = model.get_features(images)
                # print(features.requires_grad)
                # loss = cosine_angle_loss(features, labels, class_centers)
                cosine_sim = cosine_similarity_matrix(features, class_centers)
                outputs = model(images, pseudo_index=cosine_sim)
                loss = criterion(outputs, labels) * 0.1
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}, Accuracy: {100.*correct/total:.2f}%")
            evaluate(model, test_loader, device)
        # get_plot(model, dataloader, i=1)
    else:
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
        

# def evaluate(model, dataloader, device):
#     model.eval()
#     class_correct = torch.zeros(10).to(device)
#     class_total = torch.zeros(10).to(device)
#     y_true = []
#     y_pred = []
#     with torch.no_grad():
#         for images, labels in dataloader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             y_true.extend(labels.cpu().numpy())
#             y_pred.extend(predicted.cpu().numpy())
#             for i in range(len(labels)):
#                 class_total[labels[i]] += 1
#                 class_correct[labels[i]] += (predicted[i] == labels[i]).item()

#     prf_macro = precision_recall_fscore_support(y_true, y_pred, average='macro',)
#     prf_micro = precision_recall_fscore_support(y_true, y_pred, average='micro')
#     prf_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted')
#     print(f"Macro Precision: {prf_macro[0]:.4f}, Recall: {prf_macro[1]:.4f}, F1-Score: {prf_macro[2]:.4f}")
#     print(f"Micro Precision: {prf_micro[0]:.4f}, Recall: {prf_micro[1]:.4f}, F1-Score: {prf_micro[2]:.4f}")
#     print(f"Weighted Precision: {prf_weighted[0]:.4f}, Recall: {prf_weighted[1]:.4f}, F1-Score: {prf_weighted[2]:.4f}")
#     acc_weighted = sum(class_correct) / sum(class_total)
#     print(f"Weighted Accuracy: {acc_weighted:.4f}")
    
#     print("Class-wise Accuracy:", class_correct / class_total)
#     print("variance of each class:", torch.var(class_correct / class_total))

def evaluate(model, dataloader, device):
    model.eval()
    class_correct = torch.zeros(10).to(device)
    class_total = torch.zeros(10).to(device)
    y_true = []
    y_pred = []

    
    rank_sums = torch.zeros(10).to(device)
    rank_counts = torch.zeros(10).to(device)

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            for i in range(len(labels)):
                label = labels[i].item()
                pred_probs = probs[i]

                
                sorted_indices = torch.argsort(pred_probs, descending=True)
                rank = (sorted_indices == label).nonzero(as_tuple=True)[0].item() + 1  # 加1从1开始计

                rank_sums[label] += rank
                rank_counts[label] += 1

                class_total[label] += 1
                class_correct[label] += (predicted[i] == labels[i]).item()

    
    avg_ranks = rank_sums / rank_counts


    prf_macro = precision_recall_fscore_support(y_true, y_pred, average='macro')
    prf_micro = precision_recall_fscore_support(y_true, y_pred, average='micro')
    prf_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    print(f"Macro Precision: {prf_macro[0]:.4f}, Recall: {prf_macro[1]:.4f}, F1-Score: {prf_macro[2]:.4f}")
    print(f"Micro Precision: {prf_micro[0]:.4f}, Recall: {prf_micro[1]:.4f}, F1-Score: {prf_micro[2]:.4f}")
    print(f"Weighted Precision: {prf_weighted[0]:.4f}, Recall: {prf_weighted[1]:.4f}, F1-Score: {prf_weighted[2]:.4f}")
    
    acc_weighted = sum(class_correct) / sum(class_total)
    print(f"Weighted Accuracy: {acc_weighted:.4f}")
    
    class_accuracy = class_correct / class_total
    print("Class-wise Accuracy:", class_accuracy)
    print("Variance of class-wise accuracy:", torch.var(class_accuracy))

    print("Average softmax ranking of true class for each label:")
    for i in range(10):
        print(f"Class {i}: {avg_ranks[i]:.2f}")

def compute_class_means(model, dataloader, device):
    model.eval()
    class_features = [[] for _ in range(10)]

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            features = model.get_features(images)  
            for i in range(len(labels)):
                class_features[labels[i].item()].append(features[i].detach().cpu())

    class_means = [torch.stack(f).mean(dim=0) for f in class_features]
    return torch.stack(class_means)  # [10, feature_dim]

