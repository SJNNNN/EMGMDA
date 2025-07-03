# from MAMFGAT import MAMFGAT
from torch import optim,nn
from tqdm import trange
from mulitmodel_utils import k_matrix
# from Triplet_loss_method import MAMFGAT
# from utils import mutual_information_graph
from multimodal import MAMFGAT
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import dgl
import networkx as nx
import copy
import numpy as np
import torch as th
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc,accuracy_score, precision_score, recall_score, f1_score,roc_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef
)
from sklearn.model_selection import KFold
import torch.nn.functional as F
import scipy.sparse as sp
import pandas as pd
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

kfolds=5

def print_met(list):
    print('AUC ：%.4f ' % (list[0]),
          'AUPR ：%.4f ' % (list[1]),
          'Accuracy ：%.4f ' % (list[2]),
          'precision ：%.4f ' % (list[3]),
          'recall ：%.4f ' % (list[4]),
          'f1_score ：%.4f \n' % (list[5]),
          'Specificity：%.4f \n' % (list[6]),
          'MCC：%.4f \n' % (list[7]))
def print_met2(list):
    print('AUC ：%.4f ' % (list[0]),
          'AUPR ：%.4f ' % (list[1]),
          'Accuracy ：%.4f ' % (list[2]),
          'precision ：%.4f ' % (list[3]),
          'recall ：%.4f ' % (list[4]),
          'f1_score ：%.4f \n' % (list[5]),
          'Specificity：%.4f \n' % (list[6]),
          'MCC：%.4f \n' % (list[7]))

def loss_contrastive_m(m1,m2):
    m1,m2= (m1/th.norm(m1)),(m2/th.norm(m2))
    pos_m1_m2 = th.sum(m1 * m2, dim=1, keepdim=True)
    neg_m1 = th.matmul(m1, m1.t())
    neg_m2 = th.matmul(m2, m2.t())
    neg_m1 = neg_m1 - th.diag_embed(th.diag(neg_m1))
    neg_m2 = neg_m2 - th.diag_embed(th.diag(neg_m2))
    pos_m = th.mean(th.cat([pos_m1_m2],dim=1),dim=1)
    neg_m = th.mean(th.cat([neg_m1, neg_m2], dim=1), dim=1)
    loss_m = th.mean(F.softplus(neg_m-pos_m))

    return loss_m


def loss_mutual_information(d1, d2, input_dim, hidden_dim):
    fc1 = th.nn.Linear(input_dim, hidden_dim).to(d1.device)

    d1 = fc1(d1)
    d2 = fc1(d2)

    # Bilinear product
    score = th.matmul(d1, d2.t())

    # Mutual information maximization
    pos_mask = th.eye(d1.size(0), dtype=th.bool, device=d1.device)
    neg_mask = ~pos_mask

    pos_score = score[pos_mask].view(d1.size(0), -1)
    neg_score = score[neg_mask].view(d1.size(0), -1)

    # Loss calculation
    pos_loss = -F.logsigmoid(pos_score).mean()
    neg_loss = F.softplus(neg_score).mean()

    loss = pos_loss + neg_loss

    return loss
def loss_contrastive_d(d1,d2):
    d1, d2 = d1/th.norm(d1), d2/th.norm(d2)
    pos_d1_d2 = th.sum(d1 * d2, dim=1, keepdim=True)
    neg_d1 = th.matmul(d1, d1.t())
    neg_d2 = th.matmul(d2, d2.t())
    neg_d1 = neg_d1 - th.diag_embed(th.diag(neg_d1))
    neg_d2 = neg_d2 - th.diag_embed(th.diag(neg_d2))
    pos_d = th.mean(th.cat([pos_d1_d2], dim=1), dim=1)
    neg_d = th.mean(th.cat([neg_d1, neg_d2], dim=1), dim=1)
    loss_d = th.mean(F.softplus(neg_d-pos_d ))

    return loss_d



class AdaptiveTripletMarginLoss(nn.Module):
    def __init__(self, initial_margin=0.5, p=2.0, margin_update_rate=0.0001):
        super(AdaptiveTripletMarginLoss, self).__init__()
        self.margin = initial_margin
        self.p = p
        self.margin_update_rate = margin_update_rate

    def forward(self, anchor, positive, negative):
        distance_pos = th.norm(anchor - positive, p=self.p, dim=1)
        distance_neg = th.norm(anchor - negative, p=self.p, dim=1)

        # 计算当前的动态边界
        dynamic_margin = self.margin + self.margin_update_rate * (distance_pos.mean() - distance_neg.mean())

        losses = th.clamp(distance_pos - distance_neg + dynamic_margin, min=0.0)
        return losses.mean()


# 示例：使用加权的 Triplet Loss


def visualize_graph(graph):
    """
    Visualize the constructed graph.

    Parameters:
    - graph: Graph to visualize (numpy array, square matrix).
    """
    G = nx.Graph()

    # Add nodes
    for i in range(graph.shape[0]):
        G.add_node(i)

    # Add edges
    threshold = np.percentile(graph, 95)  # Adjust threshold as needed
    for i in range(graph.shape[0]):
        for j in range(i + 1, graph.shape[1]):
            if graph[i, j] > threshold:
                G.add_edge(i, j, weight=graph[i, j])

    # Draw graph
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, edge_color="gray")
    plt.title("Graph Visualization")
    plt.show()

def train(data,args):
    all_score=[]
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=1)
    train_idx, valid_idx = [], []
    for train_index, valid_index in kf.split(data['train_samples']):
        train_idx.append(train_index)
        valid_idx.append(valid_index)
    max_test_auc = 0
    tpr_list = []
    fpr_list = []
    roc_auc_list = []
    for i in range(kfolds):
        one_score=[]
        model = MAMFGAT(args).to(device)
        optimizer = optim.AdamW(model.parameters(), weight_decay=args.wd, lr=args.lr)
        cross_entropy = nn.BCELoss()
        triplet_loss_fn = nn.TripletMarginLoss(margin=0.5, p=2)
        # triplet_loss_fn = AdaptiveTripletMarginLoss(initial_margin=0.5, p=2.0, margin_update_rate=0.0001)

        miRNA = data['ms']
        disease = data['ds']
        a, b = data['train_samples'][train_idx[i]], data['train_samples'][valid_idx[i]]
        # print(a.shape)
        # print(b.shape)
        c = data['triplet_samples']
        e,f = data['img'][train_idx[i]], data['img'][valid_idx[i]]
        print(f'################Fold {i + 1} of {kfolds}################')
        epochs = trange(args.epochs, desc='train')
        for _ in epochs:
            alpha = nn.Parameter(th.tensor(1.0), requires_grad=True)
            beta = nn.Parameter(th.tensor(1.0), requires_grad=True)
            # gamma = nn.Parameter(th.tensor(1.0), requires_grad=True)
            model.train()
            # weight_sum = alpha + beta + gamma
            # alpha = alpha / weight_sum
            # beta = beta / weight_sum
            # gamma = gamma / weight_sum
            optimizer.zero_grad()
            # mm_matrix = diffusion_maps_graph(data['ms'], args.neighbor)
            # dd_matrix = diffusion_maps_graph(data['ds'], args.neighbor)

            mm_matrix = k_matrix(data['ms'], args.neighbor)
            dd_matrix = k_matrix(data['ds'], args.neighbor)
            # visualize_graph(mm_matrix)
            mm_nx=nx.from_numpy_array(mm_matrix)
            dd_nx=nx.from_numpy_array(dd_matrix)
            mm_graph = dgl.from_networkx(mm_nx)
            dd_graph = dgl.from_networkx(dd_nx)
            md_copy = copy.deepcopy(data['train_md'])

            md_copy[:, 1] = md_copy[:, 1] + args.miRNA_number
            md_graph = dgl.graph(
                (np.concatenate((md_copy[:, 0], md_copy[:, 1])), np.concatenate((md_copy[:, 1], md_copy[:, 0]))),
                num_nodes=args.miRNA_number + args.disease_number)
            miRNA_th=th.Tensor(miRNA)
            disease_th=th.Tensor(disease)

            #train_samples_th = th.Tensor(data['train_samples']).float()
            train_samples_th = th.Tensor(a).float()
            train_img_th = th.Tensor(e).to(device)
            test_img_th = th.Tensor(f).to(device)
            train_score, anchor_mm, positive_dd, negative_dd = model(mm_graph, dd_graph, md_graph, miRNA_th, disease_th, a,c,train_img_th)

            # Compute triplet loss
            triplet_loss = triplet_loss_fn(anchor_mm, positive_dd, negative_dd)
            # print(train_samples_th[:, 2])
            # print(th.flatten(train_score))
            # loss_mutual = loss_mutual_information(d1, d2, 64, 16) + loss_mutual_information(m1, m2, 64, 16)
            train_cross_loss = cross_entropy(th.flatten(train_score), train_samples_th[:, 2].to(device))

            # train_loss = alpha*train_cross_loss + beta*train_d_loss+ gamma*train_m_loss
            train_loss =  alpha*train_cross_loss+beta*triplet_loss
            # train_loss = train_cross_loss +  triplet_loss
            scoree, _, _, _ = model(mm_graph, dd_graph, md_graph, miRNA_th, disease_th, b,c,test_img_th )
            scoree = scoree.cpu()
            scoree = scoree.detach().numpy()
            # score=score.detach().numpy()

            sc = data['train_samples'][valid_idx[i]]
            sc_true = sc[:, 2]

            aucc = roc_auc_score(sc_true, scoree)

            print("AUC=",np.round(aucc,4),"l_1=",np.round(triplet_loss.item(),4),"loss=",np.round(train_loss.item(),4))
            train_loss.backward()
            #print(train_loss.item())
            optimizer.step()

        model.eval()

        #score = model(mm_graph, dd_graph, md_graph, miRNA_th, disease_th, data['unsamples'])
        scoree,_,_,_ = model(mm_graph, dd_graph, md_graph, miRNA_th, disease_th, b,c,test_img_th)

        scoree = scoree.cpu()
        scoree = scoree.detach().numpy()
        # score=score.detach().numpy()

        sc=data['train_samples'][valid_idx[i]]
        sc_true=sc[:,2]
        fold_results = pd.DataFrame({
            'True Labels': sc_true,
            'Predictions': scoree.ravel()
        })
        fold_results.to_csv(f'result/mirRNA_img/HMDD V3.2/fold_{i + 1}_results.csv', index=False)  # Saving to CSV

        fpr, tpr, thresholds = roc_curve(sc_true, scoree)
        # 选择最佳阈值
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print("Best threshold：{:.4f}".format(optimal_threshold))
        fpr, tpr, _ = roc_curve(sc_true, scoree)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label=f'{i+1}Fold ROC curve (area = {roc_auc:.4f})')
        # 存储每一折的结果
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        roc_auc_list.append(roc_auc)
        #计算auc
        aucc = roc_auc_score(sc_true, scoree)
        # if aucc > max_test_auc:
        #     th.save(model.state_dict(), "./save_model/5_fold/HMDD v2.0_5fold_train_model.pth")
        precision, recall, thresholds = precision_recall_curve(sc_true, scoree)
        print("AUC: {:.6f}".format(aucc))
        # plt.plot(recall, precision)
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title('Precision-Recall Curve')
        # plt.show()

        auprc = auc(recall, precision)
        print("AUPRC: {:.6f}".format(auprc))

        scoree=np.array(scoree)
        # scoree=np.around(scoree, 0).astype(int)
        scoree = scoree.ravel()


        for i in range(len(scoree)):
            if scoree[i] >=optimal_threshold:
                scoree[i]=1
            else:
                scoree[i]=0
        accuracy = accuracy_score(sc_true, scoree)
        print("Accuracy: {:.6f}".format(accuracy))
        precision = precision_score(sc_true, scoree)
        print("Precision: {:.6f}".format(precision))
        recall = recall_score(sc_true, scoree)
        print("Recall: {:.6f}".format(recall))
        f1 = f1_score(sc_true, scoree)
        print("F1-score: {:.6f}".format(f1))
        tn, fp, fn, tp = confusion_matrix(sc_true, scoree).ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        print("Specificity: {:.6f}".format(specificity))

        mcc = matthews_corrcoef(sc_true, scoree)
        print("MCC: {:.6f}".format(mcc))
        #print(np.concatenate((data['m_num'][data['unsamples']],score),axis=1))
        one_score=[aucc,auprc,accuracy,precision,recall,f1,  specificity,mcc]
        all_score.append(one_score)
    cv_metric = np.mean(all_score, axis=0)
    sD_metric = np.std(all_score, axis=0)
    print('################5-Fold Result################')
    print_met(cv_metric)
    print_met2(sD_metric)
    # mean_fpr = np.linspace(0, 1, 100)
    # mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fpr_list, tpr_list)], axis=0)
    # mean_auc = auc(mean_fpr, mean_tpr)
    # plt.plot(mean_fpr, mean_tpr, 'k--', lw=2, label=f'Mean ROC curve (area = {cv_metric[0]:.4f})')
    #
    # # 绘制其他细节
    # plt.plot([0, 1], [0, 1], 'r--', lw=2)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC)')
    # plt.legend(loc='lower right')
    # plt.grid(True)
    # output_directory = './result/img_fusion_V3.2/'  # 替换为实际的文件夹路径
    # output_filename = f'{k}times_fivefold_ROC_curve.png'
    # output_path = f'{output_directory}/{output_filename}'
    # plt.savefig(output_path)
    # plt.close()
    # plt.show()
    # return scoree
    return cv_metric, sD_metric