#encoding=utf-8

import numpy as np
import scipy.spatial.distance as spd
import torch
from dataloader import RadarDataSet
from torch.utils.data import DataLoader
from scipy.optimize import minimize
from scipy.stats import genextreme



def gev_nll(params, data):
    xi, loc, scale = params
    if xi <= 0:
        return np.inf  # xi > 0
    return -np.sum(genextreme.logpdf(data, xi, loc=loc, scale=scale))



def cal_distance(query_score,mcv,eu_weight,distance_type='eucos'):
    if distance_type == 'eucos':
        query_distance = spd.euclidean(mcv, query_score) * eu_weight + \
            spd.cosine(mcv, query_score)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mcv, query_score)
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mcv, query_score)
    else:
        print("distance type not known: enter either of eucos, euclidean or cosine")
    return query_distance

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def compute_channel_distances(mavs, features, eu_weight=0.5):
    eucos_dists, eu_dists, cos_dists = [], [], []
    for channel, mcv in enumerate(mavs):  
        eu_dists.append([spd.euclidean(mcv, feat[channel]) for feat in features]) 
        cos_dists.append([spd.cosine(mcv, feat[channel]) for feat in features]) 
        eucos_dists.append([spd.euclidean(mcv, feat[channel]) * eu_weight +
                            spd.cosine(mcv, feat[channel]) for feat in features])
    return {'eucos': np.array(eucos_dists), 'cosine': np.array(cos_dists), 'euclidean': np.array(eu_dists)}



def getMavs(train_class_num,trainloader,device,net):
    scores = [[] for _ in range(train_class_num)]
    hiddensall = [[] for _ in range(train_class_num)]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, hiddens = net.encoder_q(inputs)
            i = 0
            for score,t in zip(outputs,targets):
                if torch.argmax(score) == t:
                    scores[t].append(score.unsqueeze(dim=0).unsqueeze(dim=0))
                    hiddensall[t].append(hiddens[i].unsqueeze(dim=0).unsqueeze(dim=0))
                i+= 1
    scores = [torch.cat(x).cpu().numpy() for x in scores]
    mavs = np.array([np.mean(x, axis=0) for x in scores]) 
    dists = [compute_channel_distances(mcv, score) for mcv, score in zip(mavs, scores)]
    return scores, mavs, dists 

def fit_Frechet(means,dists,categories,tailsize=20,distance_type='eucos'):
    Frechet_models = {}
    for mean,dist,category_name in zip(means,dists,categories):
        Frechet_models[category_name] = {}
        Frechet_models[category_name]['distances_{}'.format(distance_type)] = dist[distance_type]
        Frechet_models[category_name]['mean_vec'] = mean 
        Frechet_models[category_name]['frechet_model'] = []
        for channel in range(mean.shape[0]):
            tailtofit = np.sort(dist[distance_type][channel,:])[-tailsize:]
            initial_params = [0.1,np.mean(tailtofit),np.std(tailtofit)]
            result = minimize(gev_nll,initial_params,args=(tailtofit,),bounds = [(0, None), (None, None), (None, None)]) 
            c,loc,scale = result.x
            Frechet_models[category_name]['frechet_model'].append((c,loc,scale)) 
    return Frechet_models

def query_frechet(category_name, frechet_models, distance_type='eucos'):
    return [frechet_models[category_name]['mean_vec'],
            frechet_models[category_name]['distances_{}'.format(distance_type)],
            frechet_models[category_name]['frechet_model']]

def compute_openfrechet_prob(scores, scores_u):
    prob_scores, prob_unknowns = [], []
    for s, su in zip(scores, scores_u):
        channel_scores = np.exp(s)
        channel_unknown = np.exp(np.sum(su))
        total_denom = np.sum(channel_scores) + channel_unknown
        prob_scores.append(channel_scores / total_denom) 
        prob_unknowns.append(channel_unknown / total_denom) 
    
    #Take Channel Mean(We only have one)
    scores = np.mean(prob_scores, axis=0)
    unknowns = np.mean(prob_unknowns, axis=0)
    modified_scores = scores.tolist() + [unknowns]
    return modified_scores


def openFrechet(frechet_models,categories,input_score,eu_weight,alpha=7, distance_type='eucos'):
    
    nb_classes = len(categories) 
    ranked_list = input_score.argsort().ravel()[::-1][:alpha] 
    alpha_weights = [((alpha + 1) - i) / float(alpha) for i in range(1, alpha + 1)]
    omega = np.zeros(nb_classes)
    omega[ranked_list] = alpha_weights
    scores, scores_u = [], []
    for channel, input_score_channel in enumerate(input_score):
        score_channel, score_channel_u = [], [] 
        for c, category_name in enumerate(categories):
            mav,dist,frechet_param = query_frechet(category_name,frechet_models,distance_type) 
            channel_distance = cal_distance(input_score,mav[channel],eu_weight, distance_type) 
            wscore = genextreme.cdf(channel_distance,frechet_param[channel][0],frechet_param[channel][1],frechet_param[channel][2])
            modified_score = input_score_channel[c] * (1 - wscore * omega[c])
            score_channel.append(modified_score)
            score_channel_u.append(input_score_channel[c] - modified_score)
            score_channel.append(modified_score)
            score_channel_u.append(input_score_channel[c] - modified_score)
        scores.append(score_channel)
        scores_u.append(score_channel_u)
    
    scores = np.asarray(scores) 
    scores_u = np.asarray(scores_u)

    openmax_prob = np.array(compute_openfrechet_prob(scores, scores_u))
    
    return openmax_prob

def ProcessFrechet(id,device="cuda:0"):
    trainset = RadarDataSet("../Records/TrainmmGait_P11.txt") 
    testsetinv = RadarDataSet("../Records/TestmmGaitIntruder_"+str(id)+".txt")
    trainloader = DataLoader(trainset,batch_size=256,num_workers=4,pin_memory=True)  
    testloaderinv = DataLoader(testsetinv,batch_size=256,num_workers=4,pin_memory=True)
    model = torch.load("mmGait.pth")
    count = 0
    scores,mavs,dists = getMavs(11,trainloader,device=device,net=model)
    testall = 0
    testacc = 0
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    FPc = 0
    categories = [i for i in range(11)]
    frechetmodels = fit_Frechet(mavs,dists,categories,distance_type='cosine')
   
    with torch.no_grad():
        for batchidx,(testdata,testlabel) in enumerate(testloaderinv):
            testdata = testdata.to(device)
            testlabel = testlabel.numpy()
            outputs, hiddens = model.encoder_q(testdata)
            for i in range(outputs.shape[0]):
                input_score = outputs[i].unsqueeze(0).cpu().detach().numpy()
                openmaxprob = openFrechet(frechetmodels,categories,input_score,eu_weight=0.5,distance_type='cosine')
                testall += 1
                if np.max(openmaxprob) > 0.40:
                    predictLable = np.argmax(openmaxprob)
                else:
                    predictLable = 11
                if predictLable == testlabel[i]:
                    testacc += 1
                if predictLable == testlabel[i] and testlabel[i] != 11:
                    TP += 1
                elif predictLable == testlabel[i] and testlabel[i] == 11:
                    TN += 1
                elif predictLable == 11 and testlabel[i] != 11:  
                    FN += 1
                elif testlabel[i] == 11 and predictLable != 11:
                    FP += 1
                elif testlabel[i] != 11 and predictLable != 11 and testlabel[i] != predictLable: #样本已知类,预测已知类,但已知类之间预测错误
                    FPc += 1
                count += 1
    precision = TP / (TP + FP + FPc)
    recall = TP/(TP+FN)
    F1Score = 2 * (precision * recall) / (precision + recall)
    
    with open("FrechetresultmmGait.txt","a+",encoding="utf-8") as f:
            f.write("ID: "+str(id)+"\n")
            f.write("F1-Score: "+str(F1Score)+"\n")

if __name__ == "__main__":
    for id in range(14,23):
        ProcessFrechet(id)