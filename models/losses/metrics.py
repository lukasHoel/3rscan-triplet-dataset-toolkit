import torch
import torch.nn as nn

def l2_dist_sum_weighted(input_features, target_features):
    """
    Calculates l2 distance between input and target features and weights distance by number of feature maps.
    
    :param input_features: list of feature-maps
    :param target_features: list of feature-maps
    :param reduction: see nn.MSELoss --> this controls if we reduce all batches at the end by summing/averaging or not.
    :return: 
    """
    dists = []
    for input, target in zip(input_features, target_features):
        dists.append(nn.MSELoss()(input, target))
    return torch.mean(torch.stack(dists))


def l2_dist_sum_weighted_per_batch(input_features, target_features):
    """
    Calculates l2 distance between input and target features and weights distance by number of feature maps.

    :param input_features: list of feature-maps
    :param target_features: list of feature-maps
    :param reduction: see nn.MSELoss --> this controls if we reduce all batches at the end by summing/averaging or not.
    :return:
    """
    batch_dists = []
    for i in range(input_features[0].shape[0]):
        dists = []
        for input, target in zip(input_features, target_features):
            dists.append(nn.MSELoss()(input[i], target[i]))
        batch_dists.append(torch.mean(torch.stack(dists)))
    return torch.stack(batch_dists)

def cosine_similarity_sum(input_features, target_features):
    """
    Calculates cosine similarity between input and target features along all available feature maps.
    Returns average cosine similarity over all lists of feature-maps.

    :param input_features: list of feature-maps
    :param target_features: list of feature-maps
    :return:
    """
    mean_cos_sim = torch.zeros(1)
    for input, target in zip(input_features, target_features):
        print("input shape ", input.shape)
        if len(input.shape) > 2:
            batch_size = input_features[0].shape[0]
            print("bs ", batch_size)
            input = input.view(batch_size, -1)
            target = target.view(batch_size, -1)
            print("input shape ", input.shape)
        cos_sim = nn.CosineSimilarity(dim=1)(input, target).squeeze()
        mean_cos_sim += cos_sim
    mean_cos_sim /= len(input_features)
    return mean_cos_sim

def top_k_accuracy(pos_dist, neg_dists, k=1):
    if isinstance(neg_dists, list):
        distances = [n for n in neg_dists]
    else:
        distances = [neg_dists]
    distances.append(pos_dist)
    distances.sort() # default sorts ascendingly so smallest dist is at index 0


    print("ACC DISTS: ", distances)
    print("FIRST K DISTS: ", distances[:k])

    # if pos_dist is among the smallest k distances
    if pos_dist in distances[:k]:
        print("ACC TRUE")
        return True
    else:
        print("ACC FALSE")
        return False

class Top_K_Accuracy(nn.Module):
    def __init__(self, k=[1,5]):
        super(Top_K_Accuracy, self).__init__()
        self.k = k

    def forward(self, pos_dist, neg_dists):
        return {"Top-"+str(k): top_k_accuracy(pos_dist, neg_dists, k) for k in self.k}