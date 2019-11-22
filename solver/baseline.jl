def _feature(self, episodes):
    ones = episodes.mask.unsqueeze(2)
    observations = episodes.observations * ones
    cum_sum = torch.cumsum(ones, dim=0) * ones
    al = cum_sum / 100.0

    return torch.cat([observations, observations ** 2,
        al, al ** 2, al ** 3, ones], dim=2)

function gae(episodes)
    #TODO: Figure out normalization
    #TODO: Fit linear model to batch (to estimate V(s))
    #TODO: compute advantages
end