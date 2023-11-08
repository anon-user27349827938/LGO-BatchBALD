import torch
from torch.autograd import grad
from torch import optim
from params import *
import numpy as np


@torch.enable_grad()
def entropy(p, ignore_1 = True):
    if 1 in p and ignore_1:
        return 0*p.sum()
    return -((p>0)*p*torch.log(p)).sum()

@torch.enable_grad()
def batch_entropy_sum(p):
    return -(p*torch.log(p)).sum(dim=1).sum()

@torch.enable_grad()
def get_utility(model, num_samp, x_data, ent_only = False):
    #for single data point
    
    model.train()
    probs_summed = torch.zeros(10, requires_grad=True)
    ent_summed = torch.zeros(1, requires_grad=True)
    
    for i in range(num_samp): 
        probs = model(x_data)
        probs.retain_grad()
        probs_summed = probs_summed.clone().to(device) + probs
        ent = entropy(probs)
        ent.retain_grad()
        ent_summed = ent_summed.clone().to(device) + ent
    
    total_entropy = entropy(probs_summed/num_samp)
    total_entropy.retain_grad()
    if not ent_only:
        conditional_entropy = ent_summed/num_samp
        conditional_entropy.retain_grad()
        utility = total_entropy - conditional_entropy
        utility.retain_grad()
    else:
        utility = total_entropy
        utility.retain_grad()
    return utility

@torch.enable_grad()
def get_utility_z(z, vae, model, ):
    vae.eval()
    img = vae.decode(z)
    img.retain_grad()
    return get_utility(model, num_samp=100, x_data=img)


def sample_config_unif(reps, batch_size, num_classes):
    return torch.randint(low=0, high=num_classes, size=(reps,batch_size)), 1/(num_classes**batch_size)
    
    
@torch.enable_grad()
def get_batch_utility(model, num_samp_m, num_samp_k, x_data):
    '''
    model is the classifier
    num_samp_m is number of configurations (of labels) to use in joint entropy calculation
    num_samp_k is number of monte carlo samples from using different epistemic indices in estimating entropies/probabilities
    '''
    
    # help enable gradient tracking
    model.train()
    
    # batch size
    n = x_data.shape[0]
    
    
    ent_mean = torch.zeros(1, requires_grad=True, device=device) # will end up as mean joint entropy conditional on random model
    ps = torch.zeros(num_samp_m, requires_grad=True, device=device) # i'th entry will be estimate of probability of i'th configuration
    #y_probs = torch.zeros(num_samp_m, requires_grad=True) # importance sampling probabilities for configurations
    ys = [] # configurations
    
    # sample configurations and calculate IS probs
    for i in range (num_samp_m):
        probs = model(x_data)
        ys.append(torch.distributions.Categorical(probs).sample())
        #y_probs = y_probs.clone() + torch.gather(probs,1,ys[i].unsqueeze(dim=1)).prod()*torch.nn.functional.one_hot(torch.tensor(i),num_samp_m)
    
    #ys, y_probs = sample_config_unif(num_samp_m, x_data.shape[0],10)
    

    # for every random model
    for i in range(num_samp_k):
        probs = model(x_data)
        ent = batch_entropy_sum(probs)
        ent_mean = ent_mean.clone() + ent/num_samp_k
        
        #for every configuration y
        for j in range(num_samp_m):
            y = ys[j]
            p = torch.exp(torch.log(torch.gather(probs,1,y.unsqueeze(dim=1))).sum())
            one_hot = torch.nn.functional.one_hot(torch.tensor(j, device=device),num_samp_m).to(device)
            ps = ps.clone() + p/num_samp_k*one_hot
            
    total_entropy = -torch.sum(torch.log(ps))/num_samp_m
    total_entropy.retain_grad()
    conditional_entropy = ent_mean # this is average entropy once you condition on the randomness in the model
    conditional_entropy.retain_grad()
    utility = total_entropy - conditional_entropy
    utility.retain_grad()
    return utility


@torch.enable_grad()
def get_batch_utility_crn(model, num_samp_m, num_samp_k, x_data):
    '''
    model is the classifier
    num_samp_m is number of configurations (of labels) to use in joint entropy calculation
    num_samp_k is number of monte carlo samples from using different epistemic indices in estimating entropies/probabilities
    '''
    
    # help enable gradient tracking
    model.train()
    
    # batch size
    n = x_data.shape[0]
    
    ent_mean = torch.zeros(1, requires_grad=True, device=device) # will end up as mean joint entropy conditional on random model
    ps = torch.zeros(num_samp_m, requires_grad=True, device=device) # i'th entry will be estimate of probability of i'th configuration
    #y_probs = torch.zeros(num_samp_m, requires_grad=True) # importance sampling probabilities for configurations
    ys = [] # configurations
    
    # sample configurations and calculate IS probs
    for i in range (num_samp_m):
        probs = model(x_data)
        ys.append(torch.distributions.Categorical(probs).sample())
        #y_probs = y_probs.clone() + torch.gather(probs,1,ys[i].unsqueeze(dim=1)).prod()*torch.nn.functional.one_hot(torch.tensor(i),num_samp_m)
    
    #ys, y_probs = sample_config_unif(num_samp_m, x_data.shape[0],10)
    
    # for every random model
    for i in range(num_samp_k):
        probs = model(x_data)
        
        #for every configuration y
        for j in range(num_samp_m):
            y = ys[j]
            #p = torch.gather(probs,1,y.unsqueeze(dim=1)).prod()
            p = torch.exp(torch.log(torch.gather(probs,1,y.unsqueeze(dim=1))).sum())
            ent_mean = ent_mean.clone() - torch.sum(torch.log(p))/(num_samp_m*num_samp_k)
            ps = ps.clone() + p/num_samp_k*torch.nn.functional.one_hot(torch.tensor(j),num_samp_m).to(device)
            
    total_entropy = -torch.sum(torch.log(ps))/num_samp_m
    total_entropy.retain_grad()
    conditional_entropy = ent_mean # this is average entropy once you condition on the randomness in the model
    conditional_entropy.retain_grad()
    utility = total_entropy - conditional_entropy
    utility.retain_grad()
    return utility

@torch.enable_grad()
def get_batch_utility_z_crn(z, vae, model, ):
    vae.eval()
    img = vae.decode(z)
    img.retain_grad()
    return get_batch_utility_crn(model, num_samp_m=10, 
                             num_samp_k=10, x_data=img)

@torch.enable_grad()
def get_batch_utility_z(z, vae, model, ):
    vae.eval()
    img = vae.decode(z)
    img.retain_grad()
    return get_batch_utility(model, num_samp_m=100, 
                             num_samp_k=10, x_data=img)

def maximize_func(func, input_tensor, other_params, lr = 0.01, num_steps = 20):
    # Create an optimizer object for Adam
    optimizer = optim.Adam([input_tensor], lr=lr)
    #optimizer = optim.SGD([input_tensor], lr=lr)
    # optimizer = optim.Adam([input_tensor.requires_grad_(True)], lr=0.01)

    # Optimize the input tensor to maximize the function
    for i in range(num_steps):
        optimizer.zero_grad()
        if other_params is None:
            output = func(input_tensor)
        else:
            output = func(input_tensor, *other_params)
        output.retain_grad()
        loss = -output
        loss.retain_grad()
        #print('next')
        #print(-loss)
        loss.backward()
        #print(input_tensor.grad)
        optimizer.step()

    # Return the optimized input tensor
    return input_tensor

def find_z_batch(vae, model, num_samp_m, num_samp_k, z, max_dist, 
                 alpha = 0.0001, epoch=200): # should tune alpha more or use adam
    '''
    model is the classifier
    num_samp_m is number of configurations (of labels) to use in joint entropy calculation
    num_samp_k is number of monte carlo samples from using different epistemic indices in estimating entropies/probabilities
    z is starting latent variables of images in batch
    max_dist is not currently used but can be used to make the z's close to the origin/likely
    alpha is step size
    '''
    # First set the VAE to be evaluation mode
    vae.eval()

    # Make sure z has gradient descent
    # z = torch.autograd.Variable(z, requires_grad=True)

    # Start the optimization steps
    for _ in range(epoch):
        with torch.no_grad():
            img = vae.decode(z)
        img.retain_grad()
        y = get_batch_utility(model,num_samp_m, num_samp_k, img)
        y.retain_grad()
        print('y = ', y.data.cpu().numpy()) # prints score
        g = grad(outputs=y, inputs=z)[0]
        z = z + alpha*g
        z_dist = torch.norm(z,p=2)
        
        # this can be used as constraint
        if z_dist > max_dist:
            z = z*max_dist/z_dist
    return z

def eval_estimators(model, vae, img, num_samp_m = 10, num_samp_k = 10, num_sam = 10):
    vae.eval()
    sam = [get_batch_utility(model, num_samp_m=num_samp_m, num_samp_k=num_samp_k, x_data=img) for _ in range(num_sam)]
    sam = torch.Tensor(sam)
    mean = torch.mean(sam)
    std = torch.std(sam)
    
    sam_crn = [get_batch_utility_crn(model, num_samp_m=num_samp_m, num_samp_k=num_samp_k, x_data=img) for _ in range(num_sam)]
    sam_crn = torch.Tensor(sam_crn)
    mean_crn = torch.mean(sam_crn)
    std_crn = torch.std(sam_crn)
    return mean, std, mean_crn, std_crn
    
def eval_batch(model,vae,img, num_samp_m = 10, num_samp_k = 10, num_sam = 10):
    vae.eval()
    sam_crn = [get_batch_utility_crn(model, num_samp_m=num_samp_m, num_samp_k=num_samp_k, x_data=img) for _ in range(num_sam)]
    sam_crn = torch.Tensor(sam_crn)
    mean_crn = torch.mean(sam_crn)
    std_crn = torch.std(sam_crn)
    return mean_crn, std_crn
    
    
    
def before_after_batch(model, vae, batch_size = 16, lr = 1, num_steps = 100):
    z = torch.randn(batch_size, latent_size, device=device, requires_grad=True)
    before = vae.decode(z)
    found_z = maximize_func(func=get_batch_utility_z_crn, 
                            input_tensor = z,
                            num_steps=num_steps,
                            lr =lr,
                            other_params=(vae,model))
    after = vae.decode(found_z)
    return before, after

def batchBALD(model, vae, batch_size, x_data, num_samp_m = 5, num_samp_k = 5):
    assert batch_size<=len(x_data)
    scores = -np.inf*np.ones(len(x_data))
    indices = []
    for i in range(len(x_data)):
        scores[i] = get_utility(model, num_samp_k, x_data[i])
    indices.append(scores.argmax())
    while len(indices) < batch_size:
        scores = -np.inf*np.ones(len(x_data))
        for i in range(len(x_data)):
            if i not in indices: 
                scores[i] = get_utility(model, num_samp_k, x_data[indices + [i]])
        indices.append(scores.argmax())
    return x_data[[indices]]
        
    
def add_data(x_data, y_data, new_x_data, new_y_data):
    return torch.cat([x_data,new_x_data.view(len(new_x_data),1,28,28)],dim=0), torch.cat([y_data,new_y_data],dim=0)

def test(model,loader):
    correct = 0
    total = 0
    for i, (x_data, y_data) in enumerate(loader):
        x_data = x_data.to(device)
        y_data = y_data.to(device)
        for _ in range(10): # getting multiple samples from ENN
            flag = True
            if flag:
                outputs = model(x_data).to(device)
                flag = False
            else:
                outputs += model(x_data).to(device)
            preds = outputs.argmax(dim=1)
            correct += (preds==y_data).sum()
            total += len(y_data)
    return correct/total
