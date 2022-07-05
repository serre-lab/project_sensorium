import torch
torch.manual_seed(1)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.random.seed(1)
from torch.nn import init
from torch.nn import init
from torch.autograd import Function

from torchvision import datasets, models, transforms

# Seeds
import random
random.seed(1)

class L1(torch.nn.Module):
    def __init__(self, module, weight_decay):
        super().__init__()
        self.module = module
        self.weight_decay = weight_decay

        # Backward hook is registered on the specified module
        self.hook = self.module.register_backward_hook(self._weight_decay_hook)

    # Not dependent on backprop incoming values, placeholder
    def _weight_decay_hook(self, *_):
        for param in self.module.parameters():
            # If there is no gradient or it was zeroed out
            # Zeroed out using optimizer.zero_grad() usually
            # Turn on if needed with grad accumulation/more safer way
            # if param.grad is None or torch.all(param.grad == 0.0):

            # Apply regularization on it
            param.grad = self.regularize(param)

    def regularize(self, parameter):
        # L1 regularization formula
        return self.weight_decay * torch.sign(parameter.data)

    def forward(self, *args, **kwargs):
        # Simply forward and args and kwargs to module
        return self.module(*args, **kwargs)


class dummyhgru(Function):
    @staticmethod
    def forward(ctx, state_2nd_last, last_state, *args):
        ctx.save_for_backward(state_2nd_last, last_state)
        ctx.args = args
        return last_state

    @staticmethod
    def backward(ctx, grad):
        neumann_g = neumann_v = None
        neumann_g_prev = grad.clone()
        neumann_v_prev = grad.clone()

        state_2nd_last, last_state = ctx.saved_tensors
        
        args = ctx.args
        truncate_iter = args[-1]
        # exp_name = args[-2]
        # i = args[-3]
        # epoch = args[-4]

        normsv = []
        normsg = []
        normg = torch.norm(neumann_g_prev)
        # normsg.append(normg.data.item())
        # normsv.append(normg.data.item())
        prev_normv = 1e8
        for ii in range(truncate_iter):
            neumann_v = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=neumann_v_prev,
                                            retain_graph=True, allow_unused=True)
            normv = torch.norm(neumann_v[0])
            neumann_g = neumann_g_prev + neumann_v[0]
            normg = torch.norm(neumann_g)
            
            if normg > 1 or normv > prev_normv or normv < 1e-9:
                # normsg.append(normg.data.item())
                # normsv.append(normv.data.item())
                neumann_g = neumann_g_prev
                break

            prev_normv = normv
            neumann_v_prev = neumann_v
            neumann_g_prev = neumann_g
            
            # normsv.append(normv.data.item())
            # normsg.append(normg.data.item())
        return (None, neumann_g, None, None, None, None)

def noneg_sigmoid(input_):
    return (F.sigmoid(input_) - 0.5)*2

class hConvGRUCell(nn.Module):
    """
    Generate a recurrent cell
    """

    def __init__(self, hidden_size, kernel_size, timesteps, batchnorm_bool=True, grad_method='bptt', use_attention=False, \
                no_inh=False, lesion_alpha=False, lesion_gamma=False, lesion_mu=False, lesion_kappa=False, \
                noneg_constraint = False, exp_weight = False, orthogonal_init = True):
        super(hConvGRUCell, self).__init__()
        self.padding = kernel_size // 2
        self.hidden_size = hidden_size
        self.batchnorm_bool = batchnorm_bool
        self.timesteps = timesteps
        self.use_attention = use_attention
        self.no_inh = no_inh
        self.noneg_constraint = noneg_constraint
        self.exp_weight = exp_weight
        self.orthogonal_init = orthogonal_init
        
        if self.exp_weight:
            print('exp_weight exp_weight exp_weight')
            # lambda_e = ((hidden_size*(2*np.pi - 1))**(0.5))/((2*np.pi)**(0.5))
        
        if self.use_attention:
            self.a_w_gate = nn.Conv2d(hidden_size, hidden_size, 1, padding=1 // 2)
            self.a_u_gate = nn.Conv2d(hidden_size, hidden_size, 1, padding=1 // 2)
            if self.orthogonal_init:
                init.orthogonal_(self.a_w_gate.weight)
                init.orthogonal_(self.a_u_gate.weight)
            elif self.exp_weight:
                lambda_e = ((hidden_size*hidden_size*1*1*(2*np.pi - 1))**(0.5))/((2*np.pi)**(0.5))
                # lambda_e = lambda_e**(0.5)
                torch_dist = torch.distributions.exponential.Exponential((torch.tensor([lambda_e])))
                sampled_distribution_a_w_gate = torch_dist.rsample(sample_shape=torch.Size([hidden_size, hidden_size, 1, 1]))
                sampled_distribution_a_w_gate = sampled_distribution_a_w_gate.reshape(hidden_size, hidden_size, 1, 1)
                self.a_w_gate.weight.data = nn.Parameter(sampled_distribution_a_w_gate)
                #
                lambda_e = ((hidden_size*hidden_size*1*1*(2*np.pi - 1))**(0.5))/((2*np.pi)**(0.5))
                # lambda_e = lambda_e**(0.5)
                torch_dist = torch.distributions.exponential.Exponential((torch.tensor([lambda_e])))
                sampled_distribution_a_u_gate = torch_dist.rsample(sample_shape=torch.Size([hidden_size, hidden_size, 1, 1]))
                sampled_distribution_a_u_gate = sampled_distribution_a_u_gate.reshape(hidden_size, hidden_size, 1, 1)
                self.a_u_gate.weight.data = nn.Parameter(sampled_distribution_a_u_gate)
            else:
                init.xavier_normal_(self.a_w_gate.weight)
                init.xavier_normal_(self.a_u_gate.weight)
            init.constant_(self.a_w_gate.bias, 1.)
            init.constant_(self.a_u_gate.bias, 1.)

        self.i_w_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.i_u_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        
        self.e_w_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.e_u_gate = nn.Conv2d(hidden_size, hidden_size, 1)

        spatial_h_size = kernel_size
        self.h_padding = spatial_h_size // 2
        self.w_exc = nn.Parameter(torch.empty(hidden_size, hidden_size, spatial_h_size, spatial_h_size))
        if self.orthogonal_init:
            init.orthogonal_(self.w_exc)
        elif self.exp_weight:
            lambda_e = ((hidden_size*hidden_size*spatial_h_size*spatial_h_size*(2*np.pi - 1))**(0.5))/((2*np.pi)**(0.5))
            lambda_e = lambda_e**(0.75)
            torch_dist = torch.distributions.exponential.Exponential((torch.tensor([lambda_e])))
            sampled_distribution_w_exc = torch_dist.rsample(sample_shape=torch.Size([hidden_size, hidden_size, spatial_h_size, spatial_h_size]))
            sampled_distribution_w_exc = sampled_distribution_w_exc.reshape(hidden_size, hidden_size, spatial_h_size, spatial_h_size)
            self.w_exc = nn.Parameter(sampled_distribution_w_exc)
        else:
            init.xavier_normal_(self.w_exc)

        if not no_inh:
            self.w_inh = nn.Parameter(torch.empty(hidden_size, hidden_size, spatial_h_size, spatial_h_size))
            if self.orthogonal_init:
                init.orthogonal_(self.w_inh)
            elif self.exp_weight:
                lambda_e = ((hidden_size*hidden_size*spatial_h_size*spatial_h_size*(2*np.pi - 1))**(0.5))/((2*np.pi)**(0.5))
                lambda_e = lambda_e**(0.5)
                torch_dist = torch.distributions.exponential.Exponential((torch.tensor([lambda_e])))
                sampled_distribution_w_inh = torch_dist.rsample(sample_shape=torch.Size([hidden_size, hidden_size, spatial_h_size, spatial_h_size]))
                sampled_distribution_w_inh = sampled_distribution_w_inh.reshape(hidden_size, hidden_size, spatial_h_size, spatial_h_size)
                self.w_inh = nn.Parameter(sampled_distribution_w_inh)
            else:
                init.xavier_normal_(self.w_inh)
        
        self.alpha = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.mu = nn.Parameter(torch.empty((hidden_size, 1, 1)))

        self.gamma = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.kappa = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.w = nn.Parameter(torch.empty((hidden_size, 1, 1)))

        if self.batchnorm_bool:
            # self.bn = nn.ModuleList([nn.BatchNorm2d(hidden_size, eps=1e-03) for i in range(self.timesteps*2)])
            self.bn = nn.ModuleList([nn.BatchNorm2d(hidden_size, eps=1e-03) for i in range(2)])
            # self.bn = nn.ModuleList([nn.BatchNorm2d(hidden_size, eps=1e-03, track_running_stats = False) for i in range(2)])
            for bn in self.bn:
                init.constant_(bn.weight, 0.1)

        if self.orthogonal_init:
            init.orthogonal_(self.i_w_gate.weight)
            init.orthogonal_(self.i_u_gate.weight)
            init.orthogonal_(self.e_w_gate.weight)
            init.orthogonal_(self.e_u_gate.weight)
        elif self.exp_weight:
            lambda_e = ((hidden_size*hidden_size*1*1*(2*np.pi - 1))**(0.5))/((2*np.pi)**(0.5))
            lambda_e = lambda_e**(0.5)
            torch_dist = torch.distributions.exponential.Exponential((torch.tensor([lambda_e])))
            sampled_distribution_i_w_gate = torch_dist.rsample(sample_shape=torch.Size([hidden_size, hidden_size, 1, 1]))
            sampled_distribution_i_w_gate = sampled_distribution_i_w_gate.reshape(hidden_size, hidden_size, 1, 1)
            self.i_w_gate.weight.data = nn.Parameter(sampled_distribution_i_w_gate)
            #
            lambda_e = ((hidden_size*hidden_size*1*1*(2*np.pi - 1))**(0.5))/((2*np.pi)**(0.5))
            lambda_e = lambda_e**(0.5)
            torch_dist = torch.distributions.exponential.Exponential((torch.tensor([lambda_e])))
            sampled_distribution_i_u_gate = torch_dist.rsample(sample_shape=torch.Size([hidden_size, hidden_size, 1, 1]))
            sampled_distribution_i_u_gate = sampled_distribution_i_u_gate.reshape(hidden_size, hidden_size, 1, 1)
            self.i_u_gate.weight.data = nn.Parameter(sampled_distribution_i_u_gate)
            #
            lambda_e = ((hidden_size*hidden_size*1*1*(2*np.pi - 1))**(0.5))/((2*np.pi)**(0.5))
            lambda_e = lambda_e**(0.75)
            torch_dist = torch.distributions.exponential.Exponential((torch.tensor([lambda_e])))
            sampled_distribution_e_w_gate = torch_dist.rsample(sample_shape=torch.Size([hidden_size, hidden_size, 1, 1]))
            sampled_distribution_e_w_gate = sampled_distribution_e_w_gate.reshape(hidden_size, hidden_size, 1, 1)
            self.e_w_gate.weight.data = nn.Parameter(sampled_distribution_e_w_gate)
            #
            lambda_e = ((hidden_size*hidden_size*1*1*(2*np.pi - 1))**(0.5))/((2*np.pi)**(0.5))
            lambda_e = lambda_e**(0.75)
            torch_dist = torch.distributions.exponential.Exponential((torch.tensor([lambda_e])))
            sampled_distribution_e_u_gate = torch_dist.rsample(sample_shape=torch.Size([hidden_size, hidden_size, 1, 1]))
            sampled_distribution_e_u_gate = sampled_distribution_e_u_gate.reshape(hidden_size, hidden_size, 1, 1)
            self.e_u_gate.weight.data = nn.Parameter(sampled_distribution_e_u_gate)
        else:
            init.xavier_normal_(self.i_w_gate.weight)
            init.xavier_normal_(self.i_u_gate.weight)
            init.xavier_normal_(self.e_w_gate.weight)
            init.xavier_normal_(self.e_u_gate.weight)

        if not no_inh:
            init.constant_(self.alpha, 1.)
            init.constant_(self.mu, 0.)
        # init.constant_(self.alpha, 0.1)
        # init.constant_(self.mu, 1)
        init.constant_(self.gamma, 0.)
        # init.constant_(self.w, 1.)
        init.constant_(self.kappa, 1.)

        if self.use_attention:
            self.i_w_gate.bias.data = -self.a_w_gate.bias.data
            self.e_w_gate.bias.data = -self.a_w_gate.bias.data
            self.i_u_gate.bias.data = -self.a_u_gate.bias.data
            self.e_u_gate.bias.data = -self.a_u_gate.bias.data
        else:
            init.uniform_(self.i_w_gate.bias.data, 1, self.timesteps - 1 if self.timesteps > 1 else self.timesteps)
            self.i_w_gate.bias.data.log()
            self.i_u_gate.bias.data.log()
            self.e_w_gate.bias.data = -self.i_w_gate.bias.data
            self.e_u_gate.bias.data = -self.i_u_gate.bias.data
        if lesion_alpha:
            self.alpha.requires_grad = False
            self.alpha.weight = 0.
        if lesion_mu:
            self.mu.requires_grad = False
            self.mu.weight = 0.
        if lesion_gamma:
            self.gamma.requires_grad = False
            self.gamma.weight = 0.
        if lesion_kappa:
            self.kappa.requires_grad = False
            self.kappa.weight = 0.



        # ##################################################################
        # ############## Checking the initialization of weights ##############
        # ##################################################################
        # weights_to_check = {'w_exc' : self.w_exc.clone(), 'w_inh' : self.w_inh.clone(), 'kappa' : self.kappa.clone(), 'gamma' : self.gamma.clone(), \
        #                     'mu' : self.mu.clone(), 'alpha' : self.alpha.clone(), 'i_w_gate' : self.i_w_gate.weight.clone(), 'i_u_gate' : self.i_u_gate.weight.clone(), \
        #                     'e_w_gate' : self.e_w_gate.weight.clone(), 'e_u_gate' : self.e_u_gate.weight.clone()}

        # for weight_ in weights_to_check:
        #     temp = weights_to_check[weight_].reshape(-1)
        #     print('Mean weight for : ', weight_, ' :: ', torch.mean(temp))
        #     print('Max weight for : ', weight_, ' :: ', torch.max(temp))
        #     print('Min weight for : ', weight_, ' :: ', torch.min(temp))
        #     temp[temp>0] = 0
        #     temp[temp<0] = 1
        #     print('Number of  negative weights for : ', weight_, ' :: ', torch.sum(temp))

    def forward(self, t_i, input_, inhibition, excitation,  activ=F.softplus, testmode=False):  # Worked with tanh and softplus

        if self.noneg_constraint and not(self.batchnorm_bool):
                # print('new_sigmoid')
                activ = noneg_sigmoid

        # Attention gate: filter input_ and excitation
        if self.use_attention:
            print('Nooooooooooooooooo')
            att_gate = torch.sigmoid(self.a_w_gate(input_) + self.a_u_gate(excitation))  # Attention Spotlight -- MOST RECENT WORKING

        # Gate E/I with attention immediately
        if self.use_attention:
            print('Nooooooooooooooooo')
            gated_input = input_  # * att_gate  # In activ range
            gated_excitation = att_gate * excitation  # att_gate * excitation
        else:
            gated_input = input_
            gated_excitation = excitation
        gated_inhibition = inhibition

        if not self.no_inh:
            # Compute inhibition
            if self.batchnorm_bool:
                inh_intx = self.bn[0](F.conv2d(gated_excitation, self.w_inh, padding=self.h_padding))  # in activ range
            else:
                inh_intx = F.conv2d(gated_excitation, self.w_inh, padding=self.h_padding)  # in activ range
            inhibition_hat = activ(input_ - activ(inh_intx * (self.alpha * gated_inhibition + self.mu)))

            # Integrate inhibition
            if self.noneg_constraint:
                inh_gate = activ(self.i_w_gate(gated_input) + self.i_u_gate(gated_inhibition))
            else:
                inh_gate = torch.sigmoid(self.i_w_gate(gated_input) + self.i_u_gate(gated_inhibition))
            inhibition = (1 - inh_gate) * inhibition + inh_gate * inhibition_hat  # In activ range
        else:
            inhibition, gated_inhibition = gated_excitation, excitation

        # Pass to excitatory neurons
        if self.noneg_constraint:
            exc_gate = activ(self.e_w_gate(gated_inhibition) + self.e_u_gate(gated_excitation))
        else:
            exc_gate = torch.sigmoid(self.e_w_gate(gated_inhibition) + self.e_u_gate(gated_excitation))
        if self.batchnorm_bool:
            exc_intx = self.bn[1](F.conv2d(inhibition, self.w_exc, padding=self.h_padding))  # In activ range
        else:
            exc_intx = F.conv2d(inhibition, self.w_exc, padding=self.h_padding)  # In activ range
        # excitation_hat = activ(exc_intx * (self.kappa * inhibition + self.gamma))  # Skip connection OR add OR add by self-sim
        excitation_hat = activ(inhibition + exc_intx * (self.kappa * inhibition + self.gamma))  # Skip connection OR add OR add by self-sim

        excitation = (1 - exc_gate) * excitation + exc_gate * excitation_hat

        
        ##################################################################
        ################## Noting the weights to check ###################
        ##################################################################
        weights_to_check = {'w_exc' : self.w_exc.clone(), 'w_inh' : self.w_inh.clone(), 'kappa' : self.kappa.clone(), 'gamma' : self.gamma.clone(), \
                            'mu' : self.mu.clone(), 'alpha' : self.alpha.clone(), 'i_w_gate' : self.i_w_gate.weight.clone(), 'i_u_gate' : self.i_u_gate.weight.clone(), \
                            'e_w_gate' : self.e_w_gate.weight.clone(), 'e_u_gate' : self.e_u_gate.weight.clone()}
        
        if testmode:
            return inhibition, excitation, att_gate, weights_to_check
        else:
            return inhibition, excitation, weights_to_check

        # if self.noneg_constraint and not(self.batchnorm_bool):
        #         # print('new_sigmoid')
        #         activ = noneg_sigmoid

        # # Attention gate: filter input_ and excitation
        # if self.use_attention:
        #     print('Nooooooooooooooooo')
        #     att_gate = torch.sigmoid(self.a_w_gate(input_) + self.a_u_gate(excitation))  # Attention Spotlight -- MOST RECENT WORKING

        # # Gate E/I with attention immediately
        # if self.use_attention:
        #     print('Nooooooooooooooooo')
        #     gated_input = input_  # * att_gate  # In activ range
        #     gated_excitation = att_gate * excitation  # att_gate * excitation
        # else:
        #     gated_input = input_
        #     gated_excitation = excitation
        # gated_inhibition = inhibition

        # if not self.no_inh:
        #     # Compute inhibition
        #     if self.batchnorm_bool:
        #         inh_intx = self.bn[t_i*2 + 0](F.conv2d(gated_excitation, self.w_inh, padding=self.h_padding))  # in activ range
        #     else:
        #         inh_intx = F.conv2d(gated_excitation, self.w_inh, padding=self.h_padding)  # in activ range
        #     inhibition_hat = activ(input_ - activ(inh_intx * (self.alpha * gated_inhibition + self.mu)))

        #     # Integrate inhibition
        #     if self.noneg_constraint:
        #         inh_gate = activ(self.i_w_gate(gated_input) + self.i_u_gate(gated_inhibition))
        #     else:
        #         inh_gate = torch.sigmoid(self.i_w_gate(gated_input) + self.i_u_gate(gated_inhibition))
        #     inhibition = (1 - inh_gate) * inhibition + inh_gate * inhibition_hat  # In activ range
        # else:
        #     inhibition, gated_inhibition = gated_excitation, excitation

        # # Pass to excitatory neurons
        # if self.noneg_constraint:
        #     exc_gate = activ(self.e_w_gate(gated_inhibition) + self.e_u_gate(gated_excitation))
        # else:
        #     exc_gate = torch.sigmoid(self.e_w_gate(gated_inhibition) + self.e_u_gate(gated_excitation))
        # if self.batchnorm_bool:
        #     exc_intx = self.bn[t_i*2 + 1](F.conv2d(inhibition, self.w_exc, padding=self.h_padding))  # In activ range
        # else:
        #     exc_intx = F.conv2d(inhibition, self.w_exc, padding=self.h_padding)  # In activ range
        # # excitation_hat = activ(exc_intx * (self.kappa * inhibition + self.gamma))  # Skip connection OR add OR add by self-sim
        # excitation_hat = activ(inhibition + exc_intx * (self.kappa * inhibition + self.gamma))  # Skip connection OR add OR add by self-sim

        # excitation = (1 - exc_gate) * excitation + exc_gate * excitation_hat

        
        # ##################################################################
        # ################## Noting the weights to check ###################
        # ##################################################################
        # weights_to_check = {'w_exc' : self.w_exc.clone(), 'w_inh' : self.w_inh.clone(), 'kappa' : self.kappa.clone(), 'gamma' : self.gamma.clone(), \
        #                     'mu' : self.mu.clone(), 'alpha' : self.alpha.clone(), 'i_w_gate' : self.i_w_gate.weight.clone(), 'i_u_gate' : self.i_u_gate.weight.clone(), \
        #                     'e_w_gate' : self.e_w_gate.weight.clone(), 'e_u_gate' : self.e_u_gate.weight.clone()}
        
        # if testmode:
        #     return inhibition, excitation, att_gate, weights_to_check
        # else:
        #     return inhibition, excitation, weights_to_check



class FFhGRU(nn.Module):

    def __init__(self, dimensions = 25, input_size=3, timesteps=8, kernel_size=15, jacobian_penalty=False, grad_method='bptt', no_inh=False, \
                 lesion_alpha=False, lesion_mu=False, lesion_gamma=False, lesion_kappa=False, nl=F.softplus, l1=0., output_size=3, \
                 num_rbp_steps=10, LCP=0., jv_penalty_weight=0.002, pre_kernel_size = 7, VGG_bool = True, InT_bool = True, \
                 batchnorm_bool = True, noneg_constraint = False, exp_weight = False, orthogonal_init = True, freeze_VGG = False):
        '''
        '''
        super(FFhGRU, self).__init__()
        self.timesteps = timesteps
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.hgru_size = dimensions
        self.output_size = output_size
        self.num_rbp_steps = num_rbp_steps
        self.InT_bool = InT_bool
        self.batchnorm_bool = batchnorm_bool
        self.noneg_constraint = noneg_constraint
        self.exp_weight = exp_weight
        self.orthogonal_init = orthogonal_init
        self.freeze_VGG = freeze_VGG
        self.LCP = LCP
        self.jv_penalty_weight = jv_penalty_weight
        if l1 > 0:
            self.preproc = L1(nn.Conv2d(input_size, dimensions, kernel_size=kernel_size, stride=1, padding=kernel_size // 2), weight_decay=l1)
        elif not(VGG_bool):
            self.preproc = nn.Conv2d(input_size, dimensions, kernel_size=pre_kernel_size, stride=1, padding=pre_kernel_size // 2)
        else:
            vgg_pretrained_features = models.vgg16(pretrained=True).features
            # vgg_pretrained_features = models.squeezenet1_1(pretrained=True).features

            # vgg_pretrained_features
            # print(vgg_pretrained_features)
            preproc = torch.nn.Sequential()
            for x in range(10): # VGG-16
            # for x in range(6): # SQZN
                print(vgg_pretrained_features[x])
                # if x in [4,9]:
                #     max_pool_3_1 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=0, dilation=1, ceil_mode=False)
                #     self.preproc.add_module(str(x), max_pool_3_1)
                # else:
                preproc.add_module(str(x), vgg_pretrained_features[x])

            if self.freeze_VGG:
                ##################################################################
                #################### Freezing Weights ############################
                ##################################################################
                preproc.eval()
                # freeze params
                for param in preproc.parameters():
                    param.requires_grad = False
                
            self.preproc = preproc

        # init.xavier_normal_(self.preproc.weight)
        # init.constant_(self.preproc.bias, 0)


        if self.InT_bool:
            '''
            self, hidden_size, kernel_size, timesteps, batchnorm=True, grad_method='bptt', use_attention=False, \
            no_inh=False, lesion_alpha=False, lesion_gamma=False, lesion_mu=False, lesion_kappa=False
            '''
            self.unit1 = hConvGRUCell(
                # input_size=input_size,
                hidden_size=self.hgru_size,
                kernel_size=kernel_size,
                use_attention=False,
                no_inh=no_inh,
                batchnorm_bool = self.batchnorm_bool,
                noneg_constraint = self.noneg_constraint,
                exp_weight = self.exp_weight,
                orthogonal_init = self.orthogonal_init,
                # l1=l1,
                lesion_alpha=lesion_alpha,
                lesion_mu=lesion_mu,
                lesion_gamma=lesion_gamma,
                lesion_kappa=lesion_kappa,
                timesteps=timesteps)

        else:
            self.unit1 = hConvGRUCell_org(
                input_size=input_size, 
                hidden_size=self.hgru_size, 
                kernel_size=kernel_size, 
                batchnorm=True, 
                timesteps=timesteps)

        self.dropout = nn.Dropout(p=0.4)

        self.nl = nl


    def forward(self, x, testmode=False, give_timesteps = False):
        # First step: replicate x over the channel dim self.hgru_size times
        xbn = self.preproc(x)
        # Akash changed from batchnorm to batchnorm_bool
        if self.batchnorm_bool:
            xbn = self.dropout(xbn)
        # print('input map : ', xbn.shape)
        # xbn = self.bn(xbn)  # This might be hurting me...
        xbn = self.nl(xbn)  # TEST TO SEE IF THE NL STABLIZES


        # Now run RNN
        x_shape = xbn.shape
        if self.InT_bool:
            excitation = torch.zeros((x_shape[0], x_shape[1], x_shape[2], x_shape[3]), requires_grad=False).to(x.device)
            inhibition = torch.zeros((x_shape[0], x_shape[1], x_shape[2], x_shape[3]), requires_grad=False).to(x.device)
        else:
            excitation = None
            inhibition = None

        # Loop over frames
        states = []
        gates = []

        time_steps_exc = []
        time_steps_inh = []

        if self.grad_method == "bptt":
            for t in range(self.timesteps):
                if self.InT_bool:
                    out = self.unit1(
                        input_=xbn,
                        inhibition=inhibition,
                        excitation=excitation,
                        activ=self.nl,
                        testmode=testmode,
                        t_i = t
                        )
                else:
                    out = self.unit1(
                        input_=xbn,
                        prev_state2=excitation,
                        timestep = t
                        )
                if testmode:
                    inhibition, excitation, gate, weights_to_check = out 
                    time_steps_exc.append(excitation)
                    time_steps_inh.append(inhibition)
                    gates.append(gate)  # This should learn to keep the winner
                    states.append(self.readout_conv(excitation))  # This should learn to keep the winner
                else:
                    inhibition, excitation, weights_to_check = out 
                    time_steps_exc.append(excitation)
                    time_steps_inh.append(inhibition)
        elif self.grad_method == "rbp":
            with torch.no_grad():
               for t in range(self.timesteps - 1):
                    out = self.unit1(
                        input_=xbn,
                        inhibition=inhibition,
                        excitation=excitation,
                        activ=self.nl,
                        testmode=testmode)
                    if testmode:
                        inhibition, excitation, gate = out
                        gates.append(gate)  # This should learn to keep the winner
                        states.append(self.readout_conv(excitation))  # This should learn to keep the winner
                    else:
                        inhibition, excitation = out
            pen_exc = excitation.detach().requires_grad_()
            last_inh, last_exc = self.unit1(xbn, inhibition=inhibition, excitation=pen_exc, activ=self.nl, testmode=testmode)
            import pdb;pdb.set_trace()
            # Need to attach exc with inh to propoagate grads
            excitation = dummyhgru.apply(pen_exc, last_exc, self.num_rbp_steps)
        else:
            raise NotImplementedError(self.grad_method)

        

        return excitation #, time_steps_exc, time_steps_inh, xbn, weights_to_check


            

################################################################################################################################
################################################################################################################################
################################################################################################################################

#torch.manual_seed(42)

class hConvGRUCell_org(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size, batchnorm=True, timesteps=8):
        super().__init__()
        self.padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        
        self.u1_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.u2_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        
        self.w_gate_inh = nn.Parameter(torch.empty(hidden_size , hidden_size , kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(torch.empty(hidden_size , hidden_size , kernel_size, kernel_size))
        
        self.alpha = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.gamma = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.kappa = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.w = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.mu= nn.Parameter(torch.empty((hidden_size,1,1)))

        if self.batchnorm:
            #self.bn = nn.ModuleList([nn.GroupNorm(25, 25, eps=1e-03) for i in range(32)])
            # self.bn = nn.ModuleList([nn.BatchNorm2d(self.hidden_size, eps=1e-03) for i in range(self.timesteps*4)])
            self.bn = nn.ModuleList([nn.BatchNorm2d(self.hidden_size, eps=1e-03) for i in range(4)])

            for bn in self.bn:
                init.constant_(bn.weight, 0.1)
        else:
            self.n = nn.Parameter(torch.randn(self.timesteps,1,1))

        init.orthogonal_(self.w_gate_inh)
        init.orthogonal_(self.w_gate_exc)
        
#        self.w_gate_inh = nn.Parameter(self.w_gate_inh.reshape(hidden_size , hidden_size , kernel_size, kernel_size))
#        self.w_gate_exc = nn.Parameter(self.w_gate_exc.reshape(hidden_size , hidden_size , kernel_size, kernel_size))
        self.w_gate_inh.register_hook(lambda grad: (grad + torch.transpose(grad,1,0))*0.5)
        self.w_gate_exc.register_hook(lambda grad: (grad + torch.transpose(grad,1,0))*0.5)
#        self.w_gate_inh.register_hook(lambda grad: print("inh"))
#        self.w_gate_exc.register_hook(lambda grad: print("exc"))
        
        init.orthogonal_(self.u1_gate.weight)
        init.orthogonal_(self.u2_gate.weight)
        
        
        init.constant_(self.alpha, 0.1)
        init.constant_(self.gamma, 1.0)
        init.constant_(self.kappa, 0.5)
        init.constant_(self.w, 0.5)
        init.constant_(self.mu, 1)
        
        init.uniform_(self.u1_gate.bias.data, 1, 8.0 - 1)
        self.u1_gate.bias.data.log()
        self.u2_gate.bias.data =  -self.u1_gate.bias.data


    def forward(self, input_, prev_state2, timestep=0):

        if timestep == 0:
            prev_state2 = torch.empty_like(input_)
            init.xavier_normal_(prev_state2)

        #import pdb; pdb.set_trace()
        i = timestep
        if self.batchnorm:
            # g1_t = torch.sigmoid(self.bn[i*4+0](self.u1_gate(prev_state2)))
            # c1_t = self.bn[i*4+1](F.conv2d(prev_state2 * g1_t, self.w_gate_inh, padding=self.padding))
            
            # next_state1 = F.relu(input_ - F.relu(c1_t*(self.alpha*prev_state2 + self.mu)))
            # #next_state1 = F.relu(input_ - c1_t*(self.alpha*prev_state2 + self.mu))
            
            # g2_t = torch.sigmoid(self.bn[i*4+2](self.u2_gate(next_state1)))
            # c2_t = self.bn[i*4+3](F.conv2d(next_state1, self.w_gate_exc, padding=self.padding))
            
            # h2_t = F.relu(self.kappa*next_state1 + self.gamma*c2_t + self.w*next_state1*c2_t)
            # #h2_t = F.relu(self.kappa*next_state1 + self.kappa*self.gamma*c2_t + self.w*next_state1*self.gamma*c2_t)
            
            # prev_state2 = (1 - g2_t)*prev_state2 + g2_t*h2_t

            g1_t = torch.sigmoid(self.bn[0](self.u1_gate(prev_state2)))
            c1_t = self.bn[1](F.conv2d(prev_state2 * g1_t, self.w_gate_inh, padding=self.padding))
            
            next_state1 = F.relu(input_ - F.relu(c1_t*(self.alpha*prev_state2 + self.mu)))
            #next_state1 = F.relu(input_ - c1_t*(self.alpha*prev_state2 + self.mu))
            
            g2_t = torch.sigmoid(self.bn[2](self.u2_gate(next_state1)))
            c2_t = self.bn[3](F.conv2d(next_state1, self.w_gate_exc, padding=self.padding))
            
            h2_t = F.relu(self.kappa*next_state1 + self.gamma*c2_t + self.w*next_state1*c2_t)
            #h2_t = F.relu(self.kappa*next_state1 + self.kappa*self.gamma*c2_t + self.w*next_state1*self.gamma*c2_t)
            
            prev_state2 = (1 - g2_t)*prev_state2 + g2_t*h2_t

        else:
            g1_t = F.sigmoid(self.u1_gate(prev_state2))
            c1_t = F.conv2d(prev_state2 * g1_t, self.w_gate_inh, padding=self.padding)
            next_state1 = F.tanh(input_ - c1_t*(self.alpha*prev_state2 + self.mu))
            g2_t = F.sigmoid(self.bn[i*4+2](self.u2_gate(next_state1)))
            c2_t = F.conv2d(next_state1, self.w_gate_exc, padding=self.padding)
            h2_t = F.tanh(self.kappa*(next_state1 + self.gamma*c2_t) + (self.w*(next_state1*(self.gamma*c2_t))))
            prev_state2 = self.n[timestep]*((1 - g2_t)*prev_state2 + g2_t*h2_t)

        return next_state1, prev_state2


class hConvGRU(nn.Module):

    def __init__(self, hidden_size = 25, timesteps=8, filt_size = 9):
        super().__init__()
        self.timesteps = timesteps
        
        self.conv0 = nn.Conv2d(1, hidden_size = 25, kernel_size=7, padding=3)
        part1 = np.load("/gpfs/data/tserre/aarjun1/gabor_filters/gabor_serre.npy")
        self.conv0.weight.data = torch.FloatTensor(part1)
        
        self.unit1 = hConvGRUCell_org(hidden_size, hidden_size, filt_size)
        print("Training with filter size:",filt_size,"x",filt_size)
        self.unit1.train()
        
        #self.bn = nn.GroupNorm(25, 25, eps=1e-03)
        self.bn = nn.BatchNorm2d(hidden_size, eps=1e-03)
        
        self.conv6 = nn.Conv2d(hidden_size, 2, kernel_size=1)
        init.xavier_normal_(self.conv6.weight)
        init.constant_(self.conv6.bias, 0)
        
        self.maxpool = nn.MaxPool2d(150, stride=1)
        
        #self.bn2 = nn.GroupNorm(2, 2, eps=1e-03)
        self.bn2 = nn.BatchNorm2d(2, eps=1e-03)
        
        self.fc = nn.Linear(2, 2)
        init.xavier_normal_(self.fc.weight)
        init.constant_(self.fc.bias, 0)

    def forward(self, x):
        internal_state = None
        #import pdb; pdb.set_trace()
        #print(x.shape)
        x = self.conv0(x)
        # x = torch.pow(x, 2)
        
        for i in range(self.timesteps):
            internal_state  = self.unit1(x, internal_state, timestep=i)
        #import pdb; pdb.set_trace()
        output = self.bn(internal_state)
        output = F.leaky_relu(self.conv6(output))
        output = self.maxpool(output)
        output = self.bn2(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

            

