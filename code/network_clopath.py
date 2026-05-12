from eqn_clopath import *
from brian2 import *
import numpy as np

def build_network_train():
    # Input Layer
    inp_group = PoissonGroup(n_input, rates='input_rates(t - batch_start_time, i)', name='inp')

    # Excitatory Layer
    exc_group = NeuronGroup(n_e, eqs_e, threshold='u > (VT + theta_hom)', 
                            reset=reset_exc, method='euler', name='exc')
    exc_group.u = EL
    exc_group.dubar_minus = EL
    exc_group.dubar_plus = EL
    
    # Inhibitory Layer
    inh_group = NeuronGroup(n_i, eqs_i, threshold='v > I_thresh', 
                            reset='v = I_reset', refractory=I_refrac, 
                            method='euler', name='inh')
    inh_group.v = I_rest
    
    # Input -> Excitatory (VDSP Plasticity)
    S_inp_exc = Synapses(inp_group, exc_group, model=eqs_clopath,
                         on_pre=on_pre_clopath, name='s_inp_exc')
	
    S_inp_exc.connect(p=1.0)
    S_inp_exc.w = 'rand() * wmax'
    S_inp_exc.delay = 'rand() * max_delay'

    # Excitatory -> Inhibitory (1-to-1)
    S_exc_inh = Synapses(exc_group, inh_group, on_pre='ge_post += w_ei', name='s_exc_inh')
    S_exc_inh.connect(j='i')

    # Inhibitory -> Excitatory (Winner-Take-All lateral inhibition)
    S_inh_exc = Synapses(inh_group, exc_group, on_pre='I_syn_post -= w_ie * 100 * pA', name='s_inh_exc')
    S_inh_exc.connect(condition='i != j')

    # Monitors
    spike_monitor = SpikeMonitor(exc_group, name='sp_exc')
        
    # Weight Normalization
    @network_operation(dt=time_per_img)
    def normalize_weights():
        w_array = S_inp_exc.w[:]
        col_sums = np.bincount(S_inp_exc.j, weights=w_array, minlength=n_e)
        col_sums[col_sums == 0] = 1.0 
        scale_factors = target_weight / col_sums
        S_inp_exc.w = w_array * scale_factors[S_inp_exc.j]

    net = Network(inp_group, exc_group, inh_group, S_inp_exc, S_exc_inh, S_inh_exc, 
                  spike_monitor, normalize_weights)
    
    return net, inp_group, spike_monitor


def build_network_test():
    ## Input Layer
    inp_group = PoissonGroup(n_input, rates='input_rates(t - batch_start_time, i)', name='inp')

    # Excitatory Layer
    exc_group = NeuronGroup(n_e, eqs_e_test, threshold='u > (VT + theta_hom)', 
                            reset=reset_exc, method='euler', name='exc')
    exc_group.u = EL
    exc_group.dubar_minus = EL
    exc_group.dubar_plus = EL
    
    # Inhibitory Layer
    inh_group = NeuronGroup(n_i, eqs_i, threshold='v > I_thresh', 
                            reset='v = I_reset', refractory=I_refrac, 
                            method='euler', name='inh')
    inh_group.v = I_rest
    
    # Input -> Excitatory (VDSP Plasticity)
    S_inp_exc = Synapses(inp_group, exc_group, model=eqs_clopath,
                         on_pre=on_pre_clopath, name='s_inp_exc')
    S_inp_exc.connect(p=1.0)
    S_inp_exc.w = 'rand() * wmax'
    S_inp_exc.delay = 'rand() * max_delay'

    # Excitatory -> Inhibitory (1-to-1)
    S_exc_inh = Synapses(exc_group, inh_group, on_pre='ge_post += w_ei', name='s_exc_inh')
    S_exc_inh.connect(j='i')

    # Inhibitory -> Excitatory (Winner-Take-All lateral inhibition)
    S_inh_exc = Synapses(inh_group, exc_group, on_pre='I_syn_post -= w_ie * 100 * pA', name='s_inh_exc')
    S_inh_exc.connect(condition='i != j')

    # Monitors
    spike_monitor = SpikeMonitor(exc_group, name='sp_exc')
    
    net = Network(inp_group, exc_group, inh_group, S_inp_exc, S_exc_inh, S_inh_exc, 
                  spike_monitor)
    
    return net, inp_group, spike_monitor