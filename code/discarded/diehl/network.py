from eqn_garg import *
from brian2 import *

temp = np.zeros((1, n_input))
pixel_input = TimedArray(temp, dt=50*ms) # dt will be overwritten in notebook
t_offset = 0 * ms

def build_network_train():
    # Input Layer
    inp_group = NeuronGroup(n_input, eqs_in, threshold='v > V_thresh', 
							reset='v = V_reset', refractory=refrac_in, 
							method='euler', name='inp')
    inp_group.v = V_rest
    # inp_group.I = 0 * volt
    
    # Excitatory Layer
    exc_group = NeuronGroup(n_e, eqs_e, threshold='v > adaptive_thresh', 
                            reset='v = E_reset; theta += d_theta', 
                            refractory=E_refrac, method='euler', name='exc')
    exc_group.v = E_rest
    exc_group.theta = init_theta
    
    # Inhibitory Layer
    inh_group = NeuronGroup(n_i, eqs_i, threshold='v > I_thresh', 
                            reset='v = I_reset', refractory=I_refrac, 
                            method='euler', name='inh')
    inh_group.v = I_rest
    
    # Input -> Excitatory (VDSP Plasticity)
    S_inp_exc = Synapses(inp_group, exc_group, model=eqs_vdsp,
                         on_pre=eqs_vdsp_pre, on_post=eqs_vdsp_post, 
                         name='s_inp_exc')
    S_inp_exc.connect(p=1.0)
    S_inp_exc.w = 'rand() * wmax'
    S_inp_exc.delay = 'rand() * max_delay'

    # Excitatory -> Inhibitory (1-to-1)
    S_exc_inh = Synapses(exc_group, inh_group, on_pre='ge_post += w_ei', name='s_exc_inh')
    S_exc_inh.connect(j='i')

    # Inhibitory -> Excitatory (Winner-Take-All lateral inhibition)
    S_inh_exc = Synapses(inh_group, exc_group, on_pre='gi_post += w_ie', name='s_inh_exc')
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
    # Input Layer
    inp_group = NeuronGroup(n_input, eqs_in, threshold='v > V_thresh', 
							reset='v = V_reset', refractory=refrac_in, 
							method='euler', name='inp')
    inp_group.v = V_rest
    inp_group.I = 0 * volt
    
    # Excitatory Layer
    exc_group = NeuronGroup(n_e, eqs_e_test, threshold='v > adaptive_thresh', reset='v = E_reset', 
							refractory=E_refrac, method='euler', name='exc')
    exc_group.v = E_rest
    
    # Inhibitory Layer
    inh_group = NeuronGroup(n_i, eqs_i, threshold='v > I_thresh', 
                            reset='v = I_reset', refractory=I_refrac, 
                            method='euler', name='inh')
    inh_group.v = I_rest
    
    # Input -> Excitatory (No VDSP)
    S_inp_exc = Synapses(inp_group, exc_group, model=eqs_vdsp,
                         on_pre='ge_post += w',name='s_inp_exc')
    S_inp_exc.connect(p=1.0)
    S_inp_exc.delay = 'rand() * max_delay'

    # Excitatory -> Inhibitory (1-to-1)
    S_exc_inh = Synapses(exc_group, inh_group, on_pre='ge_post += w_ei', name='s_exc_inh')
    S_exc_inh.connect(j='i')

    # Inhibitory -> Excitatory (Winner-Take-All lateral inhibition)
    S_inh_exc = Synapses(inh_group, exc_group, on_pre='gi_post += w_ie', name='s_inh_exc')
    S_inh_exc.connect(condition='i != j')

    # Monitors
    spike_monitor = SpikeMonitor(exc_group, name='sp_exc')
    
    net = Network(inp_group, exc_group, inh_group, S_inp_exc, S_exc_inh, S_inh_exc, 
                  spike_monitor)
    
    return net, inp_group, spike_monitor