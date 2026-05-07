from brian2 import *

n_input = 784
n_e = 400
n_i = n_e 
time_per_img = 350 * ms
target_weight = 78.0    
max_delay = 10 * ms

# vdsp
lr = 0.05
refrac_in = 5 * ms
tau_in = 30 * ms
#input layer
v_reset_in = -1.0 * volt
v_rest_in = 0 * volt
v_thresh_in = 1.0 * volt
input_bias = 0.5 * volt

# diehl and cook neuron parameters
E_rest = -65. * mV
I_rest = -60. * mV 
E_reset = -65. * mV
I_reset = -45. * mV
E_thresh = -52. * mV
I_thresh = -40. * mV
E_refrac = 5. * ms
I_refrac = 2. * ms
E_exc = 0.0 * mV
E_inh = -100.0 * mV
init_theta = 0.0 * mV
d_theta = 0.05 * mV
theta_offset = 20.0 * mV

# synapse weight
w_ei = 10.4
w_ie = 17.0
wmax = 1.0

# Time Constants
tau_e = 100 * ms
tau_i = 10 * ms
tau_ge = 1 * ms
tau_gi = 2 * ms
tau_theta = 1e7 * ms


# input layer 
eqs_in = '''
dv/dt = ((v_rest_in - v) + I + bias) / tau_in : volt (unless refractory)
I : volt  
'''

# excitatory neurons
eqs_e = '''
dv/dt = ((E_rest - v) + ge*(E_exc - v) + gi*(E_inh - v)) / tau_e : volt (unless refractory)
dge/dt = -ge / tau_ge : 1
dgi/dt = -gi / tau_gi : 1
dtheta/dt = -theta / tau_theta : volt
v_thresh = E_thresh + theta - theta_offset: volt
'''

# inhibitory neurons
eqs_i = '''
dv/dt = ((I_rest - v) + ge*(E_exc - v) + gi*(E_inh - v)) / tau_i : volt (unless refractory)
dge/dt = -ge / tau_ge : 1
dgi/dt = -gi / tau_gi : 1
'''

# VDSP synapse equations

eqs_vdsp = '''
w : 1
'''

eqs_vdsp_pre = '''
ge_post += w
'''
eqs_vdsp_post = '''
v_norm = v_pre / volt
dw_pot = (wmax - w) * (exp(-v_norm) - 1.0) * lr * int(v_norm < 0)
dw_dep = -w * (exp(v_norm) - 1.0) * lr * int(v_norm >= 0)
w = clip(w + dw_pot + dw_dep, 0.0, wmax)
'''