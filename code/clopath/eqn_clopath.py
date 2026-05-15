from brian2 import *

n_input = 784
n_e = 100
n_i = n_e 
time_per_img = 500 * ms
target_weight = 78.0    
max_delay = 10 * ms

C = 281*pF
gL = 30*nS
EL = -70.6*mV
VT = -50.4*mV
DeltaT = 2*mV
tau_w = 144*ms
a = 4*nS
b_ad = 0.805*pA

tau_minus = 10*ms
tau_plus = 7*ms
theta_minus = -70.6 * mV  # Resting potential threshold
theta_plus = -45.3*mV   # High voltage threshold for LTP
I_sp = 400*pA   # Spike current after a spike
tau_z = 40*ms   # Spike current time constant
tau_x = 15*ms
A_LTD = 2e-5
A_LTP = 12e-5 
w_max = 1.0

# diehl and cook neuron parameters
# E is for excitory and I for inhibitory
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
tau_hom = 1e5 * ms         # Homeostatic time constant
theta_hom_inc = 0.5 * mV  # Homeostatic threshold increment

# synapse weight
w_ei = 10.4
w_ie = 17.0
wmax = 1.0

# Time Constants
tau_e = 100 * ms
tau_i = 10 * ms
tau_ge = 10 * ms
tau_gi = 15 * ms  # Increased from 2
tau_theta = 1e7 * ms

VT_max = -30.4 * mV
tau_VT = 50 * ms

# excitatory neurons

# eqs_e = '''
# 	du/dt = (-gL*(u - EL) + gL*DeltaT*exp((u - VT)/DeltaT) - w_ad + z + I_syn)/C : volt
# 	dw_ad/dt = (a*(u - EL) - w_ad)/tau_w : amp
# 	dz/dt = -z/tau_z : amp
# 	dubar_minus/dt = (u - ubar_minus)/tau_minus : volt
# 	dubar_plus/dt = (u - ubar_plus)/tau_plus : volt
# 	dtheta_hom/dt = -theta_hom/tau_hom : volt
# 	dI_syn/dt = -I_syn/tau_ge : amp
# 	'''
eqs_e = '''
	du/dt = (-gL*(u - EL) + gL*DeltaT*exp((u - VT_eff)/DeltaT) - w_ad + z + I_syn + g_inh*(E_inh - u))/C : volt
	VT_eff = VT + VT_spike + theta_hom : volt
	dVT_spike/dt = -VT_spike / tau_VT : volt 
	dw_ad/dt = (a*(u - EL) - w_ad)/tau_w : amp
	dz/dt = -z/tau_z : amp 
	dubar_minus/dt = (u - ubar_minus)/tau_minus : volt
	dubar_plus/dt = (u - ubar_plus)/tau_plus : volt
	dtheta_hom/dt = -theta_hom/tau_hom : volt
	dI_syn/dt = -I_syn/tau_ge : amp
	dg_inh/dt = -g_inh/tau_gi : siemens
	'''

reset_exc = '''
	u = EL
	w_ad += b_ad
	z = I_sp
	VT_spike = VT_max - VT 
	theta_hom += theta_hom_inc
	'''

# Disable homeostasis during the inference

# eqs_e_test = '''
# 	du/dt = (-gL*(u - EL) + gL*DeltaT*exp((u - VT)/DeltaT) - w_ad + z + I_syn)/C : volt
# 	dw_ad/dt = (a*(u - EL) - w_ad)/tau_w : amp
# 	dz/dt = -z/tau_z : amp
# 	dubar_minus/dt = (u - ubar_minus)/tau_minus : volt
# 	dubar_plus/dt = (u - ubar_plus)/tau_plus : volt
# 	theta_hom : volt
# 	dI_syn/dt = -I_syn/tau_ge : amp  
# 	'''
eqs_e_test = '''
	du/dt = (-gL*(u - EL) + gL*DeltaT*exp((u - VT_eff)/DeltaT) - w_ad + z + I_syn + g_inh*(E_inh - u))/C : volt
	VT_eff = VT + VT_spike + theta_hom : volt
	dVT_spike/dt = -VT_spike / tau_VT : volt   
	dw_ad/dt = (a*(u - EL) - w_ad)/tau_w : amp
	dz/dt = -z/tau_z : amp
	dubar_minus/dt = (u - ubar_minus)/tau_minus : volt
	dubar_plus/dt = (u - ubar_plus)/tau_plus : volt
	theta_hom : volt
	dI_syn/dt = -I_syn/tau_ge : amp
	dg_inh/dt = -g_inh/tau_gi : siemens
'''
# inhibitory neurons

eqs_i = '''
	dv/dt = ((I_rest - v) + ge*(E_exc - v) + gi*(E_inh - v)) / tau_i : volt (unless refractory)
	dge/dt = -ge / tau_ge : 1
	dgi/dt = -gi / tau_gi : 1
	'''

# VDSP synapse equations

eqs_clopath = '''
	dx_bar/dt = -x_bar/tau_x : 1 (clock-driven)
	u_rect = int(u_post > theta_plus) * (u_post - theta_plus)/mV : 1
	u_bar_plus_rect = int(ubar_plus_post > theta_minus) * (ubar_plus_post - theta_minus)/mV : 1
	dw/dt = A_LTP * x_bar * u_rect * u_bar_plus_rect * int(w < w_max) * Hz : 1 (clock-driven)
	'''

on_pre_clopath = '''
	x_bar += 1
	u_bar_minus_rect = int(ubar_minus_post > theta_minus) * (ubar_minus_post - theta_minus)/mV
	w = clip(w - A_LTD * u_bar_minus_rect, 0, w_max)
	I_syn_post += w * 200 * pA
	'''
