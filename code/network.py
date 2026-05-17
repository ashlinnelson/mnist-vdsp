from brian2 import *

defaultclock.dt = 0.5*ms

N_inputs = 28 * 28
presentation_time = 350*ms 

# Neuron Parameters
tau_m = 20*ms 
t_ref = 2*ms       # Refractory period
v_th = 1.0         # Spike threshold

# Input LIF Parameters
v_rest_in = 0.0
v_reset_in = -1.0
bias_in = 0.5 

# Output ALIF Parameters
v_rest_out = 0.0
v_reset_out = 0.0
tau_n = 1*second   # Adaptation leak time constant
inc_n = 0.01       # Adaptation increment per spike

# VDSP Parameters
w_max = 1.0

# Input Layer (LIF)
input_eqs = '''
dv/dt = (-v + I + bias_in) / tau_m : 1 (unless refractory)
I = image_stream(t, i) : 1 
'''
# Output Layer (Adaptive LIF with WTA Clamping mechanism)
output_eqs = '''
dv/dt = (-v - n) / tau_m : 1 (unless refractory)
dn/dt = -n / tau_n : 1
clamp_release_time : second
'''
# Feedforward VDSP Synapses
vdsp_model = '''w : 1'''

on_pre_forward = 'v_post += w '

# VDSP Learning Rule (Triggered on Postsynaptic Spike)
on_post_vdsp = '''
dw_pot = lr * (w_max - w) * (exp(-v_pre) - 1) * int(v_pre < 0)
dw_dep = lr * w * (exp(v_pre) - 1) * int(v_pre > 0)
w = clip(w + dw_pot - dw_dep, 0.0, w_max)
'''
# Lateral Inhibition (Winner-Takes-All 10ms Clamp)
wta_pre = '''
v_post = 0
clamp_release_time_post = t + 10*ms
'''

def build_network_train(N_outputs=10, lr=0.01):
	
	InputLayer = NeuronGroup(N_inputs, input_eqs, threshold='v > v_th', 
	                         reset='v = v_reset_in', refractory=t_ref,
							 method='exact')
	
	OutputLayer = NeuronGroup(N_outputs, output_eqs,
							  threshold='v > v_th and t >= clamp_release_time', 
	                          reset='v = v_reset_out; n += inc_n', 
	                          refractory=t_ref, method='euler')
	
	InputLayer.v = 'v_reset_in + rand() * (v_th - v_reset_in)'
	OutputLayer.v = 'v_rest_out + rand() * v_th'
	OutputLayer.n = 0.0
	OutputLayer.clamp_release_time = 0*ms 
	
	
	FeedForward = Synapses(InputLayer, OutputLayer, model=vdsp_model,
	                       on_pre=on_pre_forward, on_post=on_post_vdsp,
	                        name="feed_forward")
	FeedForward.connect() 
	FeedForward.w = 'rand() * w_max * 1' 
	
	
	WTA_Synapses = Synapses(OutputLayer, OutputLayer, on_pre=wta_pre)
	WTA_Synapses.connect(condition='i != j') 
	
	# Monitors
	spike_mon = SpikeMonitor(OutputLayer)

	@network_operation(dt=defaultclock.dt, when='thresholds', order=-1)
	def enforce_wta(t):
		v       = OutputLayer.v[:]
		t_now   = t / second  
		clamped = OutputLayer.clamp_release_time[:] / second > t_now 
		eligible = (v > v_th) & ~clamped
	
		if np.sum(eligible) > 1:
			eligible_v     = np.where(eligible, v, -np.inf)
			winner         = int(np.argmax(eligible_v))
			losers         = eligible.copy()
			losers[winner] = False
			for j in np.where(losers)[0]:
				OutputLayer.v[j] = 0.0
	
	# Assemble Network
	net = Network(InputLayer, OutputLayer, FeedForward,
				  WTA_Synapses, spike_mon, enforce_wta)

	return net, InputLayer, FeedForward, OutputLayer, spike_mon