# Clopath VDSP-MNIST Network — Variable & Constant Reference

> Covers `eqn_clopath.py` and `network_clopath.py`.  
> Every variable, constant, and equation string is listed with its type, units, value (where fixed), and biological/computational meaning.

---

## Table of Contents

1. [Network Architecture Constants](#1-network-architecture-constants)
2. [Simulation Timing Constants](#2-simulation-timing-constants)
3. [AdEx Neuron Parameters (Clopath Paper)](#3-adex-neuron-parameters-clopath-paper)
4. [VDSP Plasticity Parameters](#4-vdsp-plasticity-parameters)
5. [Diehl & Cook Neuron Parameters](#5-diehl--cook-neuron-parameters)
6. [Synapse Weight Constants](#6-synapse-weight-constants)
7. [Time Constants](#7-time-constants)
8. [Spike-Triggered Adaptive Threshold Parameters](#8-spike-triggered-adaptive-threshold-parameters)
9. [Neuron Equations](#9-neuron-equations)
   - [Excitatory Neuron State Variables (Training)](#91-excitatory-neuron-state-variables-training--eqs_e)
   - [Excitatory Neuron State Variables (Test)](#92-excitatory-neuron-state-variables-test--eqs_e_test)
   - [Inhibitory Neuron State Variables](#93-inhibitory-neuron-state-variables--eqs_i)
   - [Reset Equations](#94-reset-equations--reset_exc)
10. [VDSP Synapse Variables](#10-vdsp-synapse-variables)
    - [Synapse Model State Variables](#101-synapse-model-state-variables--eqs_clopath)
    - [On-Pre Event Variables](#102-on-pre-event-variables--on_pre_clopath)
11. [Network Builder Variables (`network_clopath.py`)](#11-network-builder-variables-network_clopathpy)
    - [Neuron Groups](#111-neuron-groups)
    - [Synapse Objects](#112-synapse-objects)
    - [Initial Conditions](#113-initial-conditions)
    - [Weight Normalization](#114-weight-normalization)
    - [Monitors](#115-monitors)

---

## 1. Network Architecture Constants

Defined in `eqn_clopath.py`.

| Name | Value | Unit | Description |
|---|---|---|---|
| `n_input` | 784 | dimensionless | Number of input neurons. Equals 28×28, one per MNIST pixel. Each pixel drives one Poisson spike generator. |
| `n_e` | 100 | dimensionless | Number of excitatory neurons in the output layer. Each one is intended to learn a prototype of one or more digit classes. Increase for higher capacity (Diehl & Cook used 400 for best accuracy). |
| `n_i` | 100 | dimensionless | Number of inhibitory interneurons. Always set equal to `n_e` because the E→I projection is 1-to-1 (neuron `i` excites inhibitor `i`). |

---

## 2. Simulation Timing Constants

| Name | Value | Unit | Description |
|---|---|---|---|
| `time_per_img` | 500 | ms | Total simulation time allocated per image (presentation + rest). Used as the period of the weight normalization `network_operation`. |
| `max_delay` | 10 | ms | Maximum axonal conduction delay for input→excitatory synapses. Each synapse gets a uniform random delay in `[0, max_delay]`. Adds temporal jitter that can improve sparse coding. |

---

## 3. AdEx Neuron Parameters (Clopath Paper)

These govern the Adaptive Exponential Integrate-and-Fire (AdEx) membrane dynamics of excitatory neurons, taken directly from Table 1 of the Clopath & Gerstner (2010) paper.

| Name | Value | Unit | Symbol | Description |
|---|---|---|---|---|
| `C` | 281 | pF | C | Membrane capacitance. Scales how fast the voltage responds to injected current. Larger C → slower voltage changes. |
| `gL` | 30 | nS | g_L | Leak conductance. Determines how strongly the membrane is pulled back toward `EL` when no input is present. Inverse of membrane resistance. |
| `EL` | −70.6 | mV | E_L | Leak reversal potential / resting membrane potential. The voltage the neuron relaxes to at rest. Also used as the initial condition for `u`, `ubar_minus`, and `ubar_plus`. |
| `VT` | −50.4 | mV | V_T^rest | Baseline spike threshold. The voltage at which the exponential sodium-activation term begins to dominate and drives the membrane to a spike. This is the *resting* threshold, before any adaptation. |
| `DeltaT` | 2 | mV | ΔT | Slope factor of the exponential term. Controls the sharpness of spike initiation. Smaller → more abrupt threshold crossing. |
| `tau_w` | 144 | ms | τ_{w_ad} | Adaptation time constant. Controls how quickly the subthreshold adaptation current `w_ad` decays back toward zero. |
| `a` | 4 | nS | a | Subthreshold adaptation coupling. Couples `w_ad` to the membrane voltage; larger `a` means stronger steady-state adaptation at depolarised voltages (increases firing rate adaptation). |
| `b_ad` | 0.805 | pA | b | Spike-triggered adaptation increment. `w_ad` jumps by this amount after every spike, causing firing rate adaptation (spike frequency accommodation). |
| `I_sp` | 400 | pA | I_sp | Depolarising spike afterpotential current. `z` is set to this value immediately after a spike and then decays. This sustained depolarisation after a spike is required by the VDSP rule: it keeps `ubar_plus` elevated, enabling LTP from nearby pre-post-post triplets. |
| `tau_z` | 40 | ms | τ_z | Decay time constant of the spike afterpotential current `z`. Determines how long after a spike the depolarising afterpotential remains significant. |

---

## 4. VDSP Plasticity Parameters

These parameters define the voltage-dependent synaptic plasticity (VDSP) learning rule from Clopath & Gerstner (2010). Values are fitted to Figure 3 of the paper (visual cortex data).

| Name | Value | Unit | Symbol | Description |
|---|---|---|---|---|
| `tau_minus` | 10 | ms | τ_− | Low-pass filter time constant for `ubar_minus`. Determines the time window over which the post-synaptic voltage must have been elevated for LTD to be triggered at a presynaptic spike. |
| `tau_plus` | 7 | ms | τ_+ | Low-pass filter time constant for `ubar_plus`. Determines the time window over which previous postsynaptic depolarisation contributes to the LTP term. Shorter than `tau_minus` → LTP requires more recent postsynaptic activity. |
| `theta_minus` | −70.6 | mV | θ_− | LTD voltage threshold. Depression is only induced at presynaptic spike arrival if `ubar_minus > theta_minus`. Set to resting potential, so any depolarisation above rest permits LTD. |
| `theta_plus` | −45.3 | mV | θ_+ | LTP voltage threshold. Potentiation only occurs when the instantaneous voltage `u > theta_plus`, i.e., near or above spike threshold. Ensures LTP requires actual spiking activity, not just subthreshold depolarisation. |
| `tau_x` | 15 | ms | τ_x | Decay time constant of the presynaptic eligibility trace `x_bar`. Represents how long a presynaptic spike leaves a "tag" at the synapse during which LTP can occur if postsynaptic conditions are met. |
| `A_LTD` | 14×10⁻⁵ | 1/mV | A_LTD | LTD learning rate amplitude. Scales the amount of depression per presynaptic spike. Dimensionally, it converts `ubar_minus_rect` (in mV above θ_−) to a dimensionless weight change. |
| `A_LTP` | 12×10⁻⁵ | 1/(mV²·Hz) | A_LTP | LTP learning rate amplitude. Scales the continuous potentiation term. The units arise because the LTP ODE integrates `x_bar × u_rect × ubar_plus_rect × Hz` continuously. |
| `w_max` / `wmax` | 1.0 | dimensionless | w_max | Maximum allowed synaptic weight. Hard upper bound enforced by `clip()` in LTD and implicitly by the VDSP equations. Note: `w_max` and `wmax` are both defined and used interchangeably — should be unified. |

---

## 5. Diehl & Cook Neuron Parameters

Taken from the Diehl & Cook (2015) MNIST-SNN paper. Used for both excitatory and inhibitory neuron dynamics outside of the AdEx membrane equation.

| Name | Value | Unit | Description |
|---|---|---|---|
| `E_rest` | −65.0 | mV | Excitatory neuron resting potential (Diehl & Cook convention). Not used directly in `eqs_e` (which uses AdEx `EL` instead); present for reference. |
| `I_rest` | −60.0 | mV | Inhibitory neuron resting potential. Used in `eqs_i` as the leak term driving `v` toward this value at rest. |
| `E_reset` | −65.0 | mV | Excitatory reset voltage (Diehl & Cook). Superseded by AdEx reset `u = EL` in `reset_exc`. |
| `I_reset` | −45.0 | mV | Inhibitory neuron reset voltage. Applied after each inhibitory spike: `v = I_reset`. |
| `E_thresh` | −52.0 | mV | Excitatory threshold (Diehl & Cook). Superseded by AdEx exponential threshold; actual firing controlled by `u > -20*mV` (the spike peak cutoff). |
| `I_thresh` | −40.0 | mV | Inhibitory neuron firing threshold. Used in `threshold='v > I_thresh'` for `inh_group`. |
| `E_refrac` | 5.0 | ms | Excitatory refractory period. After a spike, the neuron is unable to fire for this duration. `u` and `z` are frozen during this window (`unless refractory`). |
| `I_refrac` | 2.0 | ms | Inhibitory refractory period. Shorter than excitatory, allowing inhibitory neurons to follow excitatory activity more rapidly. |
| `E_exc` | 0.0 | mV | Excitatory synapse reversal potential. Used in `eqs_i`: excitatory conductance `ge` drives `v` toward 0 mV, which is well above rest — always depolarising. |
| `E_inh` | −100.0 | mV | Inhibitory synapse reversal potential. Used in `eqs_i`: inhibitory conductance `gi` drives `v` toward −100 mV, strongly hyperpolarising. |
| `tau_hom` | 1×10⁵ | ms (= 100 s) | Homeostatic threshold time constant. `theta_hom` decays with this time constant. Very long → homeostasis is slow, requiring many spikes over many images to accumulate. |
| `theta_hom_inc` | 0.5 | mV | Homeostatic threshold increment per spike. Each excitatory spike increases `theta_hom` by this amount, gradually raising the effective threshold `VT_eff` and reducing the neuron's firing probability. Tuned to 0.5 mV (from original 0.05 mV) to allow homeostasis to function over short (100-image) training runs. |

---

## 6. Synapse Weight Constants

| Name | Value | Unit | Description |
|---|---|---|---|
| `w_ei` | 10.4 | dimensionless (conductance units) | Weight of each excitatory→inhibitory synapse. When excitatory neuron `i` spikes, `ge` of inhibitory neuron `i` increases by `w_ei`. Controls how strongly a winner activates its paired inhibitor. |
| `w_ie` | 17.0 | dimensionless (× 100 pA) | Weight of each inhibitory→excitatory synapse. When inhibitory neuron `i` fires, `I_inh` of all other excitatory neurons decreases by `w_ie × 100 pA`. High value enforces strong winner-take-all (WTA). Reducing this (e.g., to 10.0) softens competition and may help prevent mode collapse. |
| `target_weight` | 78.0 | dimensionless | Target sum of incoming weights per excitatory neuron. The weight normalizer rescales all weights to each neuron so their column sum equals this value. With 784 inputs and `wmax=1.0`, a sum of 78 ≈ 10% of maximum, keeping the network in a reasonable activity regime. |

---

## 7. Time Constants

| Name | Value | Unit | Description |
|---|---|---|---|
| `tau_e` | 100 | ms | Declared excitatory membrane time constant (`C/gL` = 281pF/30nS ≈ 9.4 ms). **Not actually used** in any equation — the AdEx equations use `C` and `gL` directly. Present as a reference value. |
| `tau_i` | 10 | ms | Inhibitory neuron membrane time constant. Used in `eqs_i` as the denominator of the `dv/dt` equation, determining how fast the inhibitory membrane voltage responds to input. |
| `tau_ge` | 10 | ms | Excitatory synaptic conductance / current decay time constant. Used for `dI_syn/dt` in excitatory neurons and `dge/dt` in inhibitory neurons. Determines the duration of excitatory postsynaptic currents. |
| `tau_gi` | 15 | ms | Inhibitory synaptic conductance / current decay time constant. Used for `dI_inh/dt` in excitatory neurons and `dgi/dt` in inhibitory neurons. Longer than `tau_ge` → inhibition outlasts excitation, aiding stable WTA dynamics. |
| `tau_theta` | 1×10⁷ | ms | Declared alternative homeostatic time constant. **Not used** in any equation in the current code — `tau_hom` is used instead. Legacy variable from an earlier version. |

---

## 8. Spike-Triggered Adaptive Threshold Parameters

These implement the fast adaptive threshold from Table 1 of Clopath & Gerstner (2010), separate from the slow Diehl & Cook homeostasis.

| Name | Value | Unit | Description |
|---|---|---|---|
| `VT_max` | −30.4 | mV | Threshold value immediately after a spike. At reset, `VT_spike = VT_max - VT = -30.4 - (-50.4) = 20 mV`. This means `VT_eff` immediately jumps to `VT + 20 mV = -30.4 mV`, far above the normal threshold, preventing re-firing during the afterpotential. |
| `tau_VT` | 50 | ms | Decay time constant of `VT_spike`. After a spike, the elevated threshold decays back to baseline with this time constant. At 5τ = 250 ms it is essentially recovered. This works in concert with `E_refrac = 5 ms` to prevent bursting. |

---

## 9. Neuron Equations

### 9.1 Excitatory Neuron State Variables (Training) — `eqs_e`

These are the differential equations and derived quantities integrated for each excitatory neuron during training.

| Variable | Unit | Type | Description |
|---|---|---|---|
| `u` | volt | ODE (unless refractory) | Membrane voltage. The central state variable of the AdEx neuron. Driven by leak, exponential spike initiation, adaptation, afterpotential, and synaptic currents. Frozen during refractory period. |
| `VT_eff` | volt | Derived (subexpression) | Effective spike threshold. `VT_eff = VT + VT_spike + theta_hom`. Combines: baseline threshold `VT`, fast post-spike elevation `VT_spike`, and slow homeostatic elevation `theta_hom`. |
| `VT_spike` | volt | ODE | Fast spike-triggered threshold elevation. Jumps to `VT_max - VT = 20 mV` at each spike and decays back to 0 with time constant `tau_VT = 50 ms`. Prevents burst firing during the `z`-current afterpotential. |
| `w_ad` | amp | ODE | Adaptation current (hyperpolarising). Increases by `b_ad` at each spike and is coupled to voltage via `a*(u - EL)`. Acts as a negative feedback that slows repeated firing — models slow K⁺ afterhyperpolarisation currents. |
| `z` | amp | ODE (unless refractory) | Spike afterpotential current (depolarising). Set to `I_sp = 400 pA` at each spike, then decays with `tau_z = 40 ms`. **Critical for VDSP**: this sustained post-spike depolarisation keeps `ubar_plus` above `theta_minus`, enabling the LTP term for subsequent presynaptic spikes. |
| `ubar_minus` | volt | ODE | Low-pass filtered membrane voltage with time constant `tau_minus = 10 ms`. Represents the recent voltage history used to gate LTD: depression only occurs at a presynaptic spike if `ubar_minus > theta_minus`. Initialised to `EL`. |
| `ubar_plus` | volt | ODE | Low-pass filtered membrane voltage with time constant `tau_plus = 7 ms`. Used in the LTP term: potentiation requires `ubar_plus > theta_minus`, ensuring the postsynaptic neuron was recently depolarised before the current spike. Initialised to `EL`. |
| `theta_hom` | volt | ODE | Slow homeostatic threshold. Increases by `theta_hom_inc` per spike, decays with `tau_hom = 100 s`. Prevents any single neuron from permanently monopolising all responses (prevents mode collapse). |
| `I_syn` | amp | ODE | Excitatory synaptic input current. Incremented by `w × 200 pA` at each presynaptic spike arrival, then decays with `tau_ge = 10 ms`. Drives the neuron toward firing threshold. |
| `I_inh` | amp | ODE | Inhibitory synaptic input current. Decremented by `w_ie × 100 pA` when a lateral inhibitory neuron fires, then decays with `tau_gi = 15 ms`. Implements winner-take-all suppression. Initialised to 0. |

---

### 9.2 Excitatory Neuron State Variables (Test) — `eqs_e_test`

Same as `eqs_e` except `theta_hom` is a **static parameter** (no ODE), not updated during inference. This preserves the learned homeostatic threshold from training while preventing further drift.

| Variable | Unit | Type | Difference from training |
|---|---|---|---|
| `u` | volt | ODE (unless refractory) | Same |
| `VT_eff` | volt | Derived | Same |
| `VT_spike` | volt | ODE | Same |
| `w_ad` | amp | ODE | Same |
| `z` | amp | ODE (unless refractory) | Same |
| `ubar_minus` | volt | ODE | Same |
| `ubar_plus` | volt | ODE | Same |
| `theta_hom` | volt | **Static parameter** | Frozen — loaded from saved training value, no longer evolves |
| `I_syn` | amp | ODE | Same |
| `I_inh` | amp | ODE | Same |

---

### 9.3 Inhibitory Neuron State Variables — `eqs_i`

Simple leaky integrate-and-fire (LIF) model for inhibitory interneurons.

| Variable | Unit | Type | Description |
|---|---|---|---|
| `v` | volt | ODE (unless refractory) | Membrane voltage of the inhibitory neuron. Driven by leak toward `I_rest`, and by conductance-based excitatory and inhibitory inputs. Spikes when `v > I_thresh = -40 mV`. |
| `ge` | dimensionless | ODE | Excitatory conductance (in units of `E_exc - v` scaling). Incremented by `w_ei` when the paired excitatory neuron fires, decays with `tau_ge`. Drives `v` toward `E_exc = 0 mV`. |
| `gi` | dimensionless | ODE | Inhibitory conductance. Decays with `tau_gi`. **Currently never incremented** by any synapse — effectively always 0. Present for completeness / future use. |

---

### 9.4 Reset Equations — `reset_exc`

Applied instantaneously when `u > -20 mV` (spike peak cutoff).

| Statement | Description |
|---|---|
| `u = EL` | Resets membrane voltage to resting potential (−70.6 mV). |
| `w_ad += b_ad` | Spike-triggered adaptation increment: increases hyperpolarising adaptation current by 0.805 pA. |
| `z = I_sp` | Sets depolarising afterpotential current to 400 pA, which then decays with `tau_z`. |
| `VT_spike = VT_max - VT` | Sets fast threshold elevation to 20 mV (so `VT_eff` jumps to −30.4 mV), then decays with `tau_VT`. |
| `theta_hom += theta_hom_inc` | Increments slow homeostatic threshold by 0.5 mV per spike. |

---

## 10. VDSP Synapse Variables

### 10.1 Synapse Model State Variables — `eqs_clopath`

One copy of these variables exists **per synapse** (i.e., per input–excitatory neuron pair; 784 × 100 = 78,400 copies total).

| Variable | Unit | Type | Description |
|---|---|---|---|
| `x_bar` | dimensionless | ODE (clock-driven) | Presynaptic eligibility trace. Jumps by +1 at each presynaptic spike (in `on_pre`), then decays with `tau_x = 15 ms`. Represents the lingering effect of a presynaptic spike at the synapse — the "glutamate trace" or NMDA receptor activation window during which LTP can be induced if postsynaptic conditions are met. |
| `u_rect` | dimensionless | Derived (subexpression) | Rectified instantaneous postsynaptic voltage: `[u_post − θ_+]_+` in units of mV. Zero when `u < theta_plus`, positive only when the neuron is actively spiking. This is the LTP gating term that ensures potentiation only happens during postsynaptic spikes. |
| `u_bar_plus_rect` | dimensionless | Derived (subexpression) | Rectified low-pass filtered postsynaptic voltage: `[ubar_plus_post − θ_−]_+` in mV. Represents recent history of postsynaptic depolarisation. Together with `u_rect` and `x_bar`, this implements the triplet (post-pre-post) requirement for LTP. |
| `w` | dimensionless | ODE (clock-driven) | Synaptic weight. Continuously increased by LTP term (`A_LTP × x_bar × u_rect × ubar_plus_rect`). Decreased by LTD term in `on_pre`. Hard-bounded to `[0, wmax]`. |

---

### 10.2 On-Pre Event Variables — `on_pre_clopath`

Executed instantaneously at the moment a presynaptic spike arrives at the synapse.

| Variable | Unit | Description |
|---|---|---|
| `x_bar += 1` | — | Increments the presynaptic trace by 1. The trace then decays with `tau_x` between spikes. |
| `u_bar_minus_rect` | dimensionless (mV) | Local temporary variable. Computed as `[ubar_minus_post − θ_−]_+` in mV at the instant of spike arrival. Non-zero only if the postsynaptic neuron was depolarised in the recent past (`tau_minus = 10 ms` window). Used immediately to compute the LTD weight decrement. |
| `w = clip(w - A_LTD × u_bar_minus_rect, 0, w_max)` | — | LTD update: depresses the weight proportionally to how depolarised the postsynaptic neuron was recently. Clipped to stay in `[0, wmax]`. This implements the "post-before-pre" depression pathway. |
| `I_syn_post += w × 200 pA` | amp | Synaptic transmission: delivers excitatory current to the postsynaptic neuron. The `200 pA` scale factor converts dimensionless weight `w ∈ [0, 1]` into a physiologically plausible current. |

---

## 11. Network Builder Variables (`network_clopath.py`)

### 11.1 Neuron Groups

| Variable | Type | Description |
|---|---|---|
| `inp_group` | `PoissonGroup(n_input)` | 784 independent Poisson spike generators. Each neuron's firing rate is given by `input_rates(t - batch_start_time, i)`, a `TimedArray` that encodes pixel intensities as firing rates (pixel / 4.0 Hz). The `t - batch_start_time` offset resets the lookup index to 0 at the start of each batch. |
| `exc_group` | `NeuronGroup(n_e, eqs_e)` | 100 AdEx excitatory neurons with VDSP homeostasis. Threshold `u > -20 mV` marks spike peak. Uses `reset_exc` and 5 ms refractory. These are the neurons that learn digit representations. |
| `inh_group` | `NeuronGroup(n_i, eqs_i)` | 100 LIF inhibitory interneurons. Threshold `v > -40 mV`, reset `v = -45 mV`, 2 ms refractory. Implement WTA competition by inhibiting all non-winning excitatory neurons. |

---

### 11.2 Synapse Objects

| Variable | Type | Connectivity | Description |
|---|---|---|---|
| `S_inp_exc` | `Synapses` (VDSP) | All-to-all (`p=1.0`) | 784 × 100 = 78,400 plastic synapses from input Poisson neurons to excitatory neurons. Carries `eqs_clopath` (LTP ODE + eligibility trace) and `on_pre_clopath` (LTD + synaptic current). Random initial weights `U[0, wmax]`, random delays `U[0, max_delay]`. |
| `S_exc_inh` | `Synapses` (static) | 1-to-1 (`j='i'`) | 100 synapses, excitatory neuron `i` → inhibitory neuron `i`. On each excitatory spike, increments `ge` of the matched inhibitory neuron by `w_ei = 10.4`. Allows winners to activate their paired inhibitor. |
| `S_inh_exc` | `Synapses` (static) | All-but-self (`i != j`) | 100 × 99 = 9,900 synapses from each inhibitory neuron to all other excitatory neurons. On each inhibitory spike, decrements `I_inh` of target by `w_ie × 100 pA = 1700 pA`. This is the lateral inhibition that implements WTA. |

---

### 11.3 Initial Conditions

Set immediately after group creation in `build_network_train()`:

| Statement | Description |
|---|---|
| `exc_group.u = EL` | Initialises all excitatory membrane voltages to resting potential (−70.6 mV). |
| `exc_group.ubar_minus = EL` | Initialises LTD trace to resting potential — no recent depolarisation history. |
| `exc_group.ubar_plus = EL` | Initialises LTP trace to resting potential — no recent depolarisation history. |
| `exc_group.VT_spike = 0 * mV` | (Recommended) Fast threshold elevation starts at zero — no recent spikes. |
| `inh_group.v = I_rest` | Initialises inhibitory membrane voltages to inhibitory rest (−60 mV). |
| `S_inp_exc.w = 'rand() * wmax'` | Randomly initialises all synaptic weights uniformly in `[0, 1]`. |
| `S_inp_exc.delay = 'rand() * max_delay'` | Assigns each synapse a random axonal delay uniformly in `[0, 10 ms]`. |

---

### 11.4 Weight Normalization

| Variable | Description |
|---|---|
| `w_array` | Numpy array snapshot of all 78,400 synaptic weights at the current timestep. |
| `col_sums` | Array of shape `(n_e,)`. The sum of all incoming weights to each excitatory neuron (column sum of the 784×100 weight matrix). Computed via `np.bincount` using synapse target indices `S_inp_exc.j`. |
| `scale_factors` | Array of shape `(n_e,)`. `target_weight / col_sums`. Multiplied into each weight to rescale incoming weight sums to `target_weight = 78.0`. |
| `normalize_weights` | `@network_operation(dt=time_per_img)` — a Brian2 callback executed every 500 ms (once per image). Prevents runaway weight growth by normalising incoming weights per excitatory neuron. |

---

### 11.5 Monitors

| Variable | Type | Description |
|---|---|---|
| `spike_monitor` | `SpikeMonitor(exc_group)` | Records spike times and neuron indices for all excitatory neurons. Used post-training for: raster plots (`spike_monitor.t`, `spike_monitor.i`), per-neuron spike counts (`spike_monitor.count`), and assigning digit labels to neurons based on which images they responded to. |

---

## Quick Reference: Variable Lifecycle Summary

```
At t=0 (network build):
  u, ubar_minus, ubar_plus = EL = -70.6 mV
  VT_spike = 0 mV  →  VT_eff = -50.4 mV  (normal threshold)
  theta_hom = 0 mV  (no homeostatic elevation yet)
  w ~ U[0, 1], then normalised to column sum = 78.0

During image presentation (0–350 ms per image):
  Poisson input drives I_syn → u rises → u crosses VT_eff

At each excitatory spike:
  u ← -70.6 mV  (reset)
  w_ad += 0.805 pA  (adaptation)
  z ← 400 pA  (afterpotential, decays 40 ms)
  VT_spike ← 20 mV → VT_eff ← -30.4 mV  (decays 50 ms)
  theta_hom += 0.5 mV  (decays 100,000 ms)
  Inhibits all other excitatory neurons via WTA pathway

At each presynaptic spike (on_pre):
  x_bar += 1  (decays 15 ms)
  LTD: w -= A_LTD × [ubar_minus - theta_minus]+
  I_syn_post += w × 200 pA

Continuously (clock-driven):
  LTP: dw/dt += A_LTP × x_bar × [u - theta_plus]+ × [ubar_plus - theta_minus]+

Every 500 ms (network_operation):
  Weights renormalized: column sums → target_weight = 78.0
```
