# Related Method Feasibility

## HALP

- detection_granularity: sample-level object yes/no hallucination probe
- required_supervision: binary hallucination labels
- required_access_type: pre-generation hidden states
- generation_timing: pre-generation
- method_type: detector
- executable_with_current_cache: true
- incompatibility_reason: 

## EnsemHalDet

- detection_granularity: ensemble over multiple internal representations
- required_supervision: binary or method-specific hallucination labels
- required_access_type: multiple internal feature families
- generation_timing: varies by setup
- method_type: detector
- executable_with_current_cache: false
- incompatibility_reason: requires heavier multi-representation ensemble access beyond Stage D fair constraints

## VIB-Probe

- detection_granularity: attention/head-level probe
- required_supervision: probe labels and bottleneck training setup
- required_access_type: attention-head or method-specific internal signals
- generation_timing: pre-generation/internal-state dependent
- method_type: detector
- executable_with_current_cache: false
- incompatibility_reason: current full-cache stores full-layer hidden trajectories, not attention-head bottleneck signals

## HaloProbe

- detection_granularity: token-level hallucination posterior
- required_supervision: token-level posterior or compatible annotations
- required_access_type: token-level Bayesian/probe signals
- generation_timing: token-level
- method_type: detector
- executable_with_current_cache: false
- incompatibility_reason: requires token-level labels or posterior signals not present in the Stage D cache
