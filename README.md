# Boltz Flow Matching: Analytical Conversion from Score-Based Diffusion

This repository implements analytical conversion from score-based diffusion to flow matching for the Boltz-2 protein structure prediction model. This approach enables faster sampling without requiring any retraining.

## Key Benefits

1. No retraining required, works with existing pretrained checkpoints
2. Same architecture, uses the existing diffusion module
3. Pure analytical transformation from score to velocity

## 📖 The Core Idea

### Score-Based Diffusion vs Flow Matching

Score-Based Diffusion Original Boltz-2
1. Uses stochastic differential equations SDEs
2. Each step involves random noise injection
3. Slower due to stochastic nature

Flow Matching This Implementation
1. Uses ordinary differential equations ODEs
2. Deterministic integration no random noise
3. Faster due to deterministic nature

### The Analytical Conversion

The key insight is that **both approaches parameterize the same underlying noise**:

```
Score Model: x_t = x_0 + σ·ε  →  ε = (x_t - x_0)/σ
Flow Model:  x_t = (1-t)·x_0 + t·ε  →  v = ε - x_0
```

Where:
- `x_t`: Noisy coordinates at time t
- `x_0`: Clean coordinates (ground truth)
- `ε`: Noise vector
- `σ`: Noise level
- `v`: Velocity field for flow matching

### Why It Is Faster

1. Fewer steps required for integration
2. Deterministic integration avoids random noise generation
3. Heun integration RK2 improves efficiency over simple Euler
4. No architectural changes reduce overhead

## Technical Implementation

### Architecture Compatibility

The implementation uses the **exact same DiffusionModule** as the original Boltz-2:

```python
# Same architecture as diffusionv2.py
self.score_model = DiffusionModule(**score_model_args)

# Only difference: analytical conversion layer
self.converter = ScoreToVelocityConverter(
    conversion_method='noise_based'  # Most accurate method
)
```

### Conversion Methods

Three analytical conversion methods are implemented:

1. **`noise_based` (RECOMMENDED)**: Most accurate
   ```python
   epsilon = (x_t - x_0_pred) / sigma
   velocity = epsilon - x_0_pred
   ```

2. **`pflow`**: Probability flow ODE
   ```python
   velocity = 0.5 * (x_0_pred - x_t)
   ```

3. **`simple`**: Direct geometric conversion
   ```python
   x_1_est = (x_t - (1-t)*x_0_pred) / t
   velocity = x_1_est - x_0_pred
   ```

### Integration Method

Uses **Heun's method (RK2)** for ODE integration:

```python
# First velocity evaluation
v1 = velocity_network_forward(x, t_curr)

# Euler step
x_euler = x + dt * v1

# Second velocity evaluation  
v2 = velocity_network_forward(x_euler, t_next)

# Heun update (average of velocities)
x_new = x + 0.5 * dt * (v1 + v2)
```

## Setup and Usage

### Environment setup with conda

```bash
conda env create -f environment.yml
conda activate boltz

# optional editable install for src package
pip install -e .
```

### Quick start

### Quick Start

```bash
# Run flow matching predictions
python run_boltz_flow_matching.py

# This will:
# 1 Load Boltz 2 checkpoint at ~/.boltz/boltz2_conf.ckpt
# 2 Convert to flow matching format
# 3 Run predictions on hackathon data
# 4 Generate results
```

### Custom parameters

```python
from run_boltz_flow_matching import BoltzFlowMatchingRunner

runner = BoltzFlowMatchingRunner(
    flow_steps=20,        # ODE integration steps
    score_steps=200,       # Original SDE steps (for comparison)
    diffusion_samples=1,   # Number of samples per protein
    device='cuda'         # Device to use
)

results = runner.run_predictions(max_proteins=5)
```

### Direct model usage

```python
from boltz.model.models.boltz2 import Boltz2

# Load converted checkpoint
model = Boltz2.load_from_checkpoint(
    "flow_matching_boltz2.ckpt",
    map_location='cuda'
)

# The model automatically uses FlowMatchingDiffusion
# when use_flow_matching=True in hyperparameters
```

## Examples and scripts

1. Main runner script run_boltz_flow_matching.py
2. Hackathon prediction script hackathon/predict_hackathon.py and helper API in hackathon/hackathon_api.py
3. Training entrypoint scripts/train/train.py with configs in scripts/train/configs
4. MSA generation scripts/generate_local_msa.py
5. Evaluation helpers under scripts/eval

## Implementation Details

### File structure

```
├── run_boltz_flow_matching.py          # Main runner script
├── src/boltz/model/modules/
│   └── diffusionv3_flow_matching.py    # Flow matching implementation
└── src/boltz/model/models/
    └── boltz2.py                       # Modified to support flow matching
```

### Key classes

1. **`BoltzFlowMatchingRunner`**: Main orchestrator
2. **`FlowMatchingDiffusion`**: Flow matching module
3. **`ScoreToVelocityConverter`**: Analytical conversion
4. **`Boltz2`**: Modified model with flow matching support

### Integration points

The flow matching is integrated into Boltz-2 through:

1. **Conditional Import**: 
   ```python
   try:
       from boltz.model.modules.diffusionv3_flow_matching import AtomDiffusion as FlowMatchingDiffusion
   except ImportError:
       FlowMatchingDiffusion = None
   ```

2. **Hyperparameter Control**:
   ```python
   if use_flow_matching and FLOW_MATCHING_AVAILABLE:
       self.structure_module = FlowMatchingDiffusion(...)
   else:
       self.structure_module = AtomDiffusion(...)
   ```

3. **Checkpoint Conversion**:
   ```python
   hparams['use_flow_matching'] = True
   hparams['flow_conversion_method'] = 'noise_based'
   ```

## Mathematical Foundation

### Score based diffusion

The score-based approach learns to predict the score function:

```
∇_x log p_t(x) ≈ s_θ(x, σ)
```

Where `s_θ` is the neural network predicting the score.

### Flow matching

Flow matching learns a velocity field:

```
dx/dt = v_θ(x, t)
```

Where `v_θ` is the neural network predicting the velocity.

### Analytical conversion

The key insight is that both parameterize the same noise:

```
Score: x_t = x_0 + σ·ε  →  ε = (x_t - x_0)/σ
Flow:  x_t = (1-t)·x_0 + t·ε  →  v = ε - x_0
```

This allows us to convert score predictions to velocity predictions **analytically**.

## Why this works

1. **Same Information**: Both models learn the same underlying data distribution
2. **Mathematical Equivalence**: The conversion is exact under certain conditions
3. **Architecture Preservation**: Same neural network weights work for both
4. **Integration Efficiency**: ODE solvers are more efficient than SDE solvers

## Future improvements

1. **Fine-tuning**: Optional 20-50 epoch fine-tuning for perfect quality
2. **Advanced ODE Solvers**: Dormand-Prince, adaptive step sizes
3. **Steering Integration**: Physical guidance for flow matching
4. **Multi-scale**: Different step counts for different protein sizes

## References

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Score-Based Diffusion Models](https://arxiv.org/abs/2011.13456)
- [Boltz-2 Paper](https://arxiv.org/abs/2402.17670)

## Contributing

This implementation provides a foundation for flow matching in protein structure prediction. Contributions welcome for

1. Advanced ODE solvers
2. Quality improvements
3. Additional conversion methods

---

The key insight is that pretrained score models can be analytically converted to flow matching without retraining.