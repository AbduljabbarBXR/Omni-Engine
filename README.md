# Omni Engine

A hybrid sparse neural system that fuses a frozen pretrained language model with learnable sparse expert clusters, localized neighborhood communication, Hebbian plasticity, and a SQLite memory hierarchy. The design is inspired by biological cortical columns: a dense base model provides structural linguistic memory while modular sparse clusters handle active localized computation. Only a tiny fraction of the network is active for any given input, mirroring neural firing patterns and drastically reducing compute per token.

The system is built to run on constrained devices. The full training and ablation harness runs on any laptop, and the resulting model with its memory store is small enough to operate on a phone.

---

## Design Philosophy

The architecture maps biological principles to concrete mechanisms.

| Biological principle | Engineering mechanism |
|---|---|
| Innate wiring from birth | Frozen Pythia 160M base model |
| Synapses that grow where used | Learnable expert clusters on hidden states |
| Sparse neural firing | Top k routing, mostly dormant clusters |
| Cortical columns | Small world neighborhood graph |
| Fire together, wire together | Hebbian co activation edge updates |
| Hippocampal memory consolidation | SQLite stores for weights and activation memories |
| Graceful degradation under pressure | Memory hierarchy with prefetch and expert offload |

The key research question is whether a graph that learns from interaction beats a fixed topology at equal parameters and equal compute. This repository is the harness to answer that question.

---

## System Architecture

```
                         TEXT TOKENS
                              |
                              v
              +-------------------------------+
              |      FROZEN FOUNDATION        |
              |   Pythia 160M, weights locked |
              |   fp16, no gradient path      |
              +-------------------------------+
                              |
                              | hidden states h[8..11]
              +---------------+----------------+
              |               |                |
              v               v                v
        +-----------+   +-----------+    +-----------+
        |  ROUTER   |   |  ROUTER   |    |  ROUTER   |
        |  linear   |   |  linear   |    |  linear   |
        |  + Hebbian|   |  + Hebbian|    |  + Hebbian|
        |  prior    |   |  prior    |    |  prior    |
        +-----+-----+   +-----+-----+    +-----+-----+
              |               |                |
              |  top k softmax selection      |
              v               v                v
        +-----------+   +-----------+    +-----------+
        | EXPERT    |   | EXPERT    |    | EXPERT    |
        | CLUSTER   |   | CLUSTER   |    | CLUSTER   |
        | 8 MLPs    |   | 8 MLPs    |    | 8 MLPs    |
        +-+-+---+---+-   +-+-+---+---+-  +-+-+---+---+-+
          | |   |   |       | |   |   |      | |   |   |
          +=====+===+=======+=====+===+======+=====+===+
              message passing over small world
              adjacency, only active clusters
                              |
                              | weighted sum of top k
                              | expert outputs
                              v
              +-------------------------------+
              |  FINAL HIDDEN (base + deltas) |
              +---------------+---------------+
                              |
                              v
              +-------------------------------+
              |  LM HEAD (frozen)             |
              +-------------------------------+
                              |
                              v
                            LOGITS
```

---

## Components

### Frozen Foundation (`base.py`)

* Loads Pythia 160M from a local path or the Hugging Face hub.
* Every base parameter is frozen with `requires_grad = False`.
* Runs in fp16 to halve memory traffic on CPU devices.
* Hidden states are collected per layer via `output_hidden_states`.

### Sparse Expert Clusters (`experts.py`)

* One `ExpertBlock` per attached base layer, currently the last four layers.
* Each block holds eight expert MLPs with GELU activation.
* The block reads the hidden state of its attached layer and emits a residual delta added to the final hidden state.
* No gradient ever flows back into the base model.

### Routing and Load Balancing

* A lightweight linear projection scores every expert for every token.
* Top k selection with k = 2 activates only a small fraction of the network.
* An auxiliary loss in the style of Switch Transformer prevents router collapse:

```
L_aux = N * sum_c( f_c * P_c )

f_c  fraction of tokens routed to expert c
P_c  mean router probability for expert c
N    number of experts
```

The total training objective is:

```
L = L_lm + lambda * L_aux
```

### Neighborhood Graph (`graph.py`)

* `small_world` builds a ring lattice with random long range shortcuts, producing a topology with dense local structure and sparse global reach.
* `MessagePassing` carries edge weights as learnable parameters masked by the adjacency matrix.
* For each message round, every active expert mixes a fraction of its neighbors' outputs:

```
h_i <- h_i + beta * W[i][j] * h_j    for all active neighbors j of i
```

* Message passing touches only active clusters, so sparsity is preserved.

### Hebbian Plasticity

* `HebbianBank` tracks co activation statistics between experts.
* Each batch strengthens edges between experts that fire together and applies exponential decay:

```
E = E * decay
E[i][j] += lr  for every co activated pair (i, j)
```

* The bank produces a routing prior that biases future selections toward clusters that historically cooperate.
* This is pure local plasticity with no global gradient, cheap enough to run during inference on device.

### Memory Hierarchy (`memory.py`)

The SQLite store acts as the hippocampus of the system.

```
      +------------------------------+
      |  HOT WEIGHTS in RAM          |   active experts
      +--------------+---------------+
                     | prefetch on router decision
      +--------------+---------------+
      |  WARM WEIGHTS in SQLite      |   expert_weights table
      |  pickled tensors as blobs    |
      +--------------+---------------+
                     |
      +--------------+---------------+
      |  ACTIVATION MEMORIES         |   memories table
      |  embeddings plus text        |
      |  cosine similarity retrieval |
      +------------------------------+
```

* Expert weights are serialized and offloaded to SQLite, enabling a small memory footprint on device.
* Activation memories pair hidden state embeddings with their text, retrievable by cosine similarity.
* SQLite is in process and mmap friendly, so small queries cost microseconds.

### Training Pipeline (`train.py`)

```
  text corpus
      |
      v
  tokenize once, split train / validation
      |
      v
  batch blocks  -->  frozen base forward  -->  router  -->  top k experts
                                                              |
      +-------------------------------------------------------+
      |
      v
  residual deltas added to final hidden --> LM head --> logits
      |
      v
  L = CE loss + lambda * load balance aux loss
      |
      v
  backward through experts and router only --> AdamW step
      |
      v
  Hebbian bank update from co activations
      |
      v
  periodic validation perplexity vs frozen base
      |
      v
  checkpoint: experts, router, message passing weights
      |
      v
  expert weights offloaded to SQLite
```

---

## Quick Start

```
pip install torch transformers numpy
```

Train the flat sparse baseline:

```
python -m omni.train --steps 2000 --base EleutherAI/pythia-160m-deduped --out runs/flat
```

Train with the neighborhood graph:

```
python -m omni.train --graph-rounds 1 --steps 2000 --out runs/graph
```

Train with Hebbian plasticity:

```
python -m omni.train --hebbian-lr 0.05 --steps 2000 --out runs/hebbian
```

Evaluate any run:

```
python -m omni.eval --run runs/flat
```

---

## Configuration

| Parameter | Default | Meaning |
|---|---|---|
| base_model | pythia 160m | frozen foundation |
| n_expert_layers | 4 | number of attached base layers |
| n_experts | 8 | experts per block |
| top_k | 2 | experts active per token |
| mid_dim | 256 | expert hidden width |
| graph_rounds | 0 | message passing rounds, 0 disables graph |
| beta | 0.5 | neighbor mixing strength |
| hebbian_lr | 0.0 | co activation learning rate, 0 disables |
| hebbian_decay | 0.999 | edge strength decay |
| aux_coef | 0.01 | load balancing weight |
| delta_scale | 0.02 | residual delta multiplier |
| seq_len | 256 | context length |
| batch_size | 4 | tokens per step |
| lr | 1e-4 | AdamW learning rate |
| grad_clip | 1.0 | gradient clipping |

---

## Experiments

The core ablation matrix isolates each mechanism at equal parameter count and equal compute.

| Run | graph_rounds | hebbian_lr | Question answered |
|---|---|---|---|
| flat | 0 | 0.0 | Does sparse expansion beat the frozen base? |
| graph | 1 | 0.0 | Does learned topology beat fixed topology? |
| hebbian | 0 | 0.05 | Does local plasticity help routing? |
| full | 1 | 0.05 | Do the mechanisms compose? |

Every run reports validation perplexity for the frozen base and for Omni, plus expert utilization, aux loss, and the strongest Hebbian edges.

---

## Test Runs and Results

All runs below were executed on a Google Colab T4 GPU with the public notebook at `notebooks/omni_ablation.ipynb`. The harness was validated through three staged gates before any result was trusted, and each gate caught a real defect.

### Harness Validation

1. The first run exposed a shift error in the language modeling loss. Every prediction was scored one token too late, so training chased an impossible target and perplexity exploded to millions.
2. After the fix, the frozen base perplexity landed at 34.82, confirming the metric was correct.
3. The expansion still diverged. Random initialized experts emitted deltas at the same magnitude as the hidden state itself, corrupting the frozen logits. Zero initialization of the expert output layers fixed the start.
4. The final defect was the training regime. Adam at lr 3e-4 overshot the sharp loss landscape of the frozen head. A stability sweep over delta scale and learning rate identified the stable regime: delta scale 0.02 with lr 1e-4.

Every run is fully deterministic for a fixed seed. Two sweep configurations reproduced byte identical numbers across separate runs.

### Sanity Gate

Sixty step run with the default configuration.

```
frozen base ppl (1 block): 34.82
step    10 loss 3.943 aux 5.426 usage 0.88
step    20 loss 3.783 aux 4.631 usage 1.00
step    30 loss 3.980 aux 6.462 usage 1.00
step    40 loss 3.514 aux 7.141 usage 1.00
step    50 loss 3.785 aux 7.509 usage 1.00
step    60 loss 3.670 aux 7.987 usage 1.00
```

The training loss dipped below the frozen base nll of 3.551 at step 40, the first evidence that the expansion learns real corrections. Routing utilization is full and no divergence appears.

### Stability Sweep

Five configurations at 120 steps each.

| Config | Base ppl | Final loss |
|---|---|---|
| scale005_lr1e4 | 34.82 | 3.922 |
| scale002_lr1e4 | 34.82 | 3.844 |
| scale010_lr1e4 | 34.82 | 4.168 |
| scale005_lr3e5 | 34.82 | 3.760 |
| scale005_lr1e4_aux01 | 34.82 | 3.979 |

The best configs are scale005_lr3e5 at 3.760 and the chosen default scale002_lr1e4 at 3.844. Learning rate 3e-4 diverges at every scale and is rejected.

### Ablation Matrix

Four configurations at 400 steps each, evaluated on 100 held out blocks.

| Run | Base ppl | Omni ppl | Delta |
|---|---|---|---|
| flat | 37.65 | 128.40 | 90.75 |
| graph | 37.65 | 76.13 | 38.48 |
| hebbian | 37.65 | 129.95 | 92.30 |
| full | 37.65 | 71.81 | 34.16 |

Findings:

* The neighborhood graph beats flat routing by 52 perplexity points, a 40 percent gap. Local message passing between active clusters is not decoration; it recovers most of the gap to the frozen base.
* Combining the graph with Hebbian plasticity gives the best result of the matrix. The mechanisms compose.
* Hebbian plasticity alone matches flat routing. It neither helps nor harms at this scale.
* No configuration crossed the frozen base at 400 steps. The expansion recovers most of the damage but remains net negative. Longer runs are the next experiment.

---

## On Device Deployment

The system was developed and validated on a phone running Ubuntu under proot on aarch64.

* Training runs single threaded. Multi thread OpenMP deadlocks under proot on this device.
* A single training step at 4 x 128 tokens takes about 250 seconds on the phone, roughly 100 times slower than a laptop.
* The frozen base in fp16 plus a 12.6M parameter trainable surface fits comfortably in 1.8 GB of available RAM.
* Expert offload to SQLite is the mechanism that lets the model exceed physical RAM limits.
* Hebbian plasticity is local and gradient free, so it can run during inference and adapt the model to its user.

The development loop is: train and run ablations on a laptop, then ship the checkpoint and memory store to the phone.

---

## Repository Layout

```
omni/
  base.py       frozen foundation loader
  config.py     dataclass configuration
  dataset.py    tokenized block dataset
  experts.py    expert clusters and routing
  graph.py      small world topology, message passing, Hebbian bank
  memory.py     SQLite weight and memory store
  model.py      OmniModel composition
  train.py      training harness
  eval.py       perplexity, memory and Hebbian inspection
data/           corpora
models/         local base model cache
runs/           checkpoints, configs, memory stores
```

---

## Roadmap

* Longer 2000 step runs to cross the frozen base perplexity
* Quantized inference for the frozen base
* Prefetch scheduler for expert and memory loading
* Retrieval injection into message passing
* Export to a minimal inference engine for phones
* Continual learning protocol with forgetting metrics
