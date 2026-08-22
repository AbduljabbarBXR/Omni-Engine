# Omni Engine: A Sparse Residual Expansion on a Frozen Language Model

## Abstract

A small frozen pretrained language model can be meaningfully improved by attaching a learnable sparse expansion. We attach expert clusters that correct the frozen hidden states under a magnitude penalty, with a router waking only a fraction of experts per token. On two unrelated domains the expansion beats the frozen base by 16 to 32 percent in perplexity, reproducibly across seeds. We also examine local communication between active experts and find its benefit confined to a data scarce regime. The full pipeline is deterministic, documented, and reproducible on free cloud GPUs.

## 1. Introduction

Frozen pretrained models are attractive for deployment: their weights are fixed, so they are safe to ship and cheap to run. Their weakness is that they cannot improve on their own domain. This work asks a simple question: can a frozen model be improved by attaching a small learnable surface that corrects its own representations, without touching the frozen weights?

We answer yes. The recipe has four parts. First, a frozen base model provides the structural language capability. Second, learnable expert clusters attach to its later layers and emit residual corrections. Third, a router wakes only a small fraction of experts per token, keeping compute low. Fourth, a magnitude penalty on the corrections keeps training stable over extended runs. This last piece is the quiet key: without it, corrections grow without bound and the expansion destroys the model.

The same architecture supports a biological framing. The frozen base is the innate wiring. The experts are the synapses that grow where they are used. The router is the selective firing. A Hebbian rule and a persistent memory store extend the system toward on device adaptation, which is the eventual goal: a model that improves through use on hardware with little memory.

## 2. Method

### 2.1 The frozen foundation

We use Pythia 160M, a fully pretrained autoregressive transformer. Every weight is frozen. The model runs in half precision to halve memory traffic. Hidden states are collected per layer through the standard output interface.

### 2.2 Expert clusters

Four expert blocks attach to the last four layers of the base. Each block holds eight expert networks, each a small multilayer perceptron. The block reads the hidden state of its attached layer and emits a residual delta added to the final hidden state.

The output layers of every expert are initialized to zero. This is essential: the model must start exactly equal to the frozen base and grow only corrections that earn their place. Random initialization emits corrections as large as the hidden state itself and training cannot recover from that noise.

### 2.3 Routing and load balancing

A lightweight linear router scores every expert for every token. The top two experts are selected and their outputs combined with normalized weights. The remaining six stay dormant, so only a fraction of the network computes per token.

A load balancing penalty in the style of Switch Transformer prevents the router from collapsing onto a single expert. The auxiliary loss is the product of routed token fractions and mean router probabilities, summed over experts and scaled by the expert count.

### 2.4 The delta penalty

The total training objective is:

```
L = L_lm + lambda_aux * L_aux + lambda_pen * sum( delta squared )
```

The magnitude penalty on the deltas creates an equilibrium: a correction survives only if it reduces the language loss enough to justify its size. Without this term, corrections grow without bound. Optimizer momentum pushes them in one consistent direction each step, because the frozen features are fixed, and nothing else holds them back. The result is catastrophic divergence at extended horizons. With the penalty, deltas stay bounded and training is stable to 2000 steps and beyond.

### 2.5 The neighborhood graph

We also test local communication between experts. A small world graph arranges the eight experts of a block in a ring with a few distant shortcuts. After the selected experts produce their outputs, they exchange weighted messages with their neighbors over one round. Edge weights are learnable in the primary configuration.

Two control configurations are tested: fixed random edge weights, and plain pooling with equal neighbor weights. These isolate whether any interaction helps, or whether learning the edges matters.

### 2.6 Training regime

The regime was chosen by a stability sweep, not by preference. Learning rate 3e-4 diverges at every delta scale and was rejected. The stable regime is learning rate 1e-4 with delta scale 0.02. The optimizer is AdamW. Every run is deterministic: fixed seeds, fixed data order, fixed evaluation.

## 3. Results

### 3.1 Shakespeare at 2000 steps

All configurations evaluated on 100 held out blocks.

| Run | Base ppl | Omni ppl | Delta |
|---|---|---|---|
| flat | 37.31 | 31.38 | -5.93 |
| graph | 37.31 | 31.38 | -5.93 |
| hebbian | 37.31 | 31.37 | -5.94 |
| full | 37.31 | 31.36 | -5.95 |

### 3.2 WikiText at 2000 steps

The same configurations on a second domain, evaluated on the dataset's own held out validation file.

| Run | Base ppl | Omni ppl | Delta |
|---|---|---|---|
| flat | 49.44 | 33.65 | -15.79 |
| graph | 49.44 | 33.63 | -15.81 |
| hebbian | 49.44 | 33.64 | -15.80 |
| full | 49.44 | 33.64 | -15.80 |

### 3.3 Stability

Training is stable in every run: loss descends below the base nll, delta magnitude stays bounded near 0.17 on Shakespeare and 0.09 on WikiText, routing utilization stays full, and validation perplexity descends to the last step.

### 3.4 The firming runs

One additional kernel answered three open questions on WikiText.

| Run | Steps | Base ppl | Omni ppl | Delta |
|---|---|---|---|---|
| flat | 400 | 49.44 | 35.73 | -13.71 |
| graph | 400 | 49.44 | 35.76 | -13.68 |
| graph fixed edges | 400 | 49.44 | 35.90 | -13.54 |
| graph pooling | 400 | 49.44 | 35.82 | -13.62 |
| full seed 7 | 2000 | 49.44 | 33.67 | -15.77 |
| full seed 42 | 2000 | 49.44 | 33.64 | -15.80 |

### 3.5 Generation

The trained model generates coherent dialogue in the target register. From the prompt "To be, or not to be", the model continues:

```
, And if you have no time for my woes,
Take my leave, and if you have no time,
I'll leave you be.

SIR JOHN EDWARD:
Good night, dear.

GARY:
Good night, good night.
```

The model learned play structure itself: character headers, speech, and poetic diction.

## 4. Analysis

### 4.1 The crossing generalizes

Every configuration beats the frozen base on both domains. Two unrelated corpora, two crossings, reproducible to within 0.03 perplexity points across seeds.

### 4.2 The expansion helps most where the base is weakest

On Shakespeare the base scored 37.31 and the improvement was 16 percent. On WikiText the base scored 49.44 and the improvement was 32 percent. The expansion corrects more when the frozen model struggles more. A frozen model has a ceiling; the expansion raises it, and raises it most where the ceiling is lowest.

### 4.3 The graph is a data scarce effect

At 400 steps on Shakespeare, the graph beat flat routing by 52 perplexity points. At 400 steps on WikiText, flat and graph tie. The convergence speed advantage does not generalize. The likely mechanism is gradient sharing: in the graph, an awake expert passes its output through its neighbor's edge, so each expert receives gradient signal from tokens where its neighbor was selected. On a small corpus, where every expert starves for samples, that sharing rescues learning speed. On a corpus six times larger, experts are fed and the sharing stops mattering. The learned edge ordering (learned below pooling below fixed) points the right way but the gaps are within noise, so we make no claim about edge learning.

### 4.4 Determinism

Seed 7 reproduces seed 42 within 0.03 perplexity points on the full configuration. Every run in this report is reproducible from the repository.

## 5. Limitations

The base model is 160M parameters and the corpora are small. The graph mechanism is not fully isolated: the data scarcity hypothesis is a hypothesis, not a measurement. Hebbian plasticity is neutral at this scale, neither helping nor harming. The generation demo is qualitative. The expansion is a correction surface, not new world knowledge; it improves how the base uses what it has.

## 6. Reproducibility

The repository contains the full harness, the notebook, the kernel definitions, and the evaluation scripts. Every configuration is a command line flag. The Kaggle kernel runs the complete pipeline unattended on a free GPU and returns the console output. All results in this report came from that pipeline.

## 7. Future work

The next direction is an output harness: a second set of experts that read the final prediction distribution and correct it directly, with a sparse bridge from the input side. This is the closest buildable form of self correction: the model watches its own decisions and learns to fix them. A governor layer with a persistent memory store is the second direction. Scaling to a 0.5B base is the third.

## 8. Citation

If you use this work, cite the repository:

```
@misc{omniengine2026,
  title = {Omni Engine: A Sparse Residual Expansion on a Frozen Language Model},
  author = {Abduljabbar},
  year = {2026},
  howpublished = {https://github.com/AbduljabbarBXR/Omni-Engine}
}
```

## Appendix: the pipeline

The development pipeline is itself part of the contribution. A sanity gate runs before any result is trusted, and it caught three defects in the early harness: a shift error in the loss, a scale error in the expert initialization, and a learning rate regime error. Each defect is documented in the project journal with its fix. The journal is the written memory of the project and is kept local to the authors.
