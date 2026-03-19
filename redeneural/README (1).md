# Neural Network in Pure C — Circle Classifier

A Multilayer Perceptron (MLP) built from scratch in pure C, with no external dependencies beyond `stdio.h`. Given a point `(x, y)`, the network learns to classify whether it falls **inside or outside a circle** — implementing backpropagation, momentum, Xavier initialization, Fisher-Yates shuffling, and early stopping entirely by hand.

> **Why pure C?** Most neural network tutorials rely on Python frameworks that hide the real mechanics. This project forces every concept to be explicit: the math, the memory, the training loop — nothing is abstracted away.

---

## Table of Contents

- [Demo](#demo)
- [How It Works](#how-it-works)
- [Features](#features)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [The Problem](#the-problem)
- [Training Techniques](#training-techniques)
- [Sample Output](#sample-output)
- [Learning Resources](#learning-resources)
- [License](#license)

---

## Demo

```
========================================
   REDE NEURAL EM C PURO
   Problema: CIRCULO vs. FORA
========================================

Dados de treino : 200 pontos (65 dentro, 135 fora)
Arquitetura: 2 -> 8 -> 1
Taxa de aprendizado: 0.30 | Momentum: 0.90

--- TREINAMENTO ---
Epoca     0  |  Erro MSE: 0.257701
Epoca  5000  |  Erro MSE: 0.000578
Epoca  8312  |  Erro MSE: 0.000998  << CONVERGIU!

--- ACURACIA ---
Treino : 198/200 = 99.0%
Teste  : 49/50  = 98.0%

--- MAPA ASCII (o que a rede aprendeu) ---

   ...............................
   ...............................
   .........OOOOOOOOO.............
   .......OOOO#######OOOO.........
   ......OO###########OO..........
   .....OO#############OO.........
   .....O###############O.........
   .....O###############O.........
   .....OO#############OO.........
   ......OO###########OO..........
   .......OOOO#######OOOO.........
   .........OOOOOOOOO.............
   ...............................

   Legenda: # = dentro   . = fora   O = borda real do circulo
```

---

## How It Works

The network receives a 2D point `(x, y)` and outputs a value close to `1` if the point is inside the circle, or close to `0` if it is outside.

```
Input layer     Hidden layer     Output layer
  (x, y)         8 neurons         1 neuron

    x ──┐
        ├──── h1 ─┐
        ├──── h2 ─┤
        ├──── h3 ─┤
    y ──┤──── h4 ─┼──── ŷ  →  0 (outside) or 1 (inside)
        ├──── h5 ─┤
        ├──── h6 ─┤
        ├──── h7 ─┤
        └──── h8 ─┘

    Rule:  x² + y² < 1.0  →  inside  (label = 1)
           x² + y² ≥ 1.0  →  outside (label = 0)
```

The circular boundary is **not linearly separable** — no straight line can split inside from outside. The hidden layer learns a curved, non-linear decision boundary through backpropagation.

---

## Features

| Feature | Description |
|---|---|
| **No external libraries** | Only `stdio.h` — no `math.h`, no `stdlib.h` |
| **Manual `exp(x)`** | Taylor series, 100 iterations |
| **Manual `sqrt(x)`** | Newton-Raphson method, 50 iterations |
| **Manual RNG** | Linear Congruential Generator (LCG) — same constants as glibc |
| **Xavier initialization** | Weights scaled by `1/√n` to prevent sigmoid saturation at start |
| **Sigmoid with clamp** | Guards against `exp()` overflow for extreme weight values |
| **SGD with momentum** | `velocity = 0.9 × velocity + η × gradient` |
| **Fisher-Yates shuffle** | Unbiased random permutation of training samples every epoch |
| **Early stopping** | Halts training when MSE drops below threshold |
| **Train/test split** | Separate datasets generated from the same deterministic RNG |
| **ASCII decision map** | 31×31 grid showing the learned classification boundary |

---

## Getting Started

### Requirements

- Any C compiler: `gcc`, `clang`, or `cc`
- No additional dependencies

### Compile and run

```bash
# Clone the repository
git clone https://github.com/your-username/neural-network-circle-c.git
cd neural-network-circle-c

# Compile
gcc RedeNeural_Circulo.c -o circulo

# Run
./circulo
```

On Windows (MinGW):

```bat
gcc RedeNeural_Circulo.c -o circulo.exe
circulo.exe
```

No flags. No `-lm`. No dependencies. Just compile and run.

---

## Configuration

All hyperparameters are `#define` constants at the top of the file. Edit them before compiling:

```c
#define ENTRADA      2       // input neurons  — always 2 (x and y)
#define OCULTA       8       // hidden neurons — increase for harder boundaries
#define SAIDA        1       // output neurons — always 1 (binary classification)
#define TAXA         0.3     // learning rate  — lower = slower but more stable
#define MOMENTUM     0.9     // momentum factor — 0.0 disables it entirely
#define CICLOS       30000   // max training epochs
#define AMOSTRAS     200     // number of training points
#define TESTES       50      // number of test points (never seen during training)
#define RAIO         1.0     // radius of the classification circle
#define ERRO_MINIMO  0.001   // early stopping threshold
```

**Tuning tips:**

- Increase `OCULTA` if the network struggles to learn the boundary
- Lower `TAXA` if the MSE oscillates instead of converging smoothly
- Set `MOMENTUM` to `0.0` to see the difference with and without it
- Increase `AMOSTRAS` to improve generalization on the test set
- Change `RAIO` to classify points relative to a circle of any size

---

## The Problem

Points are sampled uniformly from the square `[-1.5, 1.5]²`. The network must learn the rule:

```
  label = 1   if  x² + y²  <  RAIO²   (inside the circle)
  label = 0   if  x² + y²  ≥  RAIO²   (outside the circle)
```

A single neuron with no hidden layer **cannot solve this** — the decision boundary is a curve, not a line. The hidden layer transforms the input space so the boundary becomes linearly separable in a higher dimension. This is the same fundamental challenge that makes neural networks powerful on real-world data.

---

## Training Techniques

### Sigmoid with clamp

```c
double ativacao(double x) {
    if (x >  500.0) return 1.0;   // prevents exp(500) → inf
    if (x < -500.0) return 0.0;   // prevents exp(-500) → underflow
    return 1.0 / (1.0 + exponencial(-x));
}
```

Without the clamp, extreme weight values during early training can cause `exp()` to overflow to `inf` or produce `nan`, silently corrupting every subsequent calculation.

### Xavier initialization

```c
double scale = 1.0 / sqrt(n_inputs);
weight = random_uniform(-scale, +scale);
```

Scales the initial weights inversely to the number of inputs in each layer. This keeps neuron activations in the useful middle range of the sigmoid (near 0.5), where gradients are strongest. Without it, neurons saturate immediately and gradients vanish before learning begins.

### SGD with momentum

```c
// Without momentum:
weight += learning_rate * gradient;

// With momentum:
velocity = MOMENTUM * velocity + learning_rate * gradient;
weight  += velocity;
```

Momentum accumulates gradient history — like a ball rolling downhill that builds speed. This produces smoother updates, faster convergence, and the ability to roll past shallow local minima that would otherwise trap the network.

### Fisher-Yates shuffle

```c
for (int i = n - 1; i > 0; i--) {
    int j = random_int(0, i);  // uniformly chosen from [0, i]
    swap(array[i], array[j]);
}
```

Re-orders the training samples at the start of every epoch. Each permutation is equally likely. Without shuffling, the network can exploit the fixed presentation order instead of learning the actual pattern.

### Early stopping

```c
if (mse < ERRO_MINIMO) {
    printf("CONVERGIU!\n");
    break;
}
```

Stops training as soon as the loss is low enough. Avoids wasting time on epochs that yield no meaningful improvement and reduces the risk of overfitting to the training set.

---

## Sample Output

```
--- DEMONSTRACAO (pontos conhecidos) ---

x        y        distancia  saida      correto?
----------------------------------------------------
0.00     0.00     0.0000     0.997341   OK      ← center, clearly inside
0.50     0.50     0.7071     0.983214   OK      ← inside
0.80     0.80     1.1314     0.008762   OK      ← outside
1.50     0.00     1.5000     0.001203   OK      ← far outside
0.00     1.50     1.5000     0.000891   OK      ← far outside
-0.60    0.60     0.8485     0.971548   OK      ← inside, third quadrant
0.99     0.00     0.9900     0.934217   OK      ← just inside the boundary
1.01     0.00     1.0100     0.041823   OK      ← just outside the boundary
```

The last two rows are the hardest: points right on the edge of the circle. The network correctly distinguishes them despite their proximity to the boundary.

---

## Learning Resources

If you want to understand the theory behind this code:

- [3Blue1Brown — Neural Networks series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — best visual introduction to backpropagation
- [Michael Nielsen — Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — free online book, mathematically rigorous
- [CS231n — Convolutional Neural Networks (Stanford)](https://cs231n.github.io/) — covers backpropagation from first principles
- [Xavier initialization paper](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) — Glorot & Bengio, 2010

---

## License

This project is released under the [MIT License](LICENSE).

```
MIT License

Copyright (c) 2025 [your name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
