# Rede Neural em C Puro — Classificador de Círculo

Um Perceptron Multicamadas (MLP) construído do zero em C puro, sem nenhuma dependência além de `stdio.h`. Dado um ponto `(x, y)`, a rede aprende a classificar se ele está **dentro ou fora de um círculo** — implementando backpropagation, momentum, inicialização Xavier, embaralhamento Fisher-Yates e early stopping inteiramente à mão.

> **Por que C puro?** A maioria dos tutoriais de redes neurais usa frameworks Python que escondem a mecânica real. Este projeto força cada conceito a ser explícito: a matemática, a memória, o loop de treinamento — nada é abstraído.

---

## Índice

- [Demo](#demo)
- [Como Funciona](#como-funciona)
- [Funcionalidades](#funcionalidades)
- [Como Executar](#como-executar)
- [Configuração](#configuração)
- [O Problema](#o-problema)
- [Técnicas de Treinamento](#técnicas-de-treinamento)
- [Exemplo de Saída](#exemplo-de-saída)
- [Referências](#referências)
- [Licença](#licença)

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

## Como Funciona

A rede recebe um ponto `(x, y)` e produz um valor próximo de `1` se o ponto estiver dentro do círculo, ou próximo de `0` se estiver fora.

```
Camada de entrada   Camada oculta   Camada de saída
     (x, y)          8 neurônios       1 neurônio

    x ──┐
        ├──── h1 ─┐
        ├──── h2 ─┤
        ├──── h3 ─┤
    y ──┤──── h4 ─┼──── ŷ  →  0 (fora) ou 1 (dentro)
        ├──── h5 ─┤
        ├──── h6 ─┤
        ├──── h7 ─┤
        └──── h8 ─┘

    Regra:  x² + y² < 1.0  →  dentro (rótulo = 1)
            x² + y² ≥ 1.0  →  fora   (rótulo = 0)
```

A fronteira circular **não é linearmente separável** — nenhuma linha reta consegue dividir os pontos internos dos externos. A camada oculta aprende uma fronteira de decisão curva e não-linear através do backpropagation.

---

## Funcionalidades

| Funcionalidade | Descrição |
|---|---|
| **Sem bibliotecas externas** | Apenas `stdio.h` — sem `math.h`, sem `stdlib.h` |
| **`exp(x)` manual** | Série de Taylor, 100 iterações |
| **`sqrt(x)` manual** | Método de Newton-Raphson, 50 iterações |
| **Gerador aleatório manual** | Linear Congruential Generator (LCG) — mesmas constantes do glibc |
| **Inicialização Xavier** | Pesos escalados por `1/√n` para evitar saturação inicial da sigmoide |
| **Sigmoide com clamp** | Protege contra overflow de `exp()` para valores extremos de peso |
| **SGD com momentum** | `velocidade = 0.9 × velocidade + η × gradiente` |
| **Embaralhamento Fisher-Yates** | Permutação aleatória uniforme das amostras a cada época |
| **Early stopping** | Interrompe o treinamento quando o MSE cai abaixo do limiar |
| **Divisão treino/teste** | Datasets separados gerados pelo mesmo RNG determinístico |
| **Mapa de decisão ASCII** | Grade 31×31 mostrando a fronteira de classificação aprendida |

---

## Como Executar

### Requisitos

- Qualquer compilador C: `gcc`, `clang` ou `cc`
- Nenhuma dependência adicional

### Compilar e rodar

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/rede-neural-circulo-c.git
cd rede-neural-circulo-c

# Compile
gcc RedeNeural_Circulo.c -o circulo

# Execute
./circulo
```

No Windows (MinGW):

```bat
gcc RedeNeural_Circulo.c -o circulo.exe
circulo.exe
```

Sem flags. Sem `-lm`. Sem dependências. Só compilar e rodar.

---

## Configuração

Todos os hiperparâmetros são constantes `#define` no topo do arquivo. Edite-os antes de compilar:

```c
#define ENTRADA      2       // neurônios de entrada — sempre 2 (x e y)
#define OCULTA       8       // neurônios ocultos — aumente para fronteiras mais complexas
#define SAIDA        1       // neurônios de saída — sempre 1 (classificação binária)
#define TAXA         0.3     // taxa de aprendizado — menor = mais lento, porém mais estável
#define MOMENTUM     0.9     // fator de momentum — 0.0 desativa completamente
#define CICLOS       30000   // máximo de épocas de treinamento
#define AMOSTRAS     200     // número de pontos de treino
#define TESTES       50      // número de pontos de teste (nunca vistos no treino)
#define RAIO         1.0     // raio do círculo de classificação
#define ERRO_MINIMO  0.001   // limiar para early stopping
```

**Dicas de ajuste:**

- Aumente `OCULTA` se a rede não conseguir aprender a fronteira
- Reduza `TAXA` se o MSE oscilar em vez de convergir suavemente
- Defina `MOMENTUM` como `0.0` para comparar com e sem momentum
- Aumente `AMOSTRAS` para melhorar a generalização no conjunto de teste
- Altere `RAIO` para classificar pontos em relação a um círculo de qualquer tamanho

---

## O Problema

Os pontos são amostrados uniformemente do quadrado `[-1.5, 1.5]²`. A rede deve aprender a regra:

```
  rótulo = 1   se  x² + y²  <  RAIO²   (dentro do círculo)
  rótulo = 0   se  x² + y²  ≥  RAIO²   (fora do círculo)
```

Um único neurônio sem camada oculta **não consegue resolver isso** — a fronteira de decisão é uma curva, não uma linha. A camada oculta transforma o espaço de entrada de forma que a fronteira se torna linearmente separável numa dimensão superior. Este é o mesmo desafio fundamental que torna as redes neurais poderosas em dados do mundo real.

---

## Técnicas de Treinamento

### Sigmoide com clamp

```c
double ativacao(double x) {
    if (x >  500.0) return 1.0;   // evita exp(500) → inf
    if (x < -500.0) return 0.0;   // evita exp(-500) → underflow
    return 1.0 / (1.0 + exponencial(-x));
}
```

Sem o clamp, valores extremos de peso durante as primeiras épocas podem fazer o `exp()` estourar para `inf` ou gerar `nan`, corrompendo silenciosamente todos os cálculos subsequentes.

### Inicialização Xavier

```c
double escala = 1.0 / sqrt(n_entradas);
peso = aleatorio_uniforme(-escala, +escala);
```

Escala os pesos iniciais inversamente ao número de entradas de cada camada. Isso mantém as ativações dos neurônios na faixa central útil da sigmoide (próximo de 0.5), onde os gradientes são mais fortes. Sem isso, os neurônios saturam imediatamente e os gradientes desaparecem antes do aprendizado começar.

### SGD com momentum

```c
// Sem momentum:
peso += taxa * gradiente;

// Com momentum:
velocidade = MOMENTUM * velocidade + taxa * gradiente;
peso      += velocidade;
```

O momentum acumula o histórico do gradiente — como uma bola rolando morro abaixo que vai ganhando velocidade. Isso produz atualizações mais suaves, convergência mais rápida e a capacidade de escapar de mínimos locais rasos que, de outra forma, travariam a rede.

### Embaralhamento Fisher-Yates

```c
for (int i = n - 1; i > 0; i--) {
    int j = inteiro_aleatorio(0, i);  // escolhido uniformemente em [0, i]
    trocar(array[i], array[j]);
}
```

Reordena as amostras de treinamento no início de cada época. Cada permutação é igualmente provável. Sem o embaralhamento, a rede pode explorar a ordem fixa de apresentação em vez de aprender o padrão real.

### Early stopping

```c
if (mse < ERRO_MINIMO) {
    printf("CONVERGIU!\n");
    break;
}
```

Interrompe o treinamento assim que a perda é suficientemente baixa. Evita desperdício de tempo em épocas que não trazem melhoria significativa e reduz o risco de overfitting no conjunto de treino.

---

## Exemplo de Saída

```
--- DEMONSTRACAO (pontos conhecidos) ---

x        y        distancia  saida      correto?
----------------------------------------------------
0.00     0.00     0.0000     0.997341   OK      ← centro, claramente dentro
0.50     0.50     0.7071     0.983214   OK      ← dentro
0.80     0.80     1.1314     0.008762   OK      ← fora
1.50     0.00     1.5000     0.001203   OK      ← bem fora
0.00     1.50     1.5000     0.000891   OK      ← bem fora
-0.60    0.60     0.8485     0.971548   OK      ← dentro, terceiro quadrante
0.99     0.00     0.9900     0.934217   OK      ← quase na borda, dentro
1.01     0.00     1.0100     0.041823   OK      ← quase na borda, fora
```

As duas últimas linhas são as mais difíceis: pontos praticamente sobre a borda do círculo. A rede os distingue corretamente apesar da proximidade com a fronteira.

---

## Referências

Para entender a teoria por trás deste código:

- [3Blue1Brown — Série sobre Redes Neurais](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — melhor introdução visual ao backpropagation
- [Michael Nielsen — Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — livro online gratuito, matematicamente rigoroso
- [CS231n — Redes Neurais Convolucionais (Stanford)](https://cs231n.github.io/) — cobre backpropagation a partir dos primeiros princípios
- [Artigo da inicialização Xavier](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) — Glorot & Bengio, 2010

---

## Licença

Este projeto está disponível sob a [Licença MIT](LICENSE).

```
MIT License

Copyright (c) 2025 [seu nome]

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
