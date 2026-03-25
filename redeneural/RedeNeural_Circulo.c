/*
 * Rede Neural em C puro
 * Implementacao de um Perceptron Multicamadas (MLP)
 * com retropropagacao (backpropagation)
 *
 * Problema: Classificar pontos em 2D como DENTRO ou FORA
 * de um circulo de raio 1.0 centrado na origem.
 *
 * Regra:
 *   x*x + y*y < 1.0  -->  saida = 1 (dentro)
 *   x*x + y*y >= 1.0 -->  saida = 0 (fora)
 *
 * Este e um problema nao-linearmente separavel: a fronteira
 * entre as classes e uma curva (circulo), nao uma reta.
 * Por isso, a camada oculta e obrigatoria.
 *
 * Arquitetura: 2 entradas -> 8 ocultos -> 1 saida
 * Funcoes matematicas implementadas manualmente (sem math.h).
 *
 * Compilar: gcc RedeNeural_Circulo.c -o RedeNeural_Circulo
 */

#include <stdio.h>

#define ENTRADA       2       /* x e y do ponto              */
#define OCULTA        8       /* neuronios na camada oculta  */
#define SAIDA         1       /* 1 = dentro, 0 = fora        */
#define TAXA          0.3     /* taxa de aprendizado          */
#define MOMENTUM      0.9     /* fator de inercia nos pesos  */
#define CICLOS        30000
#define AMOSTRAS      200     /* pontos gerados aleatorios   */
#define TESTES        50      /* pontos separados para teste */
#define RAIO          1.0     /* raio do circulo             */
#define ERRO_MINIMO   0.001   /* early stopping              */


static unsigned long semente = 987654321;

void setar_semente(unsigned long val) {
    semente = val;
}

double sortear(void) {
    semente = semente * 1103515245 + 12345;
    return (double)((semente >> 16) & 0x7FFF) / 32767.0;
}

double sortear_faixa(double margem) {
    return (sortear() * 2.0 - 1.0) * margem;
}

double exponencial(double x) {
    double soma = 1.0;
    double termo = 1.0;
    for (int k = 1; k <= 100; k++) {
        termo *= x / k;
        soma += termo;
    }
    return soma;
}

double ativacao(double x) {
    if (x > 500.0)  return 1.0;
    if (x < -500.0) return 0.0;
    return 1.0 / (1.0 + exponencial(-x));
}

double ativacao_deriv(double x) {
    return x * (1.0 - x);
}

double raiz(double n) {
    if (n < 0) return -1.0;
    if (n == 0) return 0.0;
    double r = n;
    for (int k = 0; k < 50; k++)
        r = 0.5 * (r + n / r);
    return r;
}

void embaralhar(int *v, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(sortear() * (i + 1));
        if (j > i) j = i;
        int tmp = v[i];
        v[i] = v[j];
        v[j] = tmp;
    }
}

void gerar_ponto(double *px, double *py, double *classe) {
    *px = sortear_faixa(1.5);
    *py = sortear_faixa(1.5);
    double d2 = (*px) * (*px) + (*py) * (*py);
    *classe = (d2 < RAIO * RAIO) ? 1.0 : 0.0;
}

typedef struct {
    double pesos_eo[ENTRADA][OCULTA];
    double pesos_os[OCULTA][SAIDA];
    double bias_o[OCULTA];
    double bias_s[SAIDA];
    double cam_entrada[ENTRADA];
    double cam_oculta[OCULTA];
    double cam_saida[SAIDA];
    double erro_s[SAIDA];
    double erro_o[OCULTA];
    double delta_s[SAIDA];
    double delta_o[OCULTA];
    double vel_eo[ENTRADA][OCULTA];
    double vel_os[OCULTA][SAIDA];
    double vel_bo[OCULTA];
    double vel_bs[SAIDA];
} Rede;

void iniciar_rede(Rede *r) {
    double fe = 1.0 / raiz((double)ENTRADA);
    double fo = 1.0 / raiz((double)OCULTA);

    for (int i = 0; i < ENTRADA; i++)
        for (int j = 0; j < OCULTA; j++)
            r->pesos_eo[i][j] = sortear_faixa(fe);

    for (int i = 0; i < OCULTA; i++)
        for (int j = 0; j < SAIDA; j++)
            r->pesos_os[i][j] = sortear_faixa(fo);

    for (int i = 0; i < OCULTA; i++)
        r->bias_o[i] = 0.0;

    for (int i = 0; i < SAIDA; i++)
        r->bias_s[i] = 0.0;

    for (int i = 0; i < ENTRADA; i++)
        for (int j = 0; j < OCULTA; j++)
            r->vel_eo[i][j] = 0.0;
    for (int i = 0; i < OCULTA; i++)
        for (int j = 0; j < SAIDA; j++)
            r->vel_os[i][j] = 0.0;
    for (int i = 0; i < OCULTA; i++)
        r->vel_bo[i] = 0.0;
    for (int i = 0; i < SAIDA; i++)
        r->vel_bs[i] = 0.0;
}

void propagar(Rede *r, double entrada[ENTRADA]) {
    for (int i = 0; i < ENTRADA; i++)
        r->cam_entrada[i] = entrada[i];

    for (int j = 0; j < OCULTA; j++) {
        double soma = r->bias_o[j];
        for (int i = 0; i < ENTRADA; i++)
            soma += r->cam_entrada[i] * r->pesos_eo[i][j];
        r->cam_oculta[j] = ativacao(soma);
    }

    for (int j = 0; j < SAIDA; j++) {
        double soma = r->bias_s[j];
        for (int i = 0; i < OCULTA; i++)
            soma += r->cam_oculta[i] * r->pesos_os[i][j];
        r->cam_saida[j] = ativacao(soma);
    }
}

void retropropagar(Rede *r, double alvo[SAIDA]) {
    for (int i = 0; i < SAIDA; i++) {
        r->erro_s[i] = alvo[i] - r->cam_saida[i];
        r->delta_s[i] = r->erro_s[i] * ativacao_deriv(r->cam_saida[i]);
    }

    for (int i = 0; i < OCULTA; i++) {
        r->erro_o[i] = 0.0;
        for (int j = 0; j < SAIDA; j++)
            r->erro_o[i] += r->delta_s[j] * r->pesos_os[i][j];
        r->delta_o[i] = r->erro_o[i] * ativacao_deriv(r->cam_oculta[i]);
    }

    for (int i = 0; i < OCULTA; i++)
        for (int j = 0; j < SAIDA; j++) {
            r->vel_os[i][j] = MOMENTUM * r->vel_os[i][j]
                + TAXA * r->delta_s[j] * r->cam_oculta[i];
            r->pesos_os[i][j] += r->vel_os[i][j];
        }

    for (int i = 0; i < SAIDA; i++) {
        r->vel_bs[i] = MOMENTUM * r->vel_bs[i]
            + TAXA * r->delta_s[i];
        r->bias_s[i] += r->vel_bs[i];
    }

    for (int i = 0; i < ENTRADA; i++)
        for (int j = 0; j < OCULTA; j++) {
            r->vel_eo[i][j] = MOMENTUM * r->vel_eo[i][j]
                + TAXA * r->delta_o[j] * r->cam_entrada[i];
            r->pesos_eo[i][j] += r->vel_eo[i][j];
        }

    for (int i = 0; i < OCULTA; i++) {
        r->vel_bo[i] = MOMENTUM * r->vel_bo[i]
            + TAXA * r->delta_o[i];
        r->bias_o[i] += r->vel_bo[i];
    }
}


int main(void) {
    Rede rede;

    printf("========================================\n");
    printf("   REDE NEURAL EM C PURO               \n");
    printf("   Problema: CIRCULO vs. FORA           \n");
    printf("========================================\n\n");

    setar_semente(42);

    double x_treino[AMOSTRAS][ENTRADA];
    double y_treino[AMOSTRAS][SAIDA];
    int dentro = 0;

    for (int i = 0; i < AMOSTRAS; i++) {
        double px, py, classe;
        gerar_ponto(&px, &py, &classe);
        x_treino[i][0] = px;
        x_treino[i][1] = py;
        y_treino[i][0] = classe;
        if (classe > 0.5) dentro++;
    }

    double x_teste[TESTES][ENTRADA];
    double y_teste[TESTES][SAIDA];

    for (int i = 0; i < TESTES; i++) {
        double px, py, classe;
        gerar_ponto(&px, &py, &classe);
        x_teste[i][0] = px;
        x_teste[i][1] = py;
        y_teste[i][0] = classe;
    }

    printf("Dados de treino : %d pontos (%d dentro, %d fora)\n",
           AMOSTRAS, dentro, AMOSTRAS - dentro);
    printf("Dados de teste  : %d pontos\n\n", TESTES);
    printf("Arquitetura: %d -> %d -> %d\n", ENTRADA, OCULTA, SAIDA);
    printf("Taxa de aprendizado: %.2f\n", TAXA);
    printf("Epocas: %d\n\n", CICLOS);

    iniciar_rede(&rede);

    printf("--- TREINAMENTO ---\n");

    int ordem[AMOSTRAS];
    for (int i = 0; i < AMOSTRAS; i++) ordem[i] = i;

    int ultima_epoca = CICLOS - 1;
    for (int ep = 0; ep < CICLOS; ep++) {
        double erro = 0.0;

        embaralhar(ordem, AMOSTRAS);

        for (int a = 0; a < AMOSTRAS; a++) {
            int idx = ordem[a];
            propagar(&rede, x_treino[idx]);
            retropropagar(&rede, y_treino[idx]);

            double dif = y_treino[idx][0] - rede.cam_saida[0];
            erro += dif * dif;
        }

        erro /= AMOSTRAS;

        if (ep % 5000 == 0 || ep == CICLOS - 1)
            printf("Epoca %5d  |  Erro MSE: %.6f\n", ep, erro);

        if (erro < ERRO_MINIMO) {
            printf("Epoca %5d  |  Erro MSE: %.6f  << CONVERGIU!\n", ep, erro);
            ultima_epoca = ep;
            break;
        }
    }

    int acertos_treino = 0;
    for (int a = 0; a < AMOSTRAS; a++) {
        propagar(&rede, x_treino[a]);
        int pred = (rede.cam_saida[0] >= 0.5) ? 1 : 0;
        int real = (y_treino[a][0]     >= 0.5) ? 1 : 0;
        if (pred == real) acertos_treino++;
    }

    int acertos_teste = 0;
    for (int a = 0; a < TESTES; a++) {
        propagar(&rede, x_teste[a]);
        int pred = (rede.cam_saida[0] >= 0.5) ? 1 : 0;
        int real = (y_teste[a][0]      >= 0.5) ? 1 : 0;
        if (pred == real) acertos_teste++;
    }

    printf("\n--- ACURACIA (apos %d epocas) ---\n\n", ultima_epoca + 1);
    printf("Treino : %d/%d = %.1f%%\n",
           acertos_treino, AMOSTRAS,
           100.0 * acertos_treino / AMOSTRAS);
    printf("Teste  : %d/%d = %.1f%%\n",
           acertos_teste, TESTES,
           100.0 * acertos_teste / TESTES);

    printf("\n--- DEMONSTRACAO (pontos conhecidos) ---\n\n");

    double demo[8][3] = {
        /* x       y      esperado */
        { 0.0,   0.0,   1.0 },  /* centro: claramente dentro       */
        { 0.5,   0.5,   1.0 },  /* dist = 0.707: dentro            */
        { 0.8,   0.8,   0.0 },  /* dist = 1.131: fora              */
        { 1.5,   0.0,   0.0 },  /* borda direita: fora             */
        { 0.0,   1.5,   0.0 },  /* borda superior: fora            */
        {-0.6,   0.6,   1.0 },  /* terceiro quadrante: dentro      */
        { 0.99,  0.0,   1.0 },  /* quase na borda: dentro          */
        { 1.01,  0.0,   0.0 },  /* quase na borda: fora            */
    };

    printf("%-8s %-8s %-10s %-10s %-8s\n", "x", "y", "distancia", "saida", "correto?");
    printf("----------------------------------------------------\n");

    for (int i = 0; i < 8; i++) {
        double pt[2] = { demo[i][0], demo[i][1] };
        propagar(&rede, pt);

        double dist = raiz(demo[i][0]*demo[i][0] + demo[i][1]*demo[i][1]);
        int pred     = (rede.cam_saida[0] >= 0.5) ? 1 : 0;
        int esperado = (int)demo[i][2];

        printf("%-8.2f %-8.2f %-10.4f %-10.6f %s\n",
               demo[i][0], demo[i][1], dist,
               rede.cam_saida[0],
               pred == esperado ? "OK" : "ERRO");
    }

    printf("\n--- MAPA ASCII (o que a rede aprendeu) ---\n\n");
    printf("   Legenda: # = dentro (saida >= 0.5)   . = fora   O = borda real\n\n");

    int tam = 31;
    for (int lin = 0; lin < tam; lin++) {
        printf("   ");
        for (int col = 0; col < tam; col++) {
            double mx = -1.8 + 3.6 * col / (tam - 1);
            double my =  1.8 - 3.6 * lin / (tam - 1);
            double r2 = mx * mx + my * my;

            double pt[2] = { mx, my };
            propagar(&rede, pt);
            int pred = (rede.cam_saida[0] >= 0.5) ? 1 : 0;

            double dist = raiz(r2);
            if (dist > 0.95 && dist < 1.08)
                printf("O");
            else if (pred)
                printf("#");
            else
                printf(".");
        }
        printf("\n");
    }

    printf("\n========================================\n");
    return 0;
}
