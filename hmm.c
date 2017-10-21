#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <math.h>
#include <unistd.h>
#define N 2
#define M 27
#define T 50000

int getIndex(char c) {
    if(c != ' ') {
        int index = (c - 'a');
        return index;
    }
    else
        return 26;
}

int main() {
    printf("HMM Code to Analyze English text...\n");
    sleep(1);
    double rowsum[N];
    double A[N][N];
    double tempA[N][N];
    double tempB[N][M];
    double B[N][M];
    double tempPi[N];
    double pi[N];
    double sum = 0;
    int O[T];
    int maxIterations = 300;
    int iters = 0;
    float oldLogProb = -DBL_MAX;
    float c[T];
    float alpha[T][N];
    float beta[T][N];
    float gamma[T][N];
    float digamma[T][N][N];
    /* Initialize the A matrix with close to normal values, i.e. 1/N.
       Add 5 to each value and then divide by row sum.
       Values turn out to be row stochastic. */
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            sum = 0;
            double scale = rand() / (double) RAND_MAX; 
            tempA[i][j] = 0 + scale * ( 1 - 0 );
            sum += tempA[i][j];
            for(int i = 0; i < N; i++) {

                for(int j = 0; j < N; j++) {

                    A[i][j] = tempA[i][j]/sum;
                }
            }
        }
    }
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            A[i][j] = A[i][j] + 5;
        }
    }
    double sumA = 0;
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            sumA = sumA+ A[i][j];
        }
        rowsum[i] = sumA;
        sumA = 0;
    }
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            A[i][j] = A[i][j]/rowsum[i];
        }
    }
    /* Initialize the B matrix with close to normal values, i.e. 1/N.
       Add 5 to each value and then divide by row sum.
       Values turn out to be row stochastic. */
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            sum = 0;
            double scale = rand() / (double) RAND_MAX; 
            tempB[i][j] = 0 + scale * ( 1 - 0 );
            sum += tempB[i][j];
            for(int i = 0; i < N; i++) {
                for(int j = 0; j < M; j++) {
                    B[i][j] = tempB[i][j]/sum;
                }
            }
        }
    }
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            B[i][j] = B[i][j] + 5;
        }
    }
    double sumB = 0;
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            sumB = sumB+ B[i][j];
        }
        rowsum[i] = sumB;
        sumB = 0;
    }
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            B[i][j] = B[i][j]/rowsum[i];
        }
    }
     /* Initialize the pi matrix with close to normal values, i.e. 1/N.
       Add 5 to each value and then divide by row sum.
       Values turn out to be row stochastic. */
    for(int j = 0; j < N; j++) {
        sum = 0;
        double diff = 0.6 - 0.4;
        tempPi[j] = (((float) rand() / RAND_MAX) * diff) + 0.4;
        sum += tempPi[j];
    }
    for(int j = 0; j < N; j++) {
        pi[j] = tempPi[j] / sum;
    }
    for(int j = 0; j < N; j++) {
        pi[j] = pi[j] + 5;
    }
    double sumPi = 0;
    for(int j = 0; j < N; j++) {
        sumPi = sumPi + pi[j];
    }
    for(int j = 0; j < N; j++) {
        pi[j] = pi[j]/sumPi;
    }
    int tcount = 0;
    int ch = 0;
    FILE *fptr;
    printf("Reading plain-text file...\n");
    sleep(1);
    fptr = fopen("brown.txt", "r");
    if(fptr) {
        printf("File present at specified location...\n");
        sleep(1);
    }
    printf("Creating Observation Sequence from File...\n");
    sleep(1);
    if(fptr) {
        while((ch=fgetc(fptr))!=EOF && tcount < T) {
            O[tcount] = getIndex(ch);
            tcount++;
        }
    }
    fclose(fptr);
    /* Initialize alpha, beta, gamma, digamma and the scale array to 0. */
    for(int i = 0; i < T; i++)
        for(int j = 0; j < N; j++)
            alpha[i][j] = 0;
    for(int i = 0; i < T; i++)
        for(int j = 0; j < N; j++)
            beta[i][j] = 0;
    for(int i = 0; i < T; i++)
        for(int j = 0; j < N; j++)
            gamma[i][j] = 0;
    for(int i = 0; i < T; i++)
        for(int j = 0; j < N; j++)
            for(int k = 0; k < N; k++)
                digamma[i][j][k] = 0;
    for(int i = 0; i < T; i++)
        c[i] = 0;
   /* This portion of code is where we actually train the HMM model, which is defined as λ = (A, B, pi).
   The model is trained and the A, B and pi matrices are re-estimated every iteration until certain end conditions are met.
   This algorithm is also called the Baum-Welsch re-estimation algorithm. 
   The final B matrix would reveal information regarding the English text tha we are processing.
   Various levels of information could be obtained by initializing for different values of N. */
    printf("Training the Hidden Markov Model on given Observation Sequence...\n");
    sleep(1);
    alpha:
    /* This is the alpha-pass or also known as the forward algorithm.
       This algorithm is used for scoring sequences.
       It should be noted that final alpha-pass values are scaled using the scale array. */
    for(int i = 0; i < N; i++) {
        alpha[0][i] = pi[i] * B[i][O[0]];
        c[0] = c[0] + alpha[0][i];
    }
    c[0] = 1/c[0];
    for(int i = 0; i < N; i++) {
        alpha[0][i] = c[0] * alpha[0][i];
    }
    for(int t = 1; t < T; t++) {
        c[t] = 0;
        for(int i = 0; i < N; i++) {
            alpha[t][i] = 0;
            for(int j = 0; j < N; j++) {
                alpha[t][i] = alpha[t][i] + alpha[t-1][j]*A[j][i];
            }
            alpha[t][i] = alpha[t][i] * B[i][O[t]];
            c[t] = c[t] + alpha[t][i];
        }
        c[t] = 1/c[t];
        for(int i = 0; i < N; i++) {
            alpha[t][i] = c[t] * alpha[t][i];
        }
    }
    /* This phase is the beta-pass. Similar to the alpha-pass, the values are scaled.*/
    for(int i = 0; i < N; i++)
        beta[T-1][i] = c[T-1];
    for(int t = (T-2); t >= 0; t--) {
        for(int i = 0; i < N; i++) {
            beta[t][i] = 0;
            for(int j = 0; j < N; j++) {
                beta[t][i] = beta[t][i] + A[i][j] * B[j][O[t+1]] * beta[t+1][j];
            }
            beta[t][i] = c[t] * beta[t][i];
        }
    }
    /* The alpha and beta matrices are used to caculate the gamma and di-gamma matrices.
    Similar to the alpha and beta matrices, they are scaled.*/
    for(int t = 0; t < T-1; t++) {
        float denom = 0;
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                denom = denom + alpha[t][i] * A[i][j] * B[j][O[t+1]] * beta[t+1][j];
            }
        }
        for(int i = 0; i < N; i++) {
            gamma[t][i] = 0;
            for(int j = 0; j < N; j++) {
                digamma[t][i][j] = (alpha[t][i] * A[i][j] * B[j][O[t+1]] * beta[t+1][j])/denom;
                gamma[t][i] = gamma[t][i] + digamma[t][i][j];
            }
        }
    }
    /* 
    *************** Special case scenario for gamma ****************
    */
    float denom = 0;
    for(int i = 0; i < N; i++) {
        denom = denom + alpha[T-1][i];
    }
    for(int i = 0; i < N; i++) {
        gamma[T-1][i] = alpha[T-1][i]/denom;
    }
    /* Re-estimate the pi, A and B matrices based on the calculated gamma and di-gamma valaues.
       This is nothing but "training the model".
    */
    for(int i = 0; i < N; i++) {
        pi[i] = gamma[0][i];
    }
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            float numer = 0;
            float denom = 0;
            for (int t = 0; t < T-1; t++) {
                numer = numer + digamma[t][i][j];
                denom = denom + gamma[t][i];
            }
            A[i][j] = numer/denom;
        }
    }
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            float numer = 0;
            float denom = 0;
            for(int t = 0; t < T; t++) {
                if(O[t] == j) {
                    numer = numer + gamma[t][i];
                }
                denom = denom + gamma[t][i];
            }
            B[i][j] = numer/denom;
        }
    }
    /* We need to computelog(P(O|λ)).
       This is used as part of the end conditions to stop training the model. */
    float logProb = 0;
    for(int i = 0; i < T; i++) {
        logProb = logProb + log(c[i]);
    }
    logProb = -logProb;
    /* Also we need to have a iterations check along with the probability calculated in order to decide
       when we need to stop training the model. 
       if (condition not met)
            continue algorithm
       else
            print the final A, B and pi matrices */ 
    iters = iters + 1;
    if (iters < maxIterations || logProb > oldLogProb) {
        oldLogProb = logProb;
        goto alpha;
    } else {
        printf("\n");
        printf("A\n\n");
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++)
                printf("%f\t", A[i][j]);
            printf("\n");
        }
        printf("\n\n");
        printf("B\n\n");
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < M; j++)
                printf("%f\t", B[i][j]);
            printf("\n");
        }
        printf("\n\n");
        printf("Pi\n\n");
        for(int i = 0; i < N; i++) {
            printf("%f\t", pi[i]);
        }
        printf("\n\n");

    }
    return 0;
}
