/*

MENDES-CHARRINHO Léopold && QUETU Victor
3A SIA 
TP Hardware for Signal Proccessing
Séance 1 : Addition/Multiplication sur CPU/GPU

*/

#include <stdio.h>
#include <stdlib.h>

/* PRISE EN MAIN DE CUDA AVEC CUDA_HELLO */
__global__ void cuda_hello(){
    printf("Hello World\n\n");
}


/* INITIALISATION DES MATRICES */
void MatrixInit(float *M, int n, int p){
    for (int i=0; i<=n*p-1; i++){
        // M[i]=(float)rand()/RAND_MAX*2-1; // entre -1 et 1
        M[i]=(float)rand()/RAND_MAX; // entre 0 et 1
    }
}

void MatrixInit0(float *M, int n, int p){
    for (int i=0; i<=n*p-1; i++){
        M[i]=0;
    }
}

/* AFFICHAGE SOUS FORME MATRICIELLE */
void MatrixPrint(float *M, int n, int p){
    printf("  %f  ", M[0]);
    for (int i=1; i<=n*p-1; i++){
        if(i%p==0){
            printf("\n");
        }
        printf("  %f  ", M[i]);
    }
    printf("\n");
}

/* ADDITION DE DEUX MATRICES SUR CPU */
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    for(int i=0;i<n*p-1;i++){
        Mout[i]=M1[i]+M2[i];
    }
}

/* ADDITION DE DEUX MATRICES SUR GPU */
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    
    printf("Addition depuis le GPU en cours...\n\n");
    
    for (int i=0; i<n*p; i++){
        Mout[i] = M1[i]+M2[i];
    }
}

/* MULTIPLICATION DE DEUX MATRICES SUR CPU */
void MatrixMult(float *M1, float *M2, float *Mout, int n){    
    for (int lig = 0; lig < n; lig++){
        for (int col = 0; col < n; col++){
            float s = 0.0f;
            for (int i = 0; i < n; i++) {
                s += M1[lig * n + i] * M2[i * n + col];
            }
            Mout[lig * n + col] = s;
        }
    }
}

/*MULTIPLICATION DE DEUX MATRICES SUR GPU */
__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n){
    printf("Multiplication depuis le GPU en cours...\n\n");
    
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float s = 0.0f;
    
    if(lig < n && col < n){
        for (int i = 0; i < n; i++){
            s += M1[lig * n + i] * M2[i * n + col];
        }
    }
    Mout[lig * n + col] = s;
}


/* MAIN */

int main(){
    
    cuda_hello<<<1,1>>>(); // Helloworld
    cudaDeviceSynchronize();

    /* INITIALISATION DES VARIABLES POUR CPU*/
    
    float *M1, *M2, *Mout, *Mout_A, *Mout_M;
    int n=4, p=4;
    M1 = (float*) malloc(sizeof(float)*n*p);
    M2 = (float*) malloc(sizeof(float)*n*p);
    Mout = (float*) malloc(sizeof(float)*n*p); // Matrice pour Addition CPU
    Mout_A = (float*) malloc(sizeof(float)*n*p); // Matrice pour Addition GPU
    Mout_M = (float*) malloc(sizeof(float)*n*p); // Matrice pour Multiplication GPU
    
    /* INITIALISATION DES VARIABLES POUR GPU */
    float *d_M1, *d_M2, *d_Mout_A, *d_Mout_M;
    cudaMalloc((void**)&d_M1, sizeof(float)*n*p);
    cudaMalloc((void**)&d_M2, sizeof(float)*n*p);
    cudaMalloc((void**)&d_Mout_A, sizeof(float)*n*p);
    cudaMalloc((void**)&d_Mout_M, sizeof(float)*n*p);

    /* APPEL AUX FONCTIONS */

    /* AFFICHAGE SEANCE */

    printf("MENDES-CHARRINHO Léopold && QUETU Victor\n3A SIA\nTP Hardware for Signal Proccessing\nSéance 1 : Additon/Multiplication sur CPU/GPU\n");

    /* CPU */
    
    MatrixInit(M1,n,p);
    printf("\nMatrice M1 = \n");
    MatrixPrint(M1,n,p);
    printf("\n");
    MatrixInit(M2,n,p);
    printf("\nMatrice M2 = \n");
    MatrixPrint(M2,n,p);
    printf("\n");
    MatrixAdd(M1, M2, Mout, n, p);
    printf("\nRésulat Addition CPU : \nMout =\n");
    MatrixPrint(Mout,n,p);
    printf("\n");
    MatrixMult(M1, M2, Mout, n);
    printf("\nRésulat Multiplication CPU : \nMout =\n");
    MatrixPrint(Mout,n,p);
    printf("\n");
    
    /* GPU */
    
    /* COPIE DES INITIALISATIONS SUR LE GPU */

    cudaMemcpy(d_M1, M1, sizeof(float) * n * p, cudaMemcpyHostToDevice);  
    cudaMemcpy(d_M2, M2, sizeof(float) * n * p, cudaMemcpyHostToDevice);
    
    /* ADDITION SUR GPU */
    cudaMatrixAdd<<<1, 1>>>(d_M1, d_M2, d_Mout_A, n, p);
    
    /* MULTIPLICATION SUR GPU */
    dim3 block_size(n, n);
    dim3 grid_size(1);
    
    cudaMatrixMult<<<grid_size,block_size>>>(d_M1, d_M2, d_Mout_M, n);
    cudaDeviceSynchronize();
    
    /* COPIE DES RESULTATS SUR CPU */
    cudaMemcpy(Mout_A, d_Mout_A, sizeof(float)*n*p, cudaMemcpyDeviceToHost); 
    cudaMemcpy(Mout_M, d_Mout_M, sizeof(float)*n*p, cudaMemcpyDeviceToHost);  
    cudaDeviceSynchronize();
    
    printf("\nRésultat Addition GPU :\nMout=\n");
    MatrixPrint(Mout_A, n, p);
    
    printf("\nRésultat Multiplication GPU :\nMout=\n");
    MatrixPrint(Mout_M, n, p);

    /* LIBERATION MEMOIRE */
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout_A);
    cudaFree(d_Mout_M);
    
    free(M1);
    free(M2);
    free(Mout);
    free(Mout_A);
    free(Mout_M);

    return 0;
}
