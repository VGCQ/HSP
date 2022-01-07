/*

MENDES-CHARRINHO Léopold && QUETU Victor
3A SIA
TP Hardware for Signal Proccessing
Séance 2 : LeNet-5 sur GPU

*/

#include <stdio.h>
#include <stdlib.h>

/* INITIALISATION DES MATRICES */

void MatrixInit(float *M, int n, int p, int d, int choix){

/*
On ajoute une variable "choix" à la fonction MatrixInit du TP1 afin de pouvoir faire des initialisations de matrice différentes
Matrice nulle : choix == 0 
Matrice nulle avec centre égal à 1 :  choix == 1: 
Matrice aléatoire entre 0 et 1 : choix == 2
*/
    
    if (choix == 0){
        for (int i = 0; i < n * p * d; i++){
            M[i] =  0;
        }
    }
    else if (choix == 1){
        for (int i = 0; i < n * p * d; i++){
            M[i] =  0;
        }
        M[4] = 1;
    }
    else{
        for (int i = 0; i < n * p * d; i++){
            M[i]=(float)rand()/RAND_MAX; // entre 0 et 1
        }
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

/* ADDITION DE DEUX MATRICES SUR GPU */
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    
    printf("Addition depuis le GPU en cours...\n\n");
    
    for (int i=0; i<n*p; i++){
        Mout[i] = M1[i]+M2[i];
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

/* CONVOLUTION 2D */
__global__ void cudaConv2D(float* M, float* kernel, float* Mout, int M_ligne, int M_colonne, int kernel_size, int nb_kernel, int Mout_ligne, int Mout_colonne){
    
    /*Convolution d'une matrice par un kernel*/
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float s = 0.0;

    if (lig < Mout_ligne && col < Mout_colonne){
        int tot = M_ligne * M_colonne;

        for (int kernel_lig = 0; kernel_lig < kernel_size; kernel_lig++) {
            for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                for (int n_k = 0; n_k < nb_kernel; n_k++){
                    s += M[(lig + kernel_lig) * M_colonne + col + kernel_col + n_k * tot] * kernel[kernel_lig * kernel_size + kernel_col + n_k * nb_kernel];
            
                }
            }
        }
        Mout[lig * Mout_colonne + col] = s;
    }
}

/* SOUS-ECHANTILLONNAGE */

__global__ void cudaMeanPool(float* M, float* Mout, int M_ligne, int M_colonne, int profondeur, int meanpool_size, int Mout_ligne, int Mout_colonne){
    
    /*Sous-échantillonnage d'une matrice par un kernel 2x2*/
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float s = 0.0;
    int tot_meanpool = meanpool_size * meanpool_size;

    if (lig % meanpool_size == 0 && col % meanpool_size == 0){
        int tot = M_ligne * M_colonne;

        for (int meanpool_lig = 0; meanpool_lig < meanpool_size; meanpool_lig++) {
            for (int meanpool_col = 0; meanpool_col < meanpool_size; meanpool_col++) {
                for (int n_prof = 0; n_prof < profondeur; n_prof++){
                    s += M[(lig + meanpool_lig) * M_colonne + col + meanpool_col + n_prof * tot] / tot_meanpool;
            
                }
            }
        }
        if (lig == 0){
            Mout[lig * Mout_colonne + (col / meanpool_size)] = s;
    
        }
        else if (col == 0){
            Mout[(lig / meanpool_size) * Mout_colonne + col] = s;
    
        }
        else{
            Mout[(lig / meanpool_size) * Mout_colonne + (col / meanpool_size)] = s;
        }
    }
}

/* Fonction d'activation - Tanh */

__device__ float* activation_tanh(float* M, int nThreads){
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nThreads; i+= blockDim.x * gridDim.x){
        M[i] = tanh(M[i]);
    }
    
    return M;
}


__global__ void cudaTanh(float* M, int nThreads){
    activation_tanh(M, nThreads);
}

/* MAIN */

int main(){
    
    //Création de l'image d'entrée à convoluer
    float *raw_data;    
    raw_data = (float*)malloc(32 * 32 * 1 * sizeof(float));
    
    MatrixInit(raw_data, 32, 32, 1, 2);
    
    //Création de la sortie de la conv2D
    float *C1_data;    
    C1_data = (float*)malloc(28 * 28 * 6 * sizeof(float));
    
    MatrixInit(C1_data, 28, 28, 6, 0);
    
    //Création de la sortie du sous-échantillonnage
    float *S1_data;    
    S1_data = (float*)malloc(14 * 14 * 6 * sizeof(float));
    
    MatrixInit(S1_data, 14, 14, 6, 0);
    
    
    //Création des premiers noyaux de convolution
    float *C1_kernel;    
    C1_kernel = (float*)malloc(5 * 5 * 6 * sizeof(float));
    
    MatrixInit(C1_kernel, 5, 5, 6, 1);

    float *d_raw_data, *d_C1_data, *d_C1_kernel, *d_S1_data;
    
    //Allocation des mémoires des matrices pour cuda
    cudaMalloc((void**)&d_raw_data, sizeof(float) * 32 * 32 * 1);
    cudaMalloc((void**)&d_C1_kernel, sizeof(float) * 5 * 5 * 6);
    cudaMalloc((void**)&d_C1_data, sizeof(float) * 28 * 28 * 6);
    cudaMalloc((void**)&d_S1_data, sizeof(float) * 14 * 14 * 6);

    cudaMemcpy(d_raw_data, raw_data, sizeof(float) * 32 * 32 * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, sizeof(float) * 5 * 5 * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_data, C1_data, sizeof(float) * 28 * 28 * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S1_data, S1_data, sizeof(float) * 14 * 14 * 6, cudaMemcpyHostToDevice);
  
    //Process sur GPU
    dim3 block_size(32, 32);
    dim3 grid_size(1,1);
    
    cudaConv2D<<<grid_size,block_size>>>(d_raw_data, d_C1_kernel, d_C1_data, 32, 32, 5, 6, 28, 28);
    cudaDeviceSynchronize();
    
    cudaTanh<<<grid_size, block_size>>>(d_C1_data, 28*28);
    cudaDeviceSynchronize();
    
    cudaMeanPool<<<grid_size, block_size>>>(d_C1_data, d_S1_data, 28, 28, 6, 2, 14, 14);
    cudaDeviceSynchronize();
    
    /* COPIE DES RESULTATS SUR CPU */
    cudaMemcpy(C1_data, d_C1_data, sizeof(float) * 28 * 28 * 6, cudaMemcpyDeviceToHost);
    cudaMemcpy(S1_data, d_S1_data, sizeof(float) * 14 * 14 * 6, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
        
    printf("Résultat :\nMout=\n");
    MatrixPrint(S1_data, 14, 14);

    /* LIBERATION MEMOIRE */
    cudaFree(d_raw_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_C1_data);
    cudaFree(d_S1_data);
    
    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);

    return 0;
}