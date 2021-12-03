#include <stdio.h>
#include <stdlib.h>

/* PRISE EN MAIN DE CUDA AVEC CUDA_HELLO
__global__ void cuda_hello(){
    printf("Paris Bercy Bonsoir\n");
}
*/

// INITIALISATION DE LA MATRICE
void MatrixInit(float *M, int n, int p){
    for (int i=0; i<=n*p-1; i++){
        M[i]=(float)rand()/RAND_MAX*2-1;
    }
}

// AFFICHAGE SOUS FORME MATRICIELLE
void MatrixPrint(float *M, int n, int p){
    printf("  %f  ", M[0]);
    for (int i=1; i<=n*p-1; i++){
        if(i%p==0){
            printf("\n");
        }
        printf("  %f  ", M[i]);
    }
}

// ADDITION DE DEUX MATRICES SUR GPU
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){

    for(int i=0;i<n*p-1;i++){
        Mout[i]=M1[i]+M2[i];
    }
}

// ADDITION DE DEUX MATRICES SUR GPU

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){



}

/* MAIN */

int main(){
    
    //cuda_hello<<<1,1>>>();

    /* INITIALISATION DES VARIABLES*/
    float *M1, *M2, *Mout;
    int n=4, p=4;
    M1 = (float*) malloc(sizeof(float)*n*p);
    M2 = (float*) malloc(sizeof(float)*n*p);
    Mout = (float*) malloc(sizeof(float)*n*p);

    /* APPEL AUX FONCTIONS */

    /* CPU */
    /*
    MatrixInit(M1,n,p);
    printf("Matrice M1 = \n");
    MatrixPrint(M1,n,p);
    printf("\n");
    MatrixInit(M2,n,p);
    printf("Matrice M2 = \n");
    MatrixPrint(M2,n,p);
    printf("\n");
    MatrixAdd(M1, M2, Mout, n, p);
    printf("Matrice Mout = \n");
    MatrixPrint(Mout,n,p);
    printf("\n");
    */


    cudaDeviceSynchronize();
    return 0;
}