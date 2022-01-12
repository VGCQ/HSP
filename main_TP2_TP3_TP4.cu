/*

MENDES-CHARRINHO Léopold && QUETU Victor
3A SIA
TP Hardware for Signal Proccessing
Séance 2-3-4 : LeNet-5 sur GPU

*/

/* IMPORTATION DES BIBLIOTHEQUES*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

/* DEFINITION DES VARIABLES GLOBALEs */

#define N 32
#define P 32
#define Q 6
#define K 5
#define WIDTH 28
#define HEIGHT 28

/* Fonction renvoyant la prédiction */

void indexMax(double *in) {
    int k = 0;
    double max = in[k];

    for (int i = 0; i < 10; ++i)
    {
        if (in[i] > max)
        {
            max = (double)in[i];
            k = i;
        }
    }
    printf("C'est %d \n",k);
    }

/* Fonction qui permet de lire un fichier */

void read_file(char* path, double * out){
    FILE *f = fopen(path, "r");

    if (f == NULL)
    {
        printf("Error: could not open file %s", path);
    }
    int i =0;

    while ((fscanf(f,"%lf", &out[i])) != EOF){
        i++;
    }
    fclose(f);
}

/* Fonction qui permet de lire un fichier image */

void readImage(double * data){
    FILE *fptr;
    unsigned int magic, nbImg, nbRows, nbCols;
    unsigned char val;

    //Ouvrir Fichier
    if((fptr = fopen("train-images.idx3-ubyte","rb")) == NULL){
        printf("Can't open file");
        exit(1);
    }

    //Lire Fichier
    fread(&magic, sizeof(int), 1, fptr);
    fread(&nbImg, sizeof(int), 1, fptr);
    fread(&nbRows, sizeof(int), 1, fptr);
    fread(&nbCols, sizeof(int), 1, fptr);

    int no_img = 20;
    for(int i=0; i<no_img; i++){


        for(int i=2; i<WIDTH+2; i++){
            for(int j=2; j<HEIGHT+2; j++){ 
                fread(&val, sizeof(unsigned char), 1, fptr);  
                data[i*P+j]=(double)val/255;
            }
        }

    }
    
}

/* Fonction qui permet d'afficher l'image */

void charBckgrndPrint(char *str, double val){
  printf("\033[48;2;%d;%d;%dm", (int) (val*255), 0, 0);
  printf("%s\033[0m",str);
}

void imgColorPrint(int height, int width, double *img){
  int row, col;
  char *str="  ";
  for(row=0; row<height; row++){
    for(col=0; col<width; col++){
      charBckgrndPrint(str,img[row * height + col]);
    }
    printf("\n");
  }
}

/* Fonctions qui permettent d'initialiser des matrices à 0 ou avec des valeurs aléatoires */

void MatrixInit(double *M, int n, int p,int q){
    int i;
    for (i=0;i<n*p*q;i++){
        M[i]= (double)rand()/(RAND_MAX); //aléatoire entre 0 et 1
    }
}

void MatrixInit2(double *M, int n, int p,int q){
    int i;
    for (i=0;i<n*p*q;i++){
        M[i]= (double)rand()/(RAND_MAX/2)-1; //aléatoire entre -1 et 0
    }
}

void MatrixInit0(double *M, int n, int p,int q){
    int i;
    for (i=0;i<n*p*q;i++){
        M[i]= (double) 0; //initialisation à 0
    }
}

/* Fonction qui affiche une matrice */
void MatrixPrint(double *M, int n, int p, int q){
    int i;
    for (i=0;i<n*p*q;i++){
        if(M[i]>0) printf(" ");
        printf("%1.2f ", M[i]);
        if ((i+1+p)%p==0 ){
            printf("\n");
        }
        if ((i+1)%(n*p)==0 ){
            printf("\n\n");
        }
    }
    printf("\n");
}

/* Fonction d'activation tanh */
__device__ double cudaActivationTanh(double val) {
    return tanhf(val);
}

/* Fonction d'activation softmax */
void ActivationSoftmax(double* input, size_t size) {
	int i;
	double m, sum, constant;

	m = -INFINITY;
	for (i = 0; i < size; ++i) {
		if (m < input[i]) {
			m = input[i];
		}
	}

	sum = 0.0;
	for (i = 0; i < size; ++i) {
		sum += exp(input[i] - m);
	}

	constant = m + log(sum);
	for (i = 0; i < size; ++i) {
		input[i] = exp(input[i] - constant);
	}

}


/* Fonction réalisant la convolution 2D d'une image avec des kernels */
__global__ void cudaConv2D(double *img, double *kernels, double * out, int n, int p, int q, int k ) {
    // n,p= lignes, colonnes  de l'image d'entrée
    // q = nombre de kernels
    // k = dimension kernel (k*k)

    int m=n-k+1; //dimension image de sortie
    //int id = (k-1)/2; //centre

    int l = threadIdx.x; //->28 : dimension image de sortie
    int c = threadIdx.y; //->28 : dimension image de sortie
    
    int d = blockIdx.x;  //->6 : dimension nombre de kernel deconvolution
    
    double temp=0;
    int i,j;

    //Calcul du bloc K*K

    for (int ki= 0; ki<k;ki++){
        for (int kj= 0; kj<k;kj++){
            i=l+ki;
            j=c+kj;   
            temp+=img[i*n + j] * kernels[d*k*k + ki*k + kj];
        }
    }

    out[d*m*m + l*m + c] = cudaActivationTanh(temp); //temp
}

/* Fonction réalisant la convolution 3D d'une image avec des kernels */

__global__ void cudaConv3D(double *img, double *kernels, double * out, int n, int p, int q, int k ) {
    // n,p= lignes, colonnes  de l'image d'entrée
    // q = nombre de kernels
    // k = dimension kernel (k*k)

    int m=n-k+1; //dimension image de sortie
    //int id = (k-1)/2; //centre

    int l = threadIdx.x; // dimension image de sortie
    int c = threadIdx.y; // dimension image de sortie
    
    int d = blockIdx.x;  // dimension nombre de kernel deconvolution
    //int o = blockIdx.y; //dimension nombre d'images d'entrée
    
    int i,j;
    for (int o=0; o<1; o++){
        //Calcul du bloc K*K =5*5
        double temp=0;
        for (int ki= 0; ki<k;ki++){
            for (int kj= 0; kj<k;kj++){
                i=l+ki;
                j=c+kj;   
                temp+=img[o*n*p +i*n + j] * kernels[d*k*k + ki*k + kj];
            }
        }

        out[ o*m*m*q +d*m*m+ l*m + c]= cudaActivationTanh(temp);
    }
}


/* Fonction qui combine l'entrée */
__global__ void cudaCombine(double *in, double * out, double * id ) {
    int l = threadIdx.x; // ligne
    int c = threadIdx.y; // colonne
    
    int g = blockIdx.x; // dimension de sortie profondeur

    double temp = 0;
    for(int i=0;i<6;i++){
        temp+=in[i*16*10*10 + g*10*10 +l*10 +c]*id[i*16+g];
    }
    out[g*10*10 + l*10 + c ]= temp;
}

/* Fonction qui réalise un sous échantillonnage par moyennage */

__global__ void cudaMeanPool(double *in, double *out, int n, int p, int q) {
    // n,p = lignes, colonnes  de l'image d'entrée
    // q = nombre de kernels

    int m=n/2; //dimension sortie

    int l = threadIdx.x; //->28 (dimension de sortie)
    int c = threadIdx.y;  //->28 (dimension de sortie)
    int d = blockIdx.x;  //->6  (dimension nombre de kernel)
    
    double temp=0;
    int i,j;

    //Calcul du bloc K*K

    for (int ki= 0; ki<2;ki++){
        for (int kj= 0; kj<2;kj++){
            i=l+ki;
            j=c+kj;    
            temp+=in[d*n*p + i*n + j];
        }
    }

    out[d*m*m + l*m + c]= temp/4;
}

/* Fonction qui réalise une couche complètement connectée */

__global__ void cudaFullyConnected(double *in, double *w, double *out, int n, int p,int q, int activation){
    // n, p= 5 dimension entrées
    // q = 16 profondeur de in
    // l'entier "activation" gère la fonction d'activation à choisir : 1 pour tanh, 2 pour softmax

    int l = threadIdx.x; // 120 = taille vecteur sortie
    double temp = 0;
    
    for (int i=0; i<n*p*q;i++){
        temp+=in[i]*w[i*n*p*q +l];
    }

    // 1 for tanh 2 for softmax

    if (activation ==1) {
        out[l]= cudaActivationTanh(temp); 
    }
    else if (activation ==2) {
        out[l] = temp;
    }

}


/* Fonction principale utilisant toutes les fonctions précédentes */

int main(){

    /* INITIALISATION DES VARIABLES */
    int L=(N-K+1); // dimension de sortie de la convolution
    int M=(L/2); // dimension de sortie du sous-échantillonnage par moyennage

    double *raw_data, *C1_data, *C1_kernel, *S2_data,*C3_dataTemp, *C3_data , *C3_kernel, *S4_data, *F5_data, *F6_data, *OUTPUT, *W1, *W2, *W3;

    srand(time(NULL));

    /* ALLOCATION MEMOIRE DES VARIABLES */

    raw_data = (double*)malloc(N*P * sizeof(double));
    C1_data = (double*)malloc(Q*L*L * sizeof(double));
    C1_kernel = (double*)malloc(Q*K*K * sizeof(double));
    S2_data = (double*)malloc(Q*M*M * sizeof(double));

    C3_dataTemp = (double*)malloc(96*10*10 * sizeof(double));
    C3_data = (double*)malloc(16*10*10 * sizeof(double));
    C3_kernel = (double*)malloc(16*6*K*K * sizeof(double));
    S4_data = (double*)malloc(16*5*5 * sizeof(double));
    
    F5_data = (double*)malloc(120 * sizeof(double));
    F6_data = (double*)malloc(84 * sizeof(double));
    OUTPUT = (double*)malloc(10 * sizeof(double));
    W1 = (double*)malloc(120*16*5*5 * sizeof(double));
    W2 = (double*)malloc(84*120 * sizeof(double));
    W3 = (double*)malloc(84*10 * sizeof(double));

    double combineId[96] = {
        1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,
        1,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,
        1,1,1,0,0,0,1,1,1,0,0,1,0,1,1,1,
        0,1,1,1,0,0,1,1,1,1,0,0,1,0,1,1,
        0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1,
        0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1
    };

    /* INITIALISATION DES MATRICES */

    MatrixInit0(raw_data,N,P,1);
    readImage(raw_data);

    MatrixInit0(C1_data,L,L,Q);
    MatrixInit0(C1_kernel,K,K,Q);
    //Ajout d'une couche de poids à C1_kernel;

    MatrixInit0(S2_data,M,M,Q);

    MatrixInit0(C3_dataTemp,10,10,96);
    MatrixInit0(C3_data,10,10,16);
    MatrixInit0(C3_kernel,K,K,16*6);
    //Ajout d'une couche de poids à C3_kernel;

    MatrixInit0(S4_data,5,5,16);

    MatrixInit0(F5_data,120,1,1);
    MatrixInit0(F6_data,84,1,1);
    MatrixInit0(OUTPUT,10,1,1);
    MatrixInit0(W1,400,120,1);
    //Ajout d'une couche de poids à W1;
    MatrixInit0(W2,120,84,1);
    //Ajout d'une couche de poids à W2;
    MatrixInit0(W3,84,10,1);
    //Ajout d'une couche de poids à W3;

    //printf("IMAGE\n");
    //MatrixPrint(raw_data,N,P,1);
    //printf("KERNEL\n");
    //MatrixPrint(C1_kernel,K,K,1);
    

    /* Opérations GPU */

    /* INITIALISATION */

    double *d_combine,*d_raw, *d_C1, *d_C1_kernel, *d_S2,*d_C3Temp, *d_C3, *d_C3_kernel, *d_S4, *d_F5, *d_F6, *d_OUTPUT, *d_W1, *d_W2, *d_W3;

    /* ALLOCATION EN MEMOIRE DES VARIABLES */
    cudaMalloc((void**)&d_raw, sizeof(double)*N*P);
    cudaMalloc((void**)&d_C1, sizeof(double)*Q*L*L);
    cudaMalloc((void**)&d_C1_kernel, sizeof(double)*Q*K*K);
    cudaMalloc((void**)&d_S2, sizeof(double)*Q*M*M);
    cudaMalloc((void**)&d_C3Temp, sizeof(double)*96*10*10);
    cudaMalloc((void**)&d_C3, sizeof(double)*16*10*10);
    cudaMalloc((void**)&d_C3_kernel, sizeof(double)*16*6*K*K);
    cudaMalloc((void**)&d_combine, sizeof(double)*16*6);
    cudaMalloc((void**)&d_S4, sizeof(double)*16*5*5);
    cudaMalloc((void**)&d_F5, sizeof(double)*120);
    cudaMalloc((void**)&d_F6, sizeof(double)*84);
    cudaMalloc((void**)&d_OUTPUT, sizeof(double)*10);
    cudaMalloc((void**)&d_W1, sizeof(double)*120*400);
    cudaMalloc((void**)&d_W2, sizeof(double)*120*84);
    cudaMalloc((void**)&d_W3, sizeof(double)*84*10);

    /* COPIE DES VARIABLES VERS GPU*/

    cudaMemcpy(d_raw, raw_data, sizeof(double) * N*P, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1, C1_data, sizeof(double) * Q*L*L, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, sizeof(double) * Q*K*K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S2, S2_data, sizeof(double) * Q*M*M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C3Temp, C3_dataTemp, sizeof(double) * 96*10*10, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C3, C3_data, sizeof(double) * 16*10*10, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C3_kernel, C3_kernel, sizeof(double) * 16*6*K*K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_combine, combineId, sizeof(double) * 16*6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S4, S4_data, sizeof(double) * 16*5*5, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F5, F5_data, sizeof(double) * 120, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F6, F6_data, sizeof(double) * 84, cudaMemcpyHostToDevice);
    cudaMemcpy(d_OUTPUT, OUTPUT, sizeof(double) * 10, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, W1, sizeof(double) * 120*400, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2, sizeof(double) * 120*84, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W3, W3, sizeof(double) * 10*84, cudaMemcpyHostToDevice);



    /* CONVOLUTION 1 */
    dim3 nb_thread(L,L); //L=28
    dim3 nb_block(Q);    //Q=6

    cudaConv2D<<<nb_block,nb_thread>>>(d_raw, d_C1_kernel, d_C1, N, P, Q, K );
    cudaDeviceSynchronize();

    /* SOUS ECHANTILLONNAGE PAR MOYENNAGE 1 */
    dim3 nb_thread2(M,M);  //M=14
    dim3 nb_block2(Q);     //Q=6
    cudaMeanPool<<<nb_block2,nb_thread2>>>(d_C1, d_S2, L, L, Q );
    cudaDeviceSynchronize();

    /* CONVOLUTION 2  14*14*6 -> 10*10*96 -> 10*10*16 */
    dim3 nb_thread3(10,10); //im out
    dim3 nb_block3(96); //nb kernel
    cudaConv3D<<<nb_block3,nb_thread3>>>(d_S2, d_C3_kernel, d_C3Temp, 14, 14, 96, K );


    /* COMBINAISON */ 
    cudaDeviceSynchronize();
    dim3 nb_thread4(10,10);
    dim3 nb_block4(16);
    cudaCombine<<<nb_block4,nb_thread4>>>(d_C3Temp,d_C3, d_combine);
    cudaDeviceSynchronize();

    /* SOUS ECHANTILLONNAGE PAR MOYENNAGE 2 */
    dim3 nb_thread5(5,5);
    dim3 nb_block5(16);
    cudaMeanPool<<<nb_block5,nb_thread5>>>(d_C3, d_S4, 10, 10, 16 );
    cudaDeviceSynchronize();


    /* COUCHE COMPLETEMENT CONNECTE 1 */
    dim3 nb_thread6(120);
    cudaFullyConnected<<<1,nb_thread6>>>(d_S4, d_W1, d_F5,  5, 5, 16, 1 );
    cudaDeviceSynchronize();

    /* COUCHE COMPLETEMENT CONNECTE 2 */
    dim3 nb_thread7(84);
    cudaFullyConnected<<<1,nb_thread7>>>(d_F5, d_W2, d_F6,  120, 1, 1, 1 );
    cudaDeviceSynchronize();

    /* COUCHE COMPLETEMENT CONNECTE 3 */
    dim3 nb_thread8(10);
    cudaFullyConnected<<<1,nb_thread8>>>(d_F6, d_W3, d_OUTPUT,  84, 1, 1, 2 );
    cudaDeviceSynchronize();


    /* COPIE VERS LE CPU */

    cudaMemcpy(C1_data, d_C1, sizeof(double)* Q*L*L, cudaMemcpyDeviceToHost);
    //printf("C1\n");
    //MatrixPrint(C1_data,L,L,1);
    
    cudaMemcpy(S2_data, d_S2, sizeof(double)* Q*M*M, cudaMemcpyDeviceToHost);
    //printf("MEAN\n");
    //MatrixPrint(S2_data,M,M,1); //M<L

    cudaMemcpy(C3_dataTemp, d_C3Temp, sizeof(double)* 96*10*10, cudaMemcpyDeviceToHost);
    cudaMemcpy(C3_data, d_C3, sizeof(double)* 16*10*10, cudaMemcpyDeviceToHost);
    
    //printf("C3 Temp\n");
    //MatrixPrint(C3_dataTemp,10,10,16); 
    //printf("C3\n");
    //MatrixPrint(C3_data,10,10,16); //M<L

    cudaMemcpy(S4_data, d_S4, sizeof(double)* 16*5*5, cudaMemcpyDeviceToHost);
    //printf("MEAN2\n");
    //MatrixPrint(S4_data,5,5,2); 

    cudaMemcpy(OUTPUT, d_OUTPUT, sizeof(double)* 10, cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();

    ActivationSoftmax(OUTPUT,10);
    printf("OUTPUT\n");
    MatrixPrint(OUTPUT,10,1,1); 

    imgColorPrint(32,32,raw_data);
    
    indexMax(OUTPUT);

    /* LIBERATION DE LA MEMOIRE */

    cudaFree(d_raw);
    cudaFree(d_C1);
    cudaFree(d_C1_kernel);
    cudaFree(d_combine);
    cudaFree(d_S2);
    cudaFree(d_C3Temp);
    cudaFree(d_C3);
    cudaFree(d_C3_kernel);
    cudaFree(d_S4);
    cudaFree(d_F5);
    cudaFree(d_F6);
    cudaFree(d_OUTPUT);
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_W3);

    free(raw_data);
    free(C1_data);
    free(C1_kernel);
    free(S2_data);
    free(C3_dataTemp);
    free(C3_data);
    free(C3_kernel);
    free(S4_data);
    free(F5_data);
    free(F6_data);
    free(OUTPUT);
    free(W1);
    free(W2);
    free(W3);

}