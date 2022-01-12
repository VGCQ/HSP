# TP : Implémentation d'un CNN  - LeNet-5 sur GPU

## Objectifs & méthodes de travail 

Les objectif de ces 4 séances de TP de HSP sont :
* Apprendre à utiliser CUDA,
* Etudier la complexité de vos algorithmes et l'accélération obtenue sur GPU par rapport à une éxécution sur CPU,
* Observer les limites de l'utilisation d'un GPU,
* Implémenter "from scratch" un CNN : juste la partie inférence et non l'entrainement,
* Exporter des données depuis un notebook python et les réimporter dans un projet cuda,
* Faire un suivi de votre projet et du versionning à l'outil git.

### Implémentation d'un CNN

#### LeNet-5

##### L'objectif à terme de ces 4 séances est d'implémenter l'inférence dun CNN très classique : LeNet-5

<a href="https://zupimages.net/viewer.php?id=22/02/cqff.png"><img src="https://zupimages.net/up/22/02/cqff.png" alt="" /></a>

La lecture de cet article peut vous apporter les informations nécessaires pour comprendre ce réseau de neurone :
https://www.datasciencecentral.com/profiles/blogs/lenet-5-a-classic-cnn-architecture

## Partie 1 - Prise en main de Cuda : Multiplication de matrices

### Multiplication de matrices

#### Création d'une matrice sur CPU
##### Initialisation d'une matrice de taille n x p, avec des valeurs aléatoires entre -1 et 1
```
void MatrixInit(float *M, int n, int p)
```

##### Initialisation d'une matrice de taille n x p, avec des valeurs égales à 0
```
void MatrixInit0(float *M, int n, int p)
```

#### Affichage d'une matrice sur CPU
```
void MatrixPrint(float *M, int n, int p)
```

#### Addition de deux matrices M1 et M2 de même taille n x p sur CPU
```
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p)
```

#### Addition de deux matrices M1 et M2 de taille n x p sur GPU
```
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p)
```

#### Multiplication de deux matrices M1 et M2 de taille NxN sur CPU
```
void MatrixMult(float *M1, float *M2, float *Mout, int n)
```

#### Multiplication de deux matrices M1 et M2 de taille NxN sur GPU
```
__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n)
}
```

## Partie 2 - Premières couches du réseau de neurone LeNet-5 : Convolution 2D et subsampling

### L'architecture du réseau LeNet-5 est composé de plusieurs couches :
* #### Layer 1 - Génération des données de test
```
MatrixInit0(raw_data,N,P,1); # avec N= P =32
```
* #### Layer 2- Convolution avec 6 noyaux de convolution de taille 5x5. La taille résultantes est donc de 6x28x28.
```
__global__ void cudaConv2D(double *img, double *kernels, double * out, int n, int p, int q, int k )
```
* #### Layer 3- Sous-échantillonnage d'un facteur 2. La taille résultantes des données est donc de 6x14x14.
```
__global__ void cudaMeanPool(double *in, double *out, int n, int p, int q)
```
### Tests

### Fonctions d'activation - Tanh
```
__device__ double cudaActivationTanh(double val)
```

## Partie 3 - Un peu de Python

Un notebook *LeNet5bis.ipynb* est mis à disposition et à étudier. Celui-ci réalise l'entrainement de notre réseau de neuronne, on l'utilise pour récupérer les poids et biais de chaque couche afin d'initialiser les différents kernels.

### Fonction d'activation manquante - Softmax
```
void ActivationSoftmax(double* input, size_t size)
```

### Importation du dataset MNIST et affichage des données en console
<a href="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png"><img src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png" alt="" /></a>

 Un script *printMNIST.cu* est donné, permettant l'ouverture de ce fichier et l'affichage d'une image. 
⚠️ **Spoiler** ⚠️ A ce stade nous n'avons pas réussi à exporter les poids de chaque couche à partir des poids exportés du notebook Python afin de les ajouter à notre programme en C. Toutefois, nous avons toutefois compris la démarche et avons rédigé les lignes "comme si" nous avions ces couches exportées.
