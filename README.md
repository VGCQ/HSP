MENDES-CHARRINHO Léopold && QUETU Victor, 3A SIA

TP : Implémentation d'un CNN - LeNet-5 sur GPU

Les objectif de ces 4 séances de TP de HSP sont :

  -Apprendre à utiliser CUDA
  -Etudier la complexité de vos algorithmes et l'accélération obtenue sur GPU par rapport à une éxécution sur CPU
  -Observer les limites de l'utilisation d'un GPU
  -Implémenter "from scratch" un CNN : juste la partie inférence et non l'entrainement
  -Exporter des données depuis un notebook python et les réimporter dans un projet cuda
  -Faire un suivi de votre projet et du versionning à l'outil git
   
L'objectif à terme de ces 4 séances est d'implémenter l'inférence dun CNN très claissque : LeNet-5. L'architecture du LeNet-5 se compose de deux couches convolutives et de MeanPooling, suivis d'une couche convolutive d'aplatissement, puis de deux couches entièrement connectées et enfin d'un classificateur softmax.

La première séance est consacrée à la prise en main de Cuda. On réalise tout d'abord des fonctions d'initialisation, d'affichage, d'additions ou de multiplication de matrices d'abord sur CPU, puis en utilisant le GPU.
