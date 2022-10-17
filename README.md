# Embedded_AI : attaques sur des modèles de réseaux neuronaux convolutifs embarqués sur STM32F411

## Description du projet
Le projet vise à générer trois modèles de réseaux convolutionels de tailles différentes dont l'objectif est de déterminer à partir d'une image si une vigne est atteinte de la maladie esca. On essaiera d'embarquer ces modèles, en les modifiants si nécessaire, sur une STM32F411.
Ensuite, une série d'attaques seront effectués sur nos modèles pour observer leur fragilitées et des mesures seront mises en places pour essayer de les rendre plus robustes.

L'entrainement des modèles se fera sur l'outil en ligne google colab. On utilisera l'outil STMCubeMx pour générer le code à flasher sur notre STM32F411 et STMCubeIDE pour flasher et faire du débogage.

On utilisera la librairie Keras fonctionnant sur TensorFlow pour la construction de nos modèles.

L'ensemble du projet sera mené sur une carte STM32L4R9 Discovery kit.

## Sommaire
[ToC]
## 1.Génération des modèles
### 1.1. Récupération du dataset
Au sein de ce projet on travail sur un dataset prééxistant. Celui-ci a été collecté lors d'un travail conjoint du département d'ingénierie de l'information de l'université Polytechnique de Marche et de STMicroelectronics.
Le dataset contient 1770 images de feuilles de vignes séparée en deux catégories de taille égales:

- Des feuilles saines (healthy)
- Des feuilles atteintes de la maladie d'Esca (esca). La maladie d'esca est provoquée par la contamination du bois de la vigne par des champignons et bactéries provoquant la nécrose du bois, des symptômes foliaires et progressivement la mort de la plante. (https://www.maladie-du-bois-vigne.fr/Les-maladies-du-bois/L-esca)


On récupère le dataset sur https://data.mendeley.com/datasets/89cnxc58kj/1 et on stocke celui-ci sur un google drive. Cela permet de pouvoir y accéder directement via l'outil google colab.

### 1.2. Augmentation du dataset
La performance des modèles que l'ont va générer dépend de la qualitée et de la quantité de nos données d'entrainement. Par exemple, si l'on a trop peu de donnée d'entrainement comparée à la taille de notre modèle, celui-ci va avoir tendance à faire de l'overfitting.
Grâce à cette augmentation de la taille de notre dataset, on passe de 1770 images à 24780. Pour chaque, image on vient créer 13 variations de celle-ci. Cette nouvelle base de donnée sera elle aussi enregistrée sur notre google drive.

Example des augmentation faites sur une image :

![change_queue_position](https://github.com/louise17300/embedded_AI/blob/main/Images/augmentations_example.png?raw=True)

### 1.3. Prétraitement de la base de données
Il est nécessaire de redimensionner la taille des images car chaque modèle prent en entrée des images de dimensions différentes.
De plus, on sépare notre base de données en trois parties distinctes :

- *Données d'entrainements* : ce sont les données utilisées lors de l'entrainement de notre modèle.
- *Données de validation* : à la fin de chaque époch, on vient valider l'apprentissage de notre modèle. Cette validation se fait sur les données de validation.
- *Données de test* : Une fois notre modèle complètement entrainé, on vient tester sa précision avec les données de test.

Si l'on ne fait pas cette séparation et que l'on valide ou test notre modèle avec les mêmes données que celles utilisées lors de l'entrainement, on ne sera pas à même de s'assurer que notre modèle ait bien généralisé et pas seulement mémoriser. On peut ainsi détecter les problèmes d'overfitting.

### 1.4. Création du modèle
Pour nos trois modèle, on utilise un modèle séquentiel. Le modèle séquentiel est apprioprié pour une pile de couches simple où chaque couche a exactement un tenseur d'entrée et un tenseur de sortie (https://www.tensorflow.org/guide/keras/sequential_model). 
Les trois modèle sont constitués de 5 première surcouches, elles même constituées de trois couches :
- *Conv2D* : permet de faire la convolution sur notre image en entrée et de donner en sortie un tensor. Un tensor représente un tableau d'images
    - Filtres : la première, deuxième et dernière couches possèdent 32 filtres, la troisième et quatrième 64. Donnant en sortie respectivement des tensor de taille (x,y,64), x et y correspondent à la taille de notre image en input.
    - Taille du noyau (kernel_size) : 3 par 3, la fenêtre de notre convolution se fait sur un carré de 3 pixels de côté.
    - Padding : same, signifie que la taille d'un filtre sera la même que la taille de l'image donnée en entrée.
- *Relu* : couche d'activation. Permet de définir la valeur de sortie des neurones en fonction de la fonction d'activation. Ici Relu :

![change_queue_position](https://github.com/louise17300/embedded_AI/blob/main/Images/Relu.png?raw=True)
- *MaxPooling2D* : permet de faire le pooling, donc le groupement de donnée. Cela permet de donner une invariance pour les petites variations. Le MaxPooling2D permet de prendre la variable la plus grande sur une fenêtre de 2 pixels de côtés

**Question : on connait pas le padding de MaxPooling2D**

Suivant ces 5 surcouches se trouve 6 couches :
- *Flatten* : permet d'aplatir notre donnée en 1D.
- *Dense* : Couche connectée de manière dense, tous les noeuds sont connectés à tous les noeuds précédents.
- *Activation* : couche d'activation. Permet de définir la valeur de sortie des neurones en fonction de la fonction d'activation.
- *Dropout* : Permet de mettre aléatoirement certains poids à 0 pendant l'entrainement. Cela permet de prévenir l'overfitting.
- *Dense* : Couche connectée de manière dense, avec seulement 2 noeuds car l'on a 2 classes.
- *Activation* : On utilise Softmax qui permet de donner la probabilité d'appartenance de notre image à la classe healthy ou esca.

La seule différence entre nos trois modèles, small, medium et large est la taille de l'image prise en entrée. Respectivement 80x45, 320x180 et 1280x720.

Résumé du model small : **Analyser la structure plus en détail**

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 80, 45, 32)        896       
                                                                 
 activation (Activation)     (None, 80, 45, 32)        0         
                                                                 
 max_pooling2d (MaxPooling2D  (None, 40, 22, 32)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 40, 22, 32)        9248      
                                                                 
 activation_1 (Activation)   (None, 40, 22, 32)        0         
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 20, 11, 32)       0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 20, 11, 64)        18496     
                                                                 
 activation_2 (Activation)   (None, 20, 11, 64)        0         
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 10, 5, 64)        0         
 2D)                                                             
                                                                 
 conv2d_3 (Conv2D)           (None, 10, 5, 64)         36928     
                                                                 
 activation_3 (Activation)   (None, 10, 5, 64)         0         
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 5, 2, 64)         0         
 2D)                                                             
                                                                 
 conv2d_4 (Conv2D)           (None, 5, 2, 32)          18464     
                                                                 
 activation_4 (Activation)   (None, 5, 2, 32)          0         
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 2, 1, 32)         0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 64)                0         
                                                                 
 dense (Dense)               (None, 64)                4160      
                                                                 
 activation_5 (Activation)   (None, 64)                0         
                                                                 
 dropout (Dropout)           (None, 64)                0         
                                                                 
 dense_1 (Dense)             (None, 2)                 130       
                                                                 
 activation_6 (Activation)   (None, 2)                 0         
                                                                 
=================================================================
Total params: 88,322
Trainable params: 88,322
Non-trainable params: 0
```

### 1.5. Entrainement du modèle
On entraine nos modèles sur 50 époques. Une époque correspond à un cycle d'entrainement sur l'ensemble complet de données d'entrainement.
Résultat de l'entrainement :


<figure>
  <img     style="display: block; 
           margin-left: auto;
           margin-right: auto;"
       src="https://github.com/louise17300/embedded_AI/blob/main/Images/training_metrics_small.png?raw=True"
  alt="Métriques d'entrainement, modèle petit">
    <figcaption style="text-align: center;"><b>Métriques d'entrainement, modèle petit</b></figcaption>
</figure>

<figure>
  <img     style="display: block; 
           margin-left: auto;
           margin-right: auto;"
       src="https://github.com/louise17300/embedded_AI/blob/main/Images/training_metrics_medium.png?raw=True"
  alt="Métriques d'entrainement, modèle moyen">
    <figcaption style="text-align: center;"><b>Métriques d'entrainement, modèle moyen</b></figcaption>
</figure>

**Commenter les résultats d'entrainement**

Une fois l'entrainement fais, on enregistre nos modèle sur le format h5.

## 2. Embarquement de notre réseau sur cible
à remplir

Une fois nos 3 modèle entrainé, on se pose la question : 
lequel choisir pour qu'il soit le plus adapté à notre stm32 ?
### 2.1. Test de notre modèle (taille, ressources, ram)
On a mis le model small sur la carte STM32L4R9. La taille du model est inférieure à la taille de la mémoire de la carte, il peut donc être chanrgé sans compréssion.
    
Taille du model sur la carte :
    
![change_queue_position](https://github.com/louise17300/embedded_AI/blob/main/Images/small_model_size.png?raw=True)
    
On a ensuite essayé de mettre le model medium sur la carte. Nous avons eu le message d'erreur suivant :

![change_queue_position](https://github.com/louise17300/embedded_AI/blob/main/Images/erreur_medium.png?raw=True)

On peut effectivement voir que le model fait 1,85MiB pour seulement 640KiB de RAM.

![change_queue_position](https://github.com/louise17300/embedded_AI/blob/main/Images/memoire_medium.png?raw=True)

### 2.2. Compression de notre modèle
Les méthodes de compréssions de cube ne reduisent que la taille d'utilisation de la flash. Elles ne permettent pas de resoudre nos problèmes de RAM.

#### 2.2.1. Prunning
#### 2.2.2. Quantization
### 2.3. Génération de notre code

### 2.4. Flash du code sur la cible

### 2.5. Inférence de notre modèle

## 3. Attaques sur nos modèles 


# Ressources 
Lien de l'éditeur en ligne : https://hackmd.io/5NlZjmKnTIuSRqFOd-GgoQ




