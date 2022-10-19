# Embedded_AI : attaques sur des modèles de réseaux neuronaux convolutifs embarqués sur STM32F411

## Description du projet
Le projet vise à embarquer une intelligence artificielle capable de reconnaître si une vigne atteinte de la maladie esca à partir d'image. Pour cela nous allons générer trois modèles de réseaux convolutionnels de tailles différentes dont l'objectif est de déterminer à partir d'une image si une vigne est malade. On essaiera ensuite d'embarquer ces modèles, en les modifiant si nécessaire, sur une STM32F411.
Ensuite, une série d'attaques seront effectuées sur nos modèles pour observer leurs fragilités et des mesures seront mises en place pour essayer de les rendre plus robustes.

L'entraînement des modèles se fera sur l'outil en ligne google colab et sur nos ordinateurs portables personnels. On utilisera l'outil STMCubeMx pour générer le code à flasher sur notre STM32F411 et STMCubeIDE pour flasher et faire du débogage.

On utilisera la librairie Keras fonctionnant sur TensorFlow pour la construction de nos modèles.

L'ensemble du projet sera mené sur une carte STM32L4R9 Discovery kit.

## Sommaire
[ToC]
## 1.Génération des modèles
### 1.1. Récupération du dataset
Au sein de ce projet on travaille sur un dataset préexistant. Celui-ci a été collecté lors d'un travail conjoint du département d'ingénierie de l'information de l'université Polytechnique de Marche et de STMicroelectronics.
Le dataset contient 1770 images de feuilles de vignes séparée en deux catégories de taille égales:

- Des feuilles saines (healthy)
- Des feuilles atteintes de la maladie d'Esca (esca). La maladie d'esca est provoquée par la contamination du bois de la vigne par des champignons et bactéries provoquant la nécrose du bois, des symptômes foliaires et progressivement la mort de la plante. 

On récupère le dataset sur https://data.mendeley.com/datasets/89cnxc58kj/1 et on stocke celui-ci sur un google drive. Cela permet de pouvoir y accéder directement via l'outil google colab.

### 1.2. Augmentation du dataset
La performance des modèles que l'on va générer dépend de la qualité et de la quantité de nos données d'entraînement. Par exemple, si l'on a trop peu de données d'entraînement comparée à la taille de notre modèle, celui-ci va avoir tendance à faire de l'overfitting.
Grâce à cette augmentation de la taille de notre dataset, on passe de 1770 images à 24780. Pour chaque image, on vient créer 13 variations de celle-ci. Cette nouvelle base de données sera elle aussi enregistrée sur notre google drive.

Exemple des augmentation faites sur une image :

![change_queue_position](https://github.com/louise17300/embedded_AI/blob/main/Images/augmentations_example.png?raw=True)

### 1.3. Prétraitement de la base de données
Il est nécessaire de redimensionner la taille des images car chaque modèle prend en entrée des images de dimensions différentes.
De plus, on sépare notre base de données en trois parties distinctes :

- *Données d'entrainements* : ce sont les données utilisées lors de l'entraînement de notre modèle.
- *Données de validation* : à la fin de chaque epoch, on vient valider l'apprentissage de notre modèle. Cette validation se fait sur les données de validation.
- *Données de test* : Une fois notre modèle complètement entraîné, on vient tester sa précision avec les données de test.

Si l'on ne fait pas cette séparation et que l'on valide ou test notre modèle avec les mêmes données que celles utilisées lors de l'entraînement, on ne sera pas à même de s'assurer que notre modèle ait bien généralisé et pas seulement mémoriser. On peut ainsi détecter les problèmes d'overfitting.

### 1.4. Création du modèle
Pour nos trois modèles, on utilise un modèle séquentiel. Le modèle séquentiel est approprié pour une pile de couches simples où chaque couche a exactement un tenseur d'entrée et un tenseur de sortie (https://www.tensorflow.org/guide/keras/sequential_model). 
Les trois modèle sont constitués de 5 première surcouches, elles même constituées de trois couches :
- *Conv2D* : permet de faire la convolution sur notre image en entrée et de donner en sortie un tensor. Un tensor représente un tableau d'images
    - Filtres : la première, deuxième et dernière couches possèdent 32 filtres, la troisième et quatrième 64. Donnant en sortie respectivement des tensor de taille (x,y,64), x et y correspondent à la taille de notre image en input.
    - Taille du noyau (kernel_size) : 3 par 3, la fenêtre de notre convolution se fait sur un carré de 3 pixels de côté.
    - Padding : same, signifie que la taille d'un filtre sera la même que la taille de l'image donnée en entrée.
- *Relu* : couche d'activation. Permet de définir la valeur de sortie des neurones en fonction de la fonction d'activation. Ici Relu :

![change_queue_position](https://github.com/louise17300/embedded_AI/blob/main/Images/Relu.png?raw=True)
- *MaxPooling2D* : permet de faire le pooling, donc le groupement de données. Cela permet de donner une invariance pour les petites variations. Le MaxPooling2D permet de prendre la variable la plus grande sur une fenêtre de 2 pixels de côtés
    - taille du tensor (pool_size): (2,2)
    - Padding : same

Suivant ces 5 surcouches se trouve 6 couches :
- *Flatten* : permet d'aplatir notre donnée en 1D.
- *Dense* : Couche connectée de manière dense, tous les nœuds sont connectés à tous les nœuds précédents.
- *Activation* : couche d'activation. Permet de définir la valeur de sortie des neurones en fonction de la fonction d'activation.
- *Dropout* : Permet de mettre aléatoirement certains poids à 0 pendant l'entrainement. Cela permet de prévenir l'overfitting.
- *Dense* : Couche connectée de manière dense, avec seulement 2 nœuds car l'on a 2 classes.
- *Activation* : On utilise Softmax qui permet de donner la probabilité d'appartenance de notre image à la classe healthy ou esca.

La seule différence entre nos trois modèles, small, medium et large, est la taille de l'image prise en entrée. Respectivement 80x45, 320x180 et 1280x720.

Résumé du model small :

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
On peut observer que les couches conv2D sont de la même taille que les couches précédentes. Le padding est bien same. On peut calculer le nombre de paramètres de cette première couche 3x3x3x32+32=896 ce qui nous prouve qu'il y a bien 32 filtres dans cette couche.

### 1.5. Entrainement du modèle
On entraîne nos modèles sur 50 époques. Une époque correspond à un cycle d'entraînement sur l'ensemble complet de données d'entraînement.
Résultat de l'entraînement :


<figure>
  <img     style="display: block; 
           margin-left: auto;
           margin-right: auto;"
       src="https://github.com/louise17300/embedded_AI/blob/main/Images/training_metrics_small.png?raw=True"
  alt="Métriques d'entraînement, modèle petit">
    <figcaption style="text-align: center;"><b>Métriques d'entrainement, modèle petit</b></figcaption>
</figure>

<figure>
  <img     style="display: block; 
           margin-left: auto;
           margin-right: auto;"
       src="https://github.com/louise17300/embedded_AI/blob/main/Images/training_metrics_medium.png?raw=True"
  alt="Métriques d'entraînement, modèle moyen">
    <figcaption style="text-align: center;"><b>Métriques d'entrainement, modèle moyen</b></figcaption>
</figure>

Nous n'avons pas entraîné le modèle large. En effet, ce dernier prenait beaucoup de temps à entraîner. Il dépassait les limites de temps de google colab gratuit et on ne peut pas se permettre de laisser nos ordinateur personnels tourner aussi longtemps car celà nous empêche de travailler les autres matières.

On constate que le model small overfit légèrement. En effet, les accuracy d'entraînement et de validation ne convergent pas. Le modèle médium est donc meilleur.

Une fois l'entraînement fait, on enregistre nos modèles sur le format h5.

## 2. Embarquement de notre réseau sur cible
Une fois nos 2 modèle entraîné, on se pose la question : 
lequel choisir pour qu'il soit le plus adapté à notre stm32 ?

### 2.1. Test de déploiement des modèles
Nous avons donc commencé par tester le déploiement de nos modèles sur la carte STM32L4R9.

On utilise l'outil STM32CubeMx pour configurer la STM32. Lors de la configuration, on ajoute le module X-CUBE-AI, artificial intelligence. Ce module permet de charger un .h5 d'un de nos modèles et de faire une analyse pour savoir quelle place le modèle associé prendra sur la carte.

Le model small n'a pas posé problème. La taille du modèle est inférieure à la taille de la mémoire de la carte, il peut donc être chargé sans compression.
    
Taille du modèle sur la carte :
    
![change_queue_position](https://github.com/louise17300/embedded_AI/blob/main/Images/small_model_size.png?raw=True)
    
X-CUBE-AI nous permet également de valider le model sur notre ordinateur. Les résulats d'un test de validation du model small donnent : 
![change_queue_position](https://github.com/louise17300/embedded_AI/blob/main/Images/small_model_metrics.png?raw=True)

L'accuracy est de 80% ce qui parait cohérent.

On a ensuite essayé le déploiement du modèle médium sur la carte. Nous avons eu le message d'erreur suivant : 

![change_queue_position](https://github.com/louise17300/embedded_AI/blob/main/Images/erreur_medium.png?raw=True)

On peut effectivement voir que le modèle fait 1,85MiB pour seulement 640KiB de RAM.

![change_queue_position](https://github.com/louise17300/embedded_AI/blob/main/Images/memoire_medium.png?raw=True)

### 2.2. Compression de notre modèle
Notre modèle médium ne peut pas être déployé sur la carte tel quel nous avons donc tenté des méthodes de compression afin de réduire sa taille.
#### 2.2.1. Quantization
La méthode de compression utilisée par X-CUBE-AI permet de déployer un modèle quantifié (format entier 8 bits). La quantification est une technique d'optimisation pour compresser un modèle à virgule flottante 32 bits en réduisant sa taille, en améliorant l'utilisation et la latence CPU/MCU, au prix d'une légère dégradation de la précision.

Cette méthode de compression ne réduit que la taille d'utilisation de la flash. Elles ne permettent pas de résoudre nos problèmes de RAM.
#### 2.2.2. Pruning
Nous avons donc décidé d'utiliser une autre méthode de compression : le pruning.

L'objectif du pruning, ou élagage en français, est de limiter la taille d'un réseau de neurones en supprimant certains liens tout en minimisant l'impact sur les performances. Pour cela des algorithmes permettent de déterminer la pertinence des liens afin de retirer ceux de poids faibles, qui n'influencent pas trop l'efficacité du modèle. 

**Si on fait du prunning en parler**

### 2.3. Génération de notre code
Une fois le**s ?** model accepté par STMCubeMx nous avons généré le code associé que l'on a flashé sur la cible.

### 2.4. Inférence sur la cible
Afin de tester notre modèle embarqué nous avons fait l'inférence de notre modèle sur la cible via l'UART.

Nous avons modifié le fichier "Application X-Cube AI" du code précédemment généré afin de régler les paramètres de l'UART permettant la bonne réception des images d'inférence.

Ensuite à l'aide d'un code python "communication_STM32_esca.py" nous avons envoyé des images à la carte.

**resultat de l'inference (image + commantaire)**

## 3. Attaques sur nos modèles 
Une intelligence artificielle est une cible facile pour les hackers. En effet, il est possible de l'attaquer à chaque étape du processus. Il est possible de corrompre les données d'apprentissage, de dégrader le modèle de réseaux de neurones, d'attaquer le réseau pendant l'apprentissage, de fausser les données inférentes,... Les attaques sont très variées et c'est un domaine nouveau dans lequel elles n'ont pas toutes été découvertes. Nous ne pouvons donc pas garantir la sécurité de notre réseaux de neurones. 

Nous allons tout de même tester quelques attaques afin de voir leur efficacité.
### 3.1. Modèle de menace
Afin de savoir quels sont les types d'attaques les plus susceptibles d'arriver, on crée un modèle de menace.

Pour cela il nous faut 
- L'objectif de l'attaquant
- La connaissance du model
- Les étapes du processus que l'attaquant peut attaquer

Les objectifs des d'attaques sur un réseaux de neurones peuvent être de 3 types :
- Attaque de la confidentialité : L'attaquant cherche à récupérer des données confidentielles comme des informations sur la base de données d'entraînement ou sur le modèle en lui-même.
- Attaque de l'intégrité : L'attaquant cherche à tromper le model.
- Attaque de la disponibilité : L'attaquant cherche à diminuer les performances du modèle par exemple en augmentant le temps de réponse du modèle.

Ici, la base de données et le modèle sont en ligne, il est donc peu probable que quelqu'un cherche à faire des attaques sur la confidentialité. Le temps de contamination de la maladie de l'esca étant de 10 ans, il n'est donc pas urgent de savoir qu'un plant est malade. Une attaque sur la disponibilité est donc également peu probable. Nous allons donc fixer l'intégrité comme attaque de l'attaquant.

Comme le modèle utilisé est disponible sur internet, la connaissance de l'attaquant est en boîte blanche. 

On part du principe que l'attaquant n'est pas dans l'équipe qui doit embarquer le modèle mais un simple passant qui voit l'outil dans un champ de vigne. Nous avons entraîné le modèle avant de le mettre sur la carte, l'attaquant ne peut donc pas attaquer le processus d'entraînement mais seulement l'inférence.

Ainsi pour résumer notre modèle de menace : 
- attaque sur l'intégrité
- en boîte blanche
- à l'inférence

### 3.2. Attaque adversarial
Les attaques adversarials sont des attaques sur l'intégrité, à l'inférence qui peuvent se faire aussi bien en boîte blanche qu'en boîte noire (c'est plus facile en boîte blanche). Elles correspondent donc bien à notre modèle de menace.

Le principe d'attaque adversarial est d'ajouter un bruit imperceptible à une image afin de tromper le model. 

Il y a plusieurs normes d'attaques adversarials qui se caractérisent par le types de perturbation : 
- La norme L<sub>0</sub> influe beaucoup sur peu de pixel : Une attaque va changer au maximum $\epsilon$ pixels.
- La norme L<sub>$\infty$</sub> influe peu sur beaucoup de pixel : Une attaque va changer tout les pixels mais la perturbation sur un pixel sera de maximum $\epsilon$.

Nous avons choisit de tester avec un model adversarial de norme L<sub>$\infty$</sub>.

**resultat des attaques (image + commantaire)**


# Ressources 
Maladie de l'esca : https://www.bayer-agri.fr/cultures/esca-eutypiose-et-bda-pathogenes-complexes-et-tres-nuisibles_2296/#:~:text=En%20effet%2C%20se%20pr%C3%A9sentent%20%C3%A0,des%20sympt%C3%B4mes%20de%20la%20maladie et https://www.maladie-du-bois-vigne.fr/Les-maladies-du-bois/L-esca

Lien de l'éditeur en ligne : https://hackmd.io/5NlZjmKnTIuSRqFOd-GgoQ






