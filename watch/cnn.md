 ## 1 
L'architecture typique est :
- Une entrée (image, par ex. 32x32x3)
- Une alternance de :
    - Des couches convolutives (extraction de features)
    - Des couches de pooling (réduction dimensionnelle)
- Des couches entièrement connectées (classification)
- Une sortie

LEs hyperparamètres principaux sont :
- La taille et le nombre de filtres
- Le stride, le padding
- La taille du pooling
- Le nombre de couches
- La fonction d'activation, le learning rate, etc.

## 2
Chaque filtre de convolution (ou noyau) glisse sur l’image et effectue des multiplications locales, générant une *Feature Map*.  
Ces filtres apprennent à détecter des motifs (bords, textures, formes).  

## 3
Les fonctions d'activations sont :
- ReLU (la plus utilisée) : Non-linéaire, elle évite le *vanishing gradient*. Elle est rapide à calculer et elle permet d’introduire de la complexité tout en conservant la performance de calcul.  

## 4 
La feature Map est le résultat de l’application d’un filtre sur une image. Chaque carte correspond à une “vue” de l’image selon un motif appris.  

## 5
Le pooling réduit la dimension tout en conservant les informations importantes.  
- Max pooling : prend la valeur maximale
- Average pooling : calcule la moyenne
Cela aide à rendre le réseau invariant aux petites translations.

## 6
À la fin du CNN : Les Feature Maps sont aplaties (flatten) puis envoyées dans une couche dense.  
Cette couche reçoit une représentation vectorielle condensée des informations de l’image pour prédire la classe.   

## 7 
Le CNN exploite la structure spatiale de l’image, cela nécessite moins de paramètres grâce au partage des poids. L'apprentissage de features hiérarchiques (du simple au complexe) est de meilleure qualité. Cela entraîne une réduction des coûts de calcul et meilleure généralisation.