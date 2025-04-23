 ## 1
Un PMC est un réseau de neurones artificiels constitué :  
- Couches d’entrée où chaque neurone représente une feature d’entrée.
- Couches cachées composées de neurones en couches denses.
- Couches de sortie.

Pour les MLP, il existe plusieurs hyperparamètres :  
- Le nombre de couches cachées
- Le nombre de neurones par couche
- Le type de fonction d’activation
- Le learning rate
- Le type d'optimiseur (SGD, Adam…)
- Les epochs
- Le Batch size
- La loss function

## 2

Pour un PMC, il existe plusieurs architecture différente à adapter selon le contexte ou la problématique.  
Le choix du nombre de neuronnes et de couches doit dépendre de la compléxité de la tache.  
Par exemple, dans une tache de classification, il faudra configurer la couche de sortie avec du softmax ou du sigmoide afin d'obtenir un format de sortie adéquat. On pourra aussi priviligier une fonction de cross enthropy pour la loss function.  
Pour une régression, on pourrait avoir une couche de sortie avec une activation linéaire et avoir pour loss fonction du MAE ou du MSE.  

# 3  
**Fonction d’activation** : Fonction qui transforme la sortie d'un neuronne.  
**Propagation** : Propagation des calculs couches par couches jusqu'à la couche de sortie.  
**rétropropagation** : Progation de la couche de sortie vers la couche d'entré en calculant les gradiants pour ajuster les poids.  
**Loss-function** : Fonction de mesure de l'erreur entre la prédiction et la valeur réelle.  
**Descente de gradient** : Méthode d'optimisation afin de calculer les gradients pour réduire la loss function.  
**Vanishing gradients** :  Phénomènes des gradients devenant trop petits, ralentissant l’apprentissage dans les couches profondes.  

# 4 
Une fonction ...
Il en existe plusieurs :  
- **ReLU** (Rectified Linear Unit) : max(0, x).  
- **Sigmoid** : 1/(1+e**-x).  
- **Tanh** : Variation de la sigmoid entre -1 et 1.  
- **Softmax** : Convertir une sortie en distribution de probabilité.  

# 5
**Epoch** : Passage complet sur l’ensemble des données d’entraînement.  
**Batch size** : Nombre d’échantillons traités avant une mise à jour des poids.  
**Iterations** : Nombre de batches par epoch = total d’échantillons / batch size.  

# 6
Définit la taille des pas dans la descente de gradient.  
- Trop faible : apprentissage lent.  
- Trop élevé : risque d’instabilité et de divergence.  

# 7
Technique de normalisation des activations intermédiaires pour :  
- Stabiliser l’apprentissage.  
- Accélérer la convergence.  
- Réduire l’effet du vanishing gradient.  

# 8 
Diminutif de “Adaptive Moment Estimation”, Adam est une optimisation algorithmique itérative qui vise à minimiser la loss function. Elle est la combinaison d’AdaGrad et RMSProp. Elle :  
- Maintient des moyennes mobiles des gradients et de leur carré.  
- Auto-ajuste le learning rate par paramètre.  
- Est très utilisée pour sa robustesse.  

# 9
De manière simple, un PMC est un réseaux de neuronnes multicouches qui sont entreconnectés entre l'entrée et la sortie, permettant des systèmes d'apprentissages plus complexes.  