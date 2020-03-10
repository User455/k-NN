# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)

# Le dataset principal qui contient toutes les images
print (mnist.data.shape)

# Le vecteur d'annotations associé au dataset (nombre entre 0 et 9)
print (mnist.target.shape)

sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.8)

from sklearn import neighbors

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain, ytrain)

error = 1 - knn.score(xtest, ytest)
print('Erreur: %f' % error)

errors = []
for k in range(2,15):
    knn = neighbors.KNeighborsClassifier(k)
    errors.append(100*(1 - knn.fit(xtrain, ytrain).score(xtest, ytest)))
plt.plot(range(2,15), errors, 'o-')
plt.show()

# On récupère le classifieur le plus performant

knn = neighbors.KNeighborsClassifier(4)
knn.fit(xtrain, ytrain)

# On récupère les prédictions sur les données test

predicted = knn.predict(xtest)

# On redimensionne les données sous forme d'images

images = xtest.reshape((-1, 28, 28))

# On selectionne un echantillon de 12 images au hasard

select = np.random.randint(images.shape[0], size=12)

# On affiche les images avec la prédiction associée

fig,ax = plt.subplots(3,4)

for index, value in enumerate(select):
    plt.subplot(3,4,index+1)
    plt.axis('off')
    plt.imshow(images[value],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title('Predicted: {}'.format( predicted[value]) )

plt.show()
