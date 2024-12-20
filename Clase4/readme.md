# Proyecto Clase 4
## Miguel Esteban Pinilla Leal - 20191005036

In this challenge, you will explore why the LeNet model in our example consistently achieves around 90% accuracy on the MNIST dataset. Your goal is to hypothesize, experiment, and explain the reasons behind this plateau in accuracy, and possibly suggest improvements.

## 1. Report Your Findings:
* Summarize your observations about why the model is limited to 90% accuracy.
* Include screenshots or plots (e.g., loss curves, accuracy comparisons, or visualized data) to support your conclusions.

### * Hipótesis generales sobre el desempeño limitado

#### 1. Capacidad limitada del modelo (Underfitting)
Primeramente, como se menciono en clase, LeNet es un modelo relativamente simple, diseñado en los años 90 para hardware limitado, lo cual desde un principio quedaba limitado en sus capacidades de respuesta. 

##### Impacto:
- **Número reducido de filtros y capas:** Esto restringe su capacidad para extraer características complejas.
- **Patrones sutiles no capturados:** Puede no detectar detalles importantes para clasificaciones precisas.

---

#### 2. Insuficiente aumento de datos (Data Augmentation)
Se tiene que en modelos basados en aprendizaje automático, es bastante fundamental el aumento de datos, que es una técnica que se usa especialmente en tareas de visión por computadora, donde el tamaño y la diversidad del conjunto de datos son cruciales para entrenar modelos robustos y generalizables. Y es acá donde se puede tener otra hipotesis del valle que tiene el módelo LeNet, y es que, el entrenamiento de este módelo se realiza únicamente con las imágenes originales de MNIST.

##### Impacto:
- **Generalización limitada:** Sin variaciones como rotaciones o distorsiones, el modelo depende de patrones exactos de los datos de entrenamiento.
- **Representación poco realista:** Las imágenes no reflejan posibles escenarios del mundo real, limitando al módelo en su generalización.

---

#### 3. Optimización no eficiente
Por otra parte, también se puede decir que dado que el módelo LeNet utiliza gradiente descendente estocástico (SGD) como optimizador, este se queda bastante facil en la mayor cantidad de mínimos locales que encuntre, obligando al módelo a estancarse en su entrenamiento.

##### Impacto:
- **Propenso a mínimos locales:** Esto puede ralentizar la convergencia.
- **Tasa de aprendizaje fija:** No se adapta a las diferentes fases del entrenamiento.

---

#### 4. Sobreajuste en el conjunto de entrenamiento
A diferencia de grandes módelos de aprendizaje automático como lo son, los DNN's, los CNN's, los RNN's y los Transformers, el módelo LetNet, no hace uso de técnicas de regularización, lo cual hace que el módelo, no haga la tarea de evitar memorizar patrones, haciendo así que se sobreajuste pero no generalice.

##### Impacto:
- **Rendimiento limitado en validación:** Para este caso, aunque MNIST es sencillo, la falta de regularización puede estar impidiendo una generalización adecuada del módelo.

---

#### 5. Limitaciones en el número de épocas
El entrenamiento se realiza por solo 5 épocas, lo que podría ser insuficiente.

##### Impacto:
- **Patrones incompletos:** El modelo podría no optimizar adecuadamente los pesos necesarios para clasificaciones precisas.

---

### * Hipótesis relacionadas con la arquitectura del modelo

#### 1. Cantidad insuficiente de capas convolucionales
Para este caso, como se observo en la explicación en clase, LeNet hace uso de solo dos capas convolucionales.

##### Impacto:
- **Patrones complejos no detectados:** Debido a la poca cantidad de capas convolucionales, las características de alto nivel, como combinaciones de líneas y curvas, pueden no ser aprendidas.

---

#### 2. Filtros limitados en las capas convolucionales
Adicionalmente, el módelo LeNet emplea 6 filtros en la primera capa y 16 en la segunda.

##### Impacto:
- **Capacidad de representación reducida:** Lo anterior, puede hacer que, el modelo no capte una diversidad suficiente de patrones.

---

#### 3. Ausencia de normalización entre capas
Por otra parte, dado que, LeNet no incluye técnicas modernas como Batch Normalization, este modelo se puede estar quedando en la generalidad del problema y no lo este abordando de fondo.

##### Impacto:
- **Instabilidad en los gradientes:** Lo anterior, puede ser problemático en una arquitectura profunda como es posible que se tenga.

---

#### 4. Falta de bloques residuales (Residual Connections)
Otra de las hipotesis importamtes que se puede plantear, es que, LeNet carece de conexiones residuales, comunes en arquitecturas modernas como lo fue en su momento el boom de ResNet, que empezo a utilizar bloques residuales, en los que las capas de la red profunda recibia información no solo de la capa anterior, sino también de una "conexión de atajo" que se salta una o varias capas.

##### Impacto:
- **Desvanecimiento del gradiente:** Esto puede limitar la efectividad del aprendizaje al incrementar la profundidad de la red.

---

### * Hipótesis relacionadas con las funciones de activación

#### 1. Uso de funciones de activación desactualizadas (tanh)
LeNet utiliza **tanh** como función de activación en todas las capas excepto la última, lo cual genera un problema relativamente importante,y es que dado que tanh es una función sigmoide, esta mapea los valores de entrada en un rango de [-1, 1]. Y debido a su forma en "S", cuando la entrada es muy grande o muy pequeña, la derivada de tanh se aproxima a cero. Es acá donde nace un fenomeno que se llama **desvanecimiento del gradiente**. Y es que, durante la retropropagación, cuando los gradientes son multiplicados a través de muchas capas, los gradientes pueden volverse extremadamente pequeños, lo que dificulta el ajuste de los pesos de las capas más profundas de la red.

##### Impacto:
- **Desvanecimiento del gradiente**

---

#### 2. Función de activación final no ideal
La capa de salida utiliza softmax, estándar para clasificación, pero con limitaciones, entre estas, se tiene que, Softmax maximiza las probabilidades de todas las clases sin tener en cuenta qué tan difícil es distinguir entre algunas de ellas, lo que puede ser problemático cuando existen clases difíciles de clasificar o clases con muy pocos ejemplos.

##### Impacto:
- **Correlaciones no lineales complejas:** Softmax podría no ser suficiente para manejar relaciones intrincadas entre clases.

---

### * Curva de Loss y Accuracy del modelo original
![gra1](https://pbs.twimg.com/media/Gdb1_vzXoAA-Lf-?format=png&name=small)
![gra1](https://pbs.twimg.com/media/Gdb3BhJW8AAhs53?format=png&name=small)


De lo anterior, es posible visualizar el valle en el que se queda constante el modelo, esto en la imagén del accuracy


---




## 2. Experiment Results:
* Share the changes you made (if any) to improve accuracy.
* Provide before-and-after accuracy metrics and explain why your adjustments worked (or didn’t work).

### * A.1 Cambios efectuados primer experimento

Tomando el código del script original, se hicieron los siguientes cambios:

#### 1. **Funciones de activación**
Se cambió la función de activación de `tanh` a `relu` en todas las capas convolucionales y densas.

#### 2. **Número de filtros**
Se incremento el número de filtros en las capas convolucionales para capturar características más complejas en las imágenes. Los cambios son los siguientes:
- **Primera capa convolucional**: 32 filtros (anteriormente 6).
- **Segunda capa convolucional**: 64 filtros (anteriormente 16).

#### 3. **Número de unidades en capas densas**
Se aumento el número de unidades en las capas densas para mejorar la capacidad de representación del modelo:
- **Primera capa densa**: 512 unidades (anteriormente 120).
- **Segunda capa densa**: 256 unidades (anteriormente 84).

#### 4. **Tamaño del kernel**
Se utilizó un tamaño de kernel más pequeño de `3` en lugar de `5` en las capas convolucionales. Esto con el objetivo de intentar ayudar al modelo a obtener mejores resultados sin perder resolución en las características extraídas.

### * A.2 Cambios efectuados segundo experimento

Tomando el código del script original, se hicieron los siguientes cambios:


#### 1. Funciones de activación
Nuevamente se cambió la función de activación de `tanh` a `relu` en todas las capas.

#### 2. Número de filtros
Nuevamente se incrementa el número de filtros, pero esta vez con otros valores y con una capa adicional. Así:

- **Primera capa:** 32 filtros (anteriormente 6).
- **Segunda capa:** 64 filtros (anteriormente 16).
- **Tercera capa:** 128 filtros (nueva capa añadida).

#### 3. Tamaño del kernel
Nuevamente se redujo el tamaño del kernel a 3 en lugar de 5.

#### 4. Capas densas
También, se aumentó el número de unidades en las capas densas, pero con otros valores. Así:

- **Primera capa densa:** 512 unidades (anteriormente 120).
- **Segunda capa densa:** 256 unidades (anteriormente 84).

#### 5. Optimizador
Se cambió el optimizador a `adam` para mejorar la convergencia.


### * A.3 Cambios efectuados tercer experimento

Para este caso, se tomo el codigo original y se intento  mantener una buena cantidad de los parámetros originales iguales, porque los experimentos anteriores no funcionaron totalmente bien.

#### 1. Funciones de activación
Nuevamente se cambió de `tanh` a `relu` en todas las capas para mejorar la convergencia.
#### 2. Épocas
Se incrementaron las épocas de 5 a 15 para asegurar un mejor entrenamiento del modelo.
#### 3. Parámetros generales
Se mantuvieron los parámetros principales del modelo, como el optimizador (`sgd`), la función de pérdida (`categoricalCrossentropy`), y las capas originales.


### * B.1 Curvas comparativas de precisión obtenidas dado el primer experimento

Para este primer experimento se obtuvo la siguiente curva:

![gra1](https://pbs.twimg.com/media/GdcFGIwXMAAXt7p?format=png&name=small)

Para este primer experimento, el código puede no haber funcionado correctamente por varias razones. En primer lugar, pudieron haber problemas al procesar las imágenes del conjunto de datos MNIST debido a la falta de manejo adecuado posiblemnte de los datos, o el uso incorrecto de las coordenadas para obtener las imágenes del lienzo. Por otra parte, la asignación de los datos de las imágenes a datasetBytesBuffer podría no haberse hecho correctamente, ya que la conversión de los valores de píxeles podría no haberse realizado de manera eficiente. Finalmente, segun algunos fallos, también es posible que haya fallos relacionados con la configuración de TensorFlow.js o con la memoria al tratar de cargar y entrenar el modelo con los datos.


### * B.2 Curvas comparativas de precisión obtenidas dado el segundo experimento

Para este segundo experimento se obtuvo la siguiente curva:
![gra1](https://pbs.twimg.com/media/GdcFJvYW8AAKu5M?format=png&name=small)


Para este experimento, el código que se probo, no genero buenos resultados debido a un número bajo de épocas (5), aunque mas adelante se aumentaron y no dieron alguna mejoria. Además, pudo tambien hacer falta, una normalización adecuada de los datos y también, la ausencia de técnicas de regularización como Dropout pudo haber afectado el rendimiento. 

### * B.3 Curvas comparativas de precisión obtenidas dado el tercer experimento
Para este tercer experimento se obtuvo la siguiente curva:

![gra1](https://pbs.twimg.com/media/GdcFRDpWQAAHCR3?format=png&name=small)


Para este experimento en el que se hizo la tercer parte de cambios, el modelo no logró superar el 90% de precisión probablemente debido a varios factores. En primer lugar, el uso del optimizador sgd (gradiente descendente estocástico) podría estar limitando la convergencia del modelo, ya que optimizadores como Adam son más eficaces para este tipo de tareas auqnue anteriormente no funciono como se queria. Además, el uso de averagePooling2d en lugar de maxPooling2d podría estar restando rendimiento, ya que maxPooling2d tiende a ser más efectivo para la extracción de características. Otro punto es que el número de filtros en las capas convolucionales (6 y 16) es relativamente bajo, lo que limita la capacidad del modelo para aprender características complejas de las imágenes, aunque anterimente se probo con mayores valores, la tasa de perdida aumentaba y por eso no se continuo por ese camino. Asimismo, las unidades en las capas densas (120 y 84) podrían no ser suficientes para capturar la complejidad de los datos. 

---

## 3. Propose Solutions:
* Improvement of the current implementation that could help achieve higher accuracy.

Para este caso, si bien no se obtuvo lo esperado con los experimentos anteriores, una última solución que se podria probar, es cambiar directamente el modelo usado por ejemplo, por los siguientes:

* ResNet (Redes Residuales): Las cuales son muy efectivas para tareas de clasificación de imágenes, especialmente con redes profundas.
* DenseNet: En donde, este modelo, utiliza conexiones densas entre capas, lo que permite un mejor flujo de gradientes y puede mejorar la precisión.
* InceptionNet: Por otra parte, este modelo utiliza bloques de diferentes tamaños de convolución, lo que le permite capturar características de múltiples escalas.
* Redes Neuronales Profundas (DNN): FInalmente, se puede probar con redes completamente conectadas profundas combinadas con técnicas de regularización como dropout.


Aún así, un modelo que podria servir teniendo en cuenta todos los cambios que se efectuaron es uno que combine los dos priemros experimentos, con el último experiemnto, de los primeros se puede tener en cuenta el cambio en los parámetros como el tamño de los filtros y del kernel, y por otro lado tomar la metodología del tercero en relación a aumentar el número de épocas. 

De lo anterior, a pesar de que se puso a correr, este es muy lento para cambiar de época y no se logro visualizar el resultado a tiempo, aún así, el archivo html se subira el repositorio.







