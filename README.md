# TFG_PabloGarciaBolivar
HERRAMIENTA DE EXTRACCIÓN AUTOMÁTICA DE DATOS DE PÓLIZAS DE SEGUROS
 HERRAMIENTA DE EXTRACCIÓN AUTOMÁTICA DE DATOS DE PÓLIZAS DE SEGUROS
Autor: García Bolívar, Pablo. 
Director: Durán Olivencia, Miguel Ángel.  
Entidad Colaboradora: ICAI – Universidad Pontificia Comillas

RESUMEN DEL PROYECTO 
Empresas de todo el mundo generan diariamente enormes cantidades de documentos que ralentizan el trabajo de sus empleados y cuya información generalmente queda archivada para siempre. Analizar dichos documentos con herramientas de Document AI y extraer su información de manera automatizada ayuda a mejorar el rendimiento empresarial. Sin embargo, ninguna de las soluciones existentes está centrada en pólizas de seguros, o requieren de cientos de pólizas anotadas manualmente para su correcto funcionamiento. Por ello, en este Trabajo de Fin de Grado se presenta una nueva herramienta de extracción automática de datos de pólizas de seguros, construida a partir de modelos de aprendizaje automático, que permite al usuario analizar sus pólizas sin necesidad disponer documentos anotados.
Palabras clave: Document AI, Póliza de seguro, Herramienta de extracción de datos, Inteligencia Artificial
1.	Introducción
La inmensa cantidad de documentos generados, enviados y recibidos diariamente por las empresas tiene un impacto negativo en su rendimiento. Por un lado, los trabajadores deben emplear enormes cantidades de tiempo buscando información en documentos para satisfacer las necesidades de los clientes. En concreto, según un estudio realizado por ABBYY, en Reino Unido un 27% de los empleados perdían un día laboral entero por semana buscando dicha información. Por otro lado, de no ser extraída, la información contenida en dichos documentos, muchas veces de gran importancia para las empresas, quedará archivada para siempre. Por ello, la extracción de datos de documentos juega un rol muy importante en la mejora del rendimiento de las empresas.
La extracción manual de dicha información resulta ineficiente y poco escalable. Sin embargo, el Document AI, al emplear modelos de Inteligencia Artificial, ofrece una solución mucho más precisa y escalable, permitiendo a las empresas analizar documentos de manera automatizada. Los principales modelos utilizados en este ámbito están basados en la arquitectura de los Transformers. Generalmente, estos modelos son pre-entrenados con aprendizaje autosupervisado en tareas generales, y más tarde, afinados en tareas concretas con pequeños datasets de documentos anotados.
El creciente número de empresas que demandan los servicios de Document AI hace ver que se trata de un sector al alza. Empresas como Amazon, Google o Microsoft ofrecen sus servicios de extracción de datos en forma de plataformas de pago alojadas en la nube. Sin embargo, ninguna de estas soluciones está centrada en pólizas de seguros, o requieren de cientos de pólizas anotadas manualmente para su correcto funcionamiento.

2.	Definición del proyecto
En este TFG se desarrolla una herramienta de extracción de datos de pólizas de seguros, construida a partir de modelos de aprendizaje automático, que solventa las carencias de las soluciones existentes. Esta herramienta debe ser capaz de extraer: (1) todo el texto presente en el documento, (2) todos los pares de datos relacionados y (3) elementos clave definidos por el usuario; todo ello sin necesitar entrenamiento por parte del usuario. Esta herramienta está escrita en Python y ejecutada en un cuaderno de Google Colab.
Los modelos de Inteligencia Artificial escogidos para el desarrollo de la herramienta comparten una serie características comunes. En primer lugar, son modelos de código abierto. En segundo lugar, deben poder ser descargados listos para su uso, lo que implica que deben de estar ya pre-entrenados y afinados. En tercer lugar, deben ofrecer buenos resultados en sus respectivas tareas, de no ser el estado del arte. Cumpliendo con estas características, el modelo LayoutLMv2 es el pilar central de la herramienta, y se utiliza con el objetivo de etiquetar cada token encontrado en el texto. Este modelo, desarrollado por Microsoft basado en la arquitectura BERT, incorpora información sobre el texto encontrado en el documento, su posición 2-D e información visual del mismo. A parte, también se utiliza el modelo PEGASUS de Google afinado en un dataset de comunicados y reclamaciones de litigios de la SEC. Este modelo tiene el objetivo de resumir párrafos de texto encontrados en el documento, con el fin de presentar la información lo más organizada y resumida posible. Además, se utiliza el modelo all-MiniLM-L6-v2 para hallar la similitud entre dos oraciones. Esto ayuda a extraer elementos clave del documento. Por último, se va a emplear la herramienta OCR Pytesseract, para extraer el texto del documento.
3.	Descripción de la herramienta desarrollada
El proceso de extracción de datos desarrollado en este TFG comienza por importar el documento que se desea analizar. Este documento debe contener una sola página y estar en formato imagen. En la Figura 1 se muestra la arquitectura de la herramienta desarrollada y, por tanto, el proceso de análisis que seguirán los documentos introducidos.
 
 
Figura 1. Arquitectura de la herramienta desarrollada.
El documento introducido es procesado en primer lugar por LayoutLMv2, que etiquetará cada token con una de las etiquetas {‘Other’, ‘Header-Beginning’, ‘Header-Intermediate’, ‘Question-Beginning’, ‘Question-Intermediate’, ‘Answer-Beginning’, ’Answer-Intermediate’}, como puede verse en la Figura 2. Estas etiquetas aportan información muy valiosa sobre el tipo de cada token y las posibles relaciones entre ellos. Sin embargo, LayoutLMv2 no establece dichas relaciones y etiqueta palabras de manera individual, pero no frases. Por ello, el siguiente paso es formar frases a partir de las palabras etiquetadas.
 
Figura 2. Ejemplo de etiquetado de una documento con LayoutLMv2.
Con este objetivo en mente, se ha creado una función que determina si una palabra y la de su derecha pertenecen a la misma frase o no. Dos palabras pertenecerán a la misma frase cuando estén una al lado de la otra y sean del mismo tipo {‘Other’, ‘Header’, ‘Question’, ‘Answer’}. Una vez formadas las frases, se forman a continuación párrafos siguiendo un proceso muy similar. Los párrafos estarán compuestos por frases del mismo tipo, que estén una encima de la otra. Con los párrafos ya formados, en primer lugar, se procede a extraer el texto de cada uno de ellos con la herramienta OCR Pytesseract y a guardarlo dicho texto en el archivo SoloTexto.csv. Este archivo contendrá todo el texto encontrado en el documento, cumpliéndose así el primer objetivo propuesto. 
Por otro lado, los párrafos de tipo ‘Other’ son procesados por el modelo legal-PEGASUS, que los resume sin que exista pérdida de información relevante. Paralelamente, los párrafos, esta vez todos ellos, y no solo los de tipo ‘Other’, son procesados por una función que encuentra relaciones formando pares con el formato ‘Pregunta + Respuestas’. Esta función busca estructuras determinadas partiendo de la hipótesis de que la respuesta asociada a una pregunta se encontrará a su derecha o hacia abajo. Las relaciones encontradas por esta función junto con los párrafos resumidos de tipo ‘Other’ son guardados en un nuevo archivo llamado Resumen.csv, que, como su propio nombre indica, sirve a modo de resumen de toda la información contenida en el documento. 
Por último, la herramienta analiza la similitud entre las preguntas de los pares ‘Pregunta + Respuestas’ y una serie de diccionarios con elementos que se pretenden extraer del documento. La herramienta incorpora de manera predeterminada tres diccionarios: un diccionario genérico a todas las pólizas de seguros, un diccionario específico para seguros del automóvil y un diccionario específico para seguros del hogar. Sin embargo, la creación de nuevos diccionarios personalizados es posible. Para cada elemento encontrado en cada uno de los diccionarios se selecciona la pregunta con la que tenga mayor similitud. De esta manera, la herramienta es capaz de encontrar los elementos del documento que más se parecen a cada elemento que se desea extraer de él. Todos estos elementos extraídos son guardados en un último archivo llamado ElementosClave.csv.
4.	Resultados
En este análisis de resultados se van a analizar, en primer lugar, los resultados obtenidos al procesar dos pólizas de seguros distintas con la herramienta desarrollada en este TFG. Posteriormente, se van a comparar dichos resultados con los obtenidos al analizar las mismas pólizas con la herramienta Document AI de Google.
En la primera tarea objetivo, la extracción de todo el texto presente en el documento, la herramienta desarrollada en este TFG es capaz de extraer correctamente el 89% de las frases en una póliza y el 60% de la otra. Contando cada póliza con 82 y 77 frases a extraer, respectivamente. A la hora de formar relaciones entre elementos del documento, la herramienta extrajo 27 relaciones de la primera póliza y 16 de la segunda. En la Tabla 1 se muestran parte de las relaciones extraídas de la primera póliza.
PREGUNTA	RESPUESTA 1	RESPUESTA 2
Pollcy number	12345~67-89	
Effective	(01/04/2019 12:07 AM.	
Expiration	06/01/2019 12:01AM	
Named insured(s)	Jack Smith	Jane Smith
Underwritten by	Farmers Insurance	6301 Owensmouth Ave.
Policy premium + fees:	$798.48	
Veh. # year/make/model/vin	2019 Porsche Macan 4D 4wD-	
Bodily injury	$250K each person/$500K each incident	
Property damage	$100K each incident	Included
Medical coverage	$5,000 each person	$66.50
Deductible	3500	ACV
Collision	$1,000	ACY
Tabla 1. Relaciones extraídas por la herramienta.
Por último, en la tarea de extracción de elementos clave, la herramienta extrajo 12 elementos de los 14 elementos objetivo de la primera póliza y 2 de 9 en la segunda póliza. En la Tabla 2 se pueden ver algunos de los elementos extraídos de la primera póliza de seguro.
DICCIONARIO GENÉRICO - SEGURO DEL AUTOMÓVIL	
Policy number	12345-67-89
Name insured	Jane Smith
Effective date	01/01/2019 0:01
Premium	3797.6
Address	1234 Main St
Limit	$250K each person/$800K each incident
Tabla 2. Elementos clave extraídas por la herramienta.
Por su parte, Document AI de Google, utilizando la plantilla de documento genérico, es capaz de extraer a la perfección el texto mediante OCR, sin cometer errores. Sin embargo, a la hora formar relaciones, esta herramienta solo formó 18 relaciones en la primera póliza y 2 en la segunda. Por último, la plantilla genérica utilizada para el análisis de estas pólizas no ofrece la posibilidad de extraer elementos clave, y la creación de una platilla personalizada que sí lo permita requeriría de al menos 200 pólizas con los elementos clave anotados manualmente.
Se puede concluir que la herramienta de Google incorpora una gran herramienta de OCR, pero tiene problemas a la hora de formar relaciones y no ofrece la posibilidad de extraer elementos clave. En cambio, la herramienta desarrollada en este TFG establece de manera satisfactoria relaciones entre elementos del documento y es capaz de extraer gran cantidad de los elementos clave objetivo; sin embargo, tiene problemas a la hora de extraer el texto con precisión. Cabe remarcar que esta falta de precisión a la hora de extraer texto empeora también los resultados obtenidos en el resto de las tareas.
5.	Conclusiones
Como conclusiones generales, se ha podido probar la eficacia de los modelos basados en Transformers, entrenados con el método de pre-entrenamiento más afinado. Estos modelos destacan por su capacidad de entendimiento del lenguaje humano y por la versatilidad de sus aplicaciones. También se ha podido comprobar cómo introducir elementos visuales y de posición aumenta el rendimiento de los modelos en tareas del Document AI. Además, se ha observado de manera práctica que una herramienta precisa de OCR es un elemento clave para este tipo de herramientas, y se le debe dar prioridad.
Como trabajos futuros para mejorar la herramienta presentada en este TFG se aconseja, en primer lugar, sustituir LayoutLMv2 por el recién publicado LayoutLMv3, ya que este último ofrece mejores resultados al etiquetar tokens. En segundo lugar, puede resultar interesante la adaptación de la herramienta a otros idiomas, especialmente a español. Esto se puede conseguir incorporando el modelo LayoutXLM, la versión plurilingüe de la misma familia. En tercer lugar, la mejora en la implementación y uso de Pytesseract en esta herramienta sería muy beneficiosa y mejoraría significativamente los resultados obtenidos. Por último, sería conveniente el desarrollo de una función que permita analizar documentos de diversas páginas, así como analizar documentos por lotes.

6.	Referencias
Bao, H., Dong, L., & Wei, F. (2021). BEiT: BERT Pre-Training of Image 			Transformers. arXiv:2106.08254.
Cui, L., Xu, Y., Lv, T., & Wei, F. (2021). Document AI: Benchmarks, Models and Applications. arXiv:2111.08609v1.
Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transfromers for Languaje Understanding. arXiv:1810.04805v2.
Huang, Y., Lv, T., Cui, L., Lu, Y., & Wei, F. (2022). LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. arXiv:2204.08387.
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., . . . Polosukhin, I. (2017). Attention Is All You Need. arXiv:1706.03762.
Walker, A. (2021). Document Searches Can Waste 100s of Hours. Obtenido de https://insurance-edge.net/2021/11/10/document-searches-can-waste-100s-of-hours/
Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. arXiv:2002.10957v2.
Xu, Y., Xu, Y., Lv, T., Cui, L., Wei, F., Wang, G., . . . Zhou, L. (2021). LayoutLMv2: Multi-modal Pre-training for Rich Document Understanding. arXiv:2012.14740v3.
Zhang, J., Zhao, Y., Saleh, M., & Liu, P. J. (2020). PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization. arXiv:1912.08777v3.
