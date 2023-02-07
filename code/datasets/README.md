# Datasets

Please download the datasets from the official repos.

The actual structure of this directory is:

```
.
└── datasets/
    ├── MLDoc/
    │   ├── mldoc-dev.json
    │   ├── mldoc-test.json
    │   └── mldoc-train.json
    ├── PAWS-X/
    │   └── es/
    │       ├── pawsx-dev.json
    │       ├── pawsx-test.json
    │       └── pawsx-train.json
    ├── QA/
    │   ├── MLQA/
    │   │   ├── mlqa-dev.json
    │   │   ├── mlqa-test.json
    │   │   └── mlqa-train.json
    │   ├── SQAC/
    │   │   ├── sqac-dev.json
    │   │   ├── sqac-test.json
    │   │   └── sqac-train.json
    │   ├── TAR-XQuAD/
    │   │   ├── tar-dev.json
    │   │   ├── tar-train.json
    │   │   └── xquad-test.json
    │   └── qa_datasets.py
    └── XNLI/
        ├── xnli-dev.json
        ├── xnli-test.json
        └── xnli-train.json
```


On MLDoc, PAWS-X and XNLI the expected format is jsonline, an example from XNLI:

```
{"sentence1": "Y él dijo: Mamá, estoy en casa.", "sentence2": "Llamó a su madre tan pronto como el autobús escolar lo dejó.", "label": 1}
{"sentence1": "Y él dijo: Mamá, estoy en casa.", "sentence2": "Él no dijo una palabra.", "label": 2}
```

On Question Answering datasets the expected format is a json like the following example (extracted from MLQA):

```
{
    "version": 1.0,
    "data": [
        {
            "title": "Kirguistán",
            "paragraphs": [
                {
                    "context": "El idioma nacional, kirguís, está estrechamente relacionada con las otras lenguas turcas, con las que comparte fuertes lazos culturales e históricos. Kirguistán es uno de los miembros activos del Consejo Túrquico y la comunidad TÜRKSOY. Kirguistán es, también, miembro de la Organización de Cooperación de Shanghai, la Comunidad de Estados Independientes, la Comunidad Económica de Eurasia, el movimiento de Países No Alineados y la Organización de Cooperación Islámica. El 8 de mayo de 2015, se incorporó como país miembro de pleno derecho a la Unión Económica Euroasiática junto con Armenia, Bielorrusia, Kazajistán y Rusia.",
                    "qas": [
                        {
                            "question": "¿Cómo se llama la organización islámica de la que forma parte Kirguistán?",
                            "answers": [
                                {
                                    "text": "la Organización de Cooperación Islámica",
                                    "answer_start": 430
                                }
                            ],
                            "id": "0cd6543a6ccac9874930fd2cf7e1b40ac0afcce8"
                        }
                    ]
                },
                {
                    "context": "El clima en Kirguistán varía según la región. El suroeste del valle de Fergana es subtropical y extremadamente caluroso en verano, con temperaturas que alcanzan los 40 °C. Los piedemontes septentrionales son templados y el Tian Shan varía de continental seco a clima polar, dependiendo de la elevación. En las zonas más frías las temperaturas caen a bajo cero durante unos 40 días en invierno e incluso algunas zonas desérticas experimentan nevadas constantes en este período.",
                    "qas": [
                        {
                            "question": "¿Cuántos días están bajo cero en invierno?",
                            "answers": [
                                {
                                    "text": "40",
                                    "answer_start": 373
                                }
                            ],
                            "id": "4c5d3e150b460d8b01cab7fc43d8343b6d2df1be"
                        }
                    ]
                },
                {
                    "context": "El kirguiso es una lengua túrquica de la rama de Kipchak, estrechamente relacionada con el kazajo, karakalpako y nogayo. Fue escrito en el alfabeto árabe hasta el siglo XX. La escritura latina fue introducida y adoptada en 1928, y posteriormente fue reemplazada por órdenes cirílicas de Stalin en 1941.",
                    "qas": [
                        {
                            "question": "¿En que alfabeto se escribía el kirguiso antes?",
                            "answers": [
                                {
                                    "text": "árabe",
                                    "answer_start": 148
                                }
                            ],
                            "id": "d6732b84ccc296fd04e20e4ac4a2a00817294b74"
                        }
                    ]
                },
                {
                    "context": "Muchos negocios y asuntos políticos se llevan a cabo en ruso. Hasta hace poco, el kirguiso seguía siendo un idioma que se hablaba en el hogar y rara vez se utilizaba durante reuniones u otros eventos. Sin embargo, la mayoría de las reuniones parlamentarias de hoy se llevan a cabo en kirguís, con interpretación simultánea disponible para aquellos que no hablan kirguiso.",
                    "qas": [
                        {
                            "question": "¿Para qué idioma se ofrecen servicios de interpretación?",
                            "answers": [
                                {
                                    "text": "kirguiso",
                                    "answer_start": 82
                                }
                            ],
                            "id": "e3ef1715e6a22099d2a795d2ff2fb68212552391"
                        }
                    ]
                },
                {
                    "context": "El transporte en Kirguistán está severamente limitado por la topografía montañosa del país. Las carreteras tienen que serpentear valles escarpados, desfiladeros de 3000 metros de altitud o más, y están sujetas a deslizamientos frecuentes y avalanchas de nieve. Desplazarse en invierno es casi imposible en muchas de las regiones más remotas y de gran altitud.",
                    "qas": [
                        {
                            "question": "¿A qué están sujetas las carreteras en Kirguistán?",
                            "answers": [
                                {
                                    "text": "deslizamientos frecuentes y avalanchas de nieve",
                                    "answer_start": 212
                                }
                            ],
                            "id": "4628ba458563172cfbb0233f1a293d415dcbb938"
                        }
                    ]
                }
            ]
        },
    ]
}
```
