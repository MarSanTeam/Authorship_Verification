# Text-to-Text Transformer in Authorship Verification Via Stylistic and Semantical Analysis

Authorship verification has gained much attention in recent years, due to the emphasis placed on
PAN@CLEF shared tasks.In authorship verification, linguistic patterns are analyzed to reveal information
about the author of two or more texts in order to determine if they are written by the same author.
We describe in this paper our authorship verification submission system and the deep neural network
approach that will allow us to learn the stylistic and semantic features of authors in the contributors to
the PAN@CLEF 2022 event [ 1 ], [2 ], [3]. The system uses the T5 language model as a base embedding
layer, followed by CNN and an attention mechanism to extract local and contextual features. As a result
of studying multiple language models and deep architectures, we obtained an accuracy of 91.79% on our
test dataset which was manually created from a PAN-provided dataset. However, on the official PAN test
set, our system obtained a 58.7% overall score.


![Screenshot from 2023-02-22 15-02-45](https://user-images.githubusercontent.com/86873813/220609791-99a1fe1b-4c69-4f39-88e2-84ba9ff19770.png)



# Usages
#### Train Model:

> python trainer.py

#### Test Model

> python inferencer.py
