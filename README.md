# Facility_Extractor
### Train a convolutional CNN model base on a text dataset and extract facilities with Supervised Learning

#### This network is trained with a text_format dataset. As the input length is not constant, we need a string tokenizer.
#### First, We tried to use TF-IDF keyword extracting method to get higher logic level in text processing, but finally we took a look at scikit learn and found a better tokenizer.

#### We have to mention that there is a need for multi-label binarizers for each datapoint's label. In this way, we can use a vector of binary numbers, according to our tags (labels).
