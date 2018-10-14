# EmoChat CNN and LSTM Prediction Model

This project was built for the CS6101 Deep Learning Steps Project. We aim to make use of the Twitter data and predict the desired emoji based on the Tweets text. Two models are constructed and we do a comparative study: a 2-layer CNN and a 2-layer LSTM.

## Pre-requisites

The model was built based on Keras (with Tensorflow backend; you may also use PyTorch as well), emoji and gensim. You may want to install those dependencies to test the code.


## Files

For now we only include the CNN model, word vector trainer, and the Jupyter Notebook for live demo. You may refer to cnn.py for more details on the construction of CNN model. The details of the neural networks (CNN and LSTM) can be found inside the Jupyter Notebook.

### cnn.py, cnnl.py, lstm.py, lstml.py

The 4 files contain the training model built based on CNN and LSTM. The "l" appended to the end of cnn and lstm represents the length of the data used for training; "l" means 20_train + 5_train (most frequently used 20 emojis, most frequently used 5 emojis) in total for training, while without "l" it means we are only using 20_train.

### steps.ipynb

The Jupyter Notebook for live demo, as well as explanations for the model. The parameters for CNN and LSTM are specified in the Notebook, so you may refer to the details inside. Additionally, we provide an input box and button for the user to try out the effect of our real-time prediction model: they can input any text, and check out the predicted emoji. Note: to run the live demo you may need the trained model (.h5 files, which are ignored in this repo. You can run the .py file to get them).

### wv.py

The Word2Vec model used to obtain the embeddings based on the text input. Built on top of standard Gensim Word2Vec model. 

### vectors.bin, vectors300.bin

The embedding output files. Specifically, vectors.bin contains embeddings with the dimension of 50, while vectors300.bin gives dim 300. In the implementation we are using the vectors.bin file.

<!-- ### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc -->
