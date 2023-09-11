
# Impact of News Sentiment on Cryptocurrency Market Volatility:

## Table of Contents

- [1. Introduction](#Introduction)
- [1.1. Guidelines to run the project](#Guidelines-to-run-the-project)
- [1.2. Project Manual](#Project-Manual)
- [2. Steps involved in the analysis](#Steps-involved-in-the-analysis)
- [Step 1: Data Collection through Webscraping](#Data-Collection-through-Webscraping)
- [Step 2: Data Cleaning and Pre-processing Techniques](#Data-Cleaning-and-Pre-processing-Techniques)
- [Step 3: Sentiment Analysis](#Sentiment-Analysis)
- [3. Conclusion](#Conclusion)
- [4. Question and Answer](#Question-and-Answer)


## 1. Introduction 
<a name="Introduction"></a>
On June, 2021, Elon Musk tweeted about Cryptocurrency and the price increased immediately for about 10%. And when he tweeted that Tesla would no longer accept Cryptocurrency payments, the price decreased to about 15%. So, news shared in twitter, an important social network seems to affect the volatility of cryptocurrency market.   

Cryptocurrencies represent a rapidly evolving market with a wide range of projects and tokens. Analyzing the market helps researchers and analysts gain insights into the technology, adoption, and use cases of different cryptocurrencies. This research is essential for evaluating the long-term viability of specific projects. Therefore, this study focuses on the impact of news sentiments on cryptocurrency market volatility. 


### 1.1. Guidelines to run the project
<a name="Guidelines-to-run-the-project"></a>
 **main_file.ipynb** contains all the codes and the output. This file imports functions from different other .py files. 

### 1.2. Project Manual
<a name="Project-Manual"></a>

## 2. Steps involved in the analysis 
<a name="Steps-involved-in-the-analysis"></a>

## Step 1: Data Collection through Webscraping
<a name="Data-Collection-through-Webscraping"></a>
Tweets data were collected through web scraping using Selenium. Due to twitters algorithm, it wasn't possible to collect tweets for a longer time periods. Therefore, this project employed for loop and time.sleep() and collected tweets related to Bitcoins for indiviual dates from January 1 2019 to March 30 2021. Historical Price data was taken from kaggle.

Web scraping functions are included in **SRC/Webscraping/ webscraping_functions.py**

## Step 2: Data Cleaning and Pre-processing Techniques 
<a name="Data-Cleaning-and-Pre-processing-Techniques"></a>
Some content for Section 1.

All the functions related to Data cleaning and Pre-processing are included in **SRC/Data_Cleaning_Preprocessing/preprocessing_functions.py**. Additionally, under Data_Cleaning, this project filters the tweets based on the Cryptocurrency keywords. These keywords are added to crypto_corpus.yaml. To an existing keywords list, I added additional words after analysing a sample of tweets 

## Step 3: Sentiment Analysis
<a name="Sentiment-Analysis"></a>
Some content for Section 1.

Sentiment Analysis is done using a pre-trained Hugging face model. The documentation of the model is as follows: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment

The corresponding function is added in **SRC/Sentiment analysis/sentiment_analysis.py**


## 3. Conclusion
<a name="Conclusion"></a>
This project attempts to find if news seentiments affects the direction of Bitcoin's price movements. The results are stored in the **bld** directory. So, after collecting the sentiment scores, this paper adds an additional feature - Polarity. Polarity is calculated as the Geometric Mean of postive and negative scores. Logistic Regression is employed to gauge the impact. The model renders an accuracy score of about 53%. Even after fine tuning the parameters, the accuracy score remained the same. This shows that with Logistic Regression as a baseline model, one could expand this scope of research by employing different other models like Neural Networks, SVM, Random Forests, etc.

The result showed that Polarity had about 37% impact on the price movements, whereas other feature had a weak relationship with price movement. 


## 4. Question and Answer 
<a name="Question-and-Answer"></a>

**1. List five different tasks that belong to the field of natural language processing.**
Most relevant NLP tasks:
- Text classification/sentiment analysis
- Named entity recognition
- Question answering
- Summarization
- Translation

**2. What is the fundamental difference between econometrics/statistics and suprevised machine learning**

Econometrics/statistics | Supervised Machine Learning
------------------------ | --------------------------
Estimate fundamentally unobservable parameters and test hypotheses about them | Predict observable things
Cannot test how well it worked | Can check how well it works
Focus on justifying assumptions | Focus on experimentation,evaluation and finding out what works

**3. Can you use stochastic gradient descent to tune the hyperparameters of a random forrest. If not, why?**

If the function (of the optimization problem) is not differentiable, gradient descent cannot be employed. Ensemble methods like Random Forest are non-differentiable models that contain multiple decision trees. So, they do not have any continuous parameters that can be tuned using Stochastic Gradient Descent.

**4. What is imbalanced data and why can it be a problem in machine learning?**

Imbalanced data refers to a situation in machine learning where some outcomes or classes occur more frequently than others in the dataset. For instance, in a classroom of 49 students and one teacher, predicting whether someone has a PhD results in an imbalanced dataset because the majority of individuals are students without a PhD, and only one person is a teacher with a PhD.

Problems caused by imbalanced data:

- **Model Bias**: Machine learning models can "cheat" by simply predicting the majority class for most or all instances.
- **Ineffective Learning**: Models may not have enough examples of the minority class to understand its characteristics and make accurate predictions.
- **Misleading Evaluation**: In the classroom example, an accuracy of 98% might suggest good performance, but the model's inability to predict the minority class (the teacher with a PhD) is a critical problem.

Class imbalance can be addressed by re-sampling the data. 

**5. Why are samples split into training and test data in machine learning?**

Samples are split into training and test data in machine learning to train the model on one subset for *learning and experimentation* (training data) and *evaluate its performance* on another subset (test data) to ensure it generalizes well to new, unseen data.

**6. Describe the pros and cons of word and character level tokenization.**

***Word level tokenization***

  Pros | Cons
------------------------ | --------------------------
retains the original structure of words |may not handle repeated characters (Variations like "huuuuuge")
offers simplicity in mapping words to integers | Typos like "laern" can result in different integer representations 
doesn't require encoding an extensive vocabulary | Words with different forms (Morphology like "learned") might be assigned distinct integers
Efficient Memory Usage  | Incompleteness: Issues when handling unknown words

***Character level tokenization***
 
 Pros | Cons
------------------------ | --------------------------
Very Simple: easy to implement |Loses Entire Word Structure: less suitable for tasks that require word-level understanding
Tiny Vocabulary Size: Beneficial for memory efficiency | Tokenized Texts Are Very Long:  increase computational complexity and degrade model performance
No unknown words |  lacks the ability to capture the semantic meaning of words or phrases

**7. Why does fine-tuning usually give you a better performing model than feature extraction?**

Fine-tuning usually results in better model performance than feature extraction because it allows the model to adapt to the specific characteristics and nuances of the target task. By adjusting not only the output layer but also some internal layers, fine-tuning tailors the pre-trained model to better fit the task's requirements, leading to improved performance compared to using fixed, pre-trained features.

**8. What are advantages over feature extraction over fine-tuning**

Feature extraction | Fine tuning
------------------------ | --------------------------
Train parameters of classification model. Is relatively faster as we use a pre-trained model and add few layers of classification/regression tasks | Train all parameters. We can adjust output and internal layers to better suit task's characteristics.
Last hidden states contain relevant information "by accident" | Last hidden states are optimized to contain relevant information
Classifier can be anything (e.g.random forrest) | Classifier is a differentiable neural network
CPU is enough as we train only a small portion of the moodel | Very slow without GPU


**9. Why are neural networks trained on GPUs or other specialized hardware?**

GPUs are significantly faster in neural network training due to several key factors. 
- They can load data, such as elements of a matrix, in parallel, which means multiple data points can be processed simultaneously.
- They excel at performing calculations in parallel, allowing for efficient computation of complex operations required for neural network training.
- Moreover, GPUs are equipped with a higher number of floating-point units, enabling them to handle a large number of mathematical operations concurrently, further boosting their speed and efficiency in deep learning tasks.
    
**10. How can you write pytorch code that uses a GPU if it is available but also runs on a laptop that does not have a GPU.**

```javascript
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OurModel()  # Replace with our model class
model.to(device)
```

**11. How many trainable parameters would the neural network in this video have if we remove the second hidden layer but leave it otherwise unchanged.**

- Input Layer to First Hidden Layer:

Weights: 784 (neurons in the input layer) * 16 (neurons in the first hidden layer)

Biases: 16 (neurons in the first hidden layer)

- First Hidden Layer to Output Layer:
  
Weights: 16 (neurons in the first hidden layer) * 10 (neurons in the output layer)

Biases: 10 (neurons in the output layer)

So, if we remove the second hidden layer, the total number of trainable parameters would be:

Parameters = Weights + Biases

Parameters = (784 * 16) + 16 + (16 * 10) + 10

Parameters = 12544 + 16 + 160 + 10

Parameters = 12730

Therefore, the neural network would have 12,730 trainable parameters if we remove the second hidden layer but leave it otherwise unchanged.


**12. Why are nonlinearities used in neural networks? Name at least three different nonlinearities.**

Nonlinearities, in the form of activation functions, are crucial in neural networks to introduce complexity and expressiveness into the model. Without nonlinearities, the composition of multiple linear layers would collapse into a single linear transformation.

i.e., Without nonlinearities, model would be

$W_2(W_1(W_0x + b_0) + b_1) + b_2$

This could be simplified to

$Wx + b$ (one linear model)

 If we only had linear transformations between layers, the neural network would only be able to model linear relationships. It would lack the capacity to capture the intricate patterns and representations required for tasks such as image recognition, natural language understanding, etc.
 
***Activation functions***

- Rectified Linear Unit: ReLU is the most commonly used non-linearity in neural networks
- Sigmoid
- Hyperbolic Tangent (Tanh)

**13. Some would say that softmax is a bad name. What would be a better name and why?**

A potentially better name for "softmax" could be "soft argmax" because softmax is essentially a continuous and smoothed approximation of the argmax function. i.e., When the sizes of vector elements vary significantly, the softmax function tends to behave like an indicator function for the largest element. Thus "soft argmax" is a potentially better name. 

**14. What is the purpose of DataLoaders in pytorch?**

DataLoaders in PyTorch serve to facilitate looping over data in batches and enable parallel data loading.

**15. Name a few different optimizers that are used to train deep neural networks**

- SGD (Stochastic Gradient Descent)
- SGD + Momentum
- Adam (Adaptive Moment Estimation)

**16. What happens when the batch size during the optimization is set too small?**

When the batch size during the optimization is set too small, update become erratic i.e., Training can become noisy and erratic, slow convergence and poor generalization.

**17. What happens when the batch size diring the optimization is set too large?**

Using a very large batch size can lead to increased memory requirements, which may exceed the available GPU memory. It can also slow down the training process because processing a large batch size requires more computation.

**18. Why can the feed-forward neural network we implemented for image classification not be used for language modelling?**

The feed-forward neural network designed for image classification cannot be used for language modeling because it operates on *fixed-size model inputs, lacks memory between inputs*, and  they struggle to capture dependencies across different positions in the input sequence effectively. 

**19. Why is an encoder-decoder architecture used for machine translation (instead of the simpler encoder only architecture we used for language modelling)**

An encoder-decoder architecture is preferred for machine translation over the simpler encoder-only architecture used for language modeling because of the following reasons:
- Handling Varied Sentence Lengths: In machine translation, input and target sentences can differ in length, which an encoder-decoder architecture accommodates efficiently.
- Addressing Word Order Differences: Translations often involve rearranging words, making encoder-decoder models more suitable for capturing and generating translations with varying word orders.

**20. Is it a good idea to base your final project on a paper or blogpost from 2015? Why or why not?**

Not really. The field of deep learning has rapidly evolved, with the introduction of technologies like RNNs in 2015, Google Translate adopting them in 2016, and the revolutionary transformers in 2017. To understand the current advancement and make a more relevant contribution, it's advisable to focus on recent topics and current technologies in the field. While it's essential to focus on recent topics and current technologies for our final projects, it's equally important to understand how these models evolved to appreciate the advancements in the field.

**21. Do you agree with the following sentence: To get the best model performance, you should train a model from scratch in Pytorch so you can influence every step of the process.**

The sentence is not entirely correct. Using pre-trained models and fine-tuning, known as transfer learning, can be more practical and resource-efficient in many cases due to factors like dataset size, task complexity, time constraints, and the ability to leverage knowledge learned from related tasks. This approach often yields competitive or even superior model performance compared to training from scratch.

**22. What is an example of an encoder-only model?**

An example of an encoder-only model is BERT (Bidirectional Encoder Representations from Transformers).

**23. What is the vanishing gradient problem and how does it affect training?**

The vanishing gradient problem refers to a challenge in training RNNs where gradients can either vanish, becoming extremely small and hindering training, or explode, becoming exceptionally large and leading to undefined values. Both of these cases pose significant difficulties during the training process. 

The vanishing gradient problem hinders training by limiting the model's ability to capture long-range dependencies. It makes weight updates for distant time steps difficult, impacting the understanding of complex relationships in sequences. This affects tasks like sentence comprehension and machine translation.

While LSTM (Long Short-Term Memory) networks offer some improvement, they do not completely eliminate this problem. Transformers use residual connections and layer norm to avoid this.

**24. Which model has a longer memory: RNN or Transformer?**

The Transformer possesses a longer memory compared to the RNN. RNNs suffer from the issue of short memory, where their final state retains only vague information about words introduced a long time ago.

**25. What is the fundamental component of the transformer architecture?**

The fundamental component of the transformer architecture is the attention mechanism, which allows the model to focus on different parts of the input sequence when making predictions or encoding information.

