# Automated Fake News Detection Using Machine Learning Approaches
Git Repo for the Thesis on Fake News Detection.....

# A Benchmark Study on Machine Learning Methods for Fake News Detection
https://arxiv.org/abs/1905.04749

The proliferation of fake news and its propagation on social media have become a major concern due to its ability to create devastating impacts. Different machine learning approaches have been attempted to detect it. However, most of those focused on a special type of news (such as political) and did not apply many advanced techniques. In this research, we conduct a benchmark study to assess the performance of different applicable approaches on three different datasets where the largest and most diversified one was developed by us.  We also implemented some advanced deep learning models that have shown promising results. We also find that it is easier for clickbait news sources to spread disinformation on health and research related issues. This implies that, although in recent times, the media has focused mostly on combating against unauthentic political news, it should also pay attention to stop the proliferation of false health and research related news for public safety. 

Specifically, in our study, we investigate:

1. Performance of traditional machine learning and neural network models on three different datasets with the largest dataset developed by us. Unlike other two datasets that contain news on the specific topic, we have covered a wide range of topics and accumulated five times more news compared to others.
2. Performance of some advanced models such as convolutional-LSTM, character-level convolutional-LSTM, convolutional-HAN which are not used in fake news detection yet to the best of our knowledge.
3. Performance of different models that have shown promising results on other similar problems.
4. Topic-based analysis on misclassified news, especially deceptive news that is falsely identified as true.

We observe that the performance of models is not dataset invariant and so it is quite difficult to obtain a unique superior model for all datasets. We have also found that traditional machine learning architecture like Naive Bayes can achieve very high accuracy with proper feature selection. On a small dataset with less than 100k news articles, Naive Bayes(with n-gram) can be a primary choice as it achieves similar performance compared to neural network-based high overhead models. Most importantly, our newly explored neural network based models in this fake news detection achieve as high as 95% accuracy and F1-score and exhibit improvement in performance with the enrichment of dataset.
