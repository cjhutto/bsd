====================================
BSD - Bias Statement Detector
====================================

.. image:: https://travis-ci.org/cjhutto/bsd.svg?branch=master
    :target: https://travis-ci.org/cjhutto/bsd
    
Bias Statement Detector (BSD) computationally detects and quantifies the degree of bias in sentence-level text of news stories. We incorporate common linguistic and structural cues of biased language, including sentiment analysis, subjectivity analysis, modality (expressed certainty), the use of factive verbs, hedge phrases, and many other features. The model achieved greater than 97% accuracy, and accounted for 85.9% of the variance in human judgements of perceived bias in news-like text. Using 10-fold cross-validation, we verified that the model is able to consistently predict the average bias (mean of 91 human participant judgements) with remarkably good fit. It is fully open-sourced under the `[MIT License] <http://choosealicense.com/>`_ (we sincerely appreciate all attributions and readily accept most contributions, but please don't hold us liable).

* Introduction_
* `Citation Information`_

====================================
Introduction
====================================

This README file describes the model and dataset of the paper:

	|  **Computationally Detecting and Quantifying the Degree of Bias in Sentence-Level Text of News Stories**
	|  (by C.J. Hutto) 
	|  Second International Conference on Human and Social Analytics (HUSO-15). Barcelona, Spain 2015. 
 
| For questions, please contact: 
|     C.J. Hutto 
|     Georgia Institute of Technology, Atlanta, GA 30032  
|     cjhutto [at] gatech [dot] edu 
 

Citation Information
------------------------------------

If you use either the dataset or any of the BSD analysis tools (bias statement lexicons or Python code for extracting linguistic or structural features, or analysing/predicting degree of bias) in your work, please cite the above paper. For example:  

  **Hutto, C.J. (2015). Computationally Detecting and Quantifying the Degree of Bias in Sentence-Level Text of News Stories. Second International Conference on Human and Social Analytics (HUSO-15). Barcelona, Spain 2015.** 


====================================
Python Code Example
====================================

::
	
	statement = "This is a sample sentence to be tested."
	compute_bias(statement)


For a **more complete demo**, go to the install directory and run ``python bias.py``. (Be sure you are set to handle UTF-8 encoding in your terminal or IDE.)

You can also inspect the code for the ``demo_sample_news_story_sentences()`` function for an idea of how to use BS Detector.
