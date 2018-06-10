Feature Specific Sentiment Analysis Dataset

Release: 1.0 (27/01/2015)
_______________________________________________________________________________

Data : http://www.mpi-inf.mpg.de/~smukherjee/data/feature-specific-sentiment-analysis.tar.gz

Papers: 
1. http://people.mpi-inf.mpg.de/~smukherjee/research/http://people.mpi-inf.mpg.de/~smukherjee/cicling12-feature-specific-sa.pdf

The data has been used in the following paper:

1. Subhabrata Mukherjee and Pushpak Bhattacharyya
Feature Specific Sentiment Analysis for Product Reviews
Subhabrata Mukherjee and Pushpak Bhattacharyya
Proc. of the 13th International Conference on Intelligent Text Processing and Computational Linguistics, 2012

Contact: 
smukherjee@mpi-inf.mpg.de

_______________________________________________________________________________

---- FILES ----

- README.txt: This README file.

- For each file, the following details are shown below :
-- description
-- schema 
-- example record from each file

---- DATA DESCRIPTION AND FORMAT ----

1. Dataset1 (1257 reviews from different domains annotated in 2 classes - positive or negative)

-- description : This dataset was extracted from the data used by Hu and Liu et. al (KDD 2004). It consists of reviews from varied domains like antivirus, camera, dvd, ipod, music player, router, mobile etc. Each sentence is tagged with a feature and sentiment orientation of the sentence with respect to the feature. In the original dataset, majority of the sentences consisted of a single feature, and had either entirely positive or entirely negative orientation. From there a new dataset was constructed, by combining each positive sentiment sentence with a negative sentiment sentence using connectives (like but, however, although), in the same domain, describing the same entity. For Example, "The display of the camera is bad" and "It is expensive" were connected by but. To use this dataset also cite the following paper.

--- Minqing Hu and Bing Liu, "Mining and summarizing customer reviews", KDD 2004

-- schema : feature $ polarity $ review

-- example record : NIS 2003  $ pos $  NIS 2003 worked like a charm last year. but  It can not be the computer or the owner, since I purchased McAfee Anti-Virus 8 and it installs and works fine with no problems.

---------------------

2. Dataset2 (3834 reviews from different domains annotated in 2 classes - positive or negative)

-- description : This is the original dataset from Hu and Liu et. al (KDD 2004). To use this dataset also cite the following paper.
--- Minqing Hu and Bing Liu, "Mining and summarizing customer reviews", KDD 2004

-- schema : feature $ polarity $ review

-- example record : NIS 2003  $ pos $  NIS 2003 worked like a charm last year. but  It can not be the computer or the owner, since I purchased McAfee Anti-Virus 8 and it installs and works fine with no problems.

---------------------

3. Dataset3 (425 reviews from 3 domains annotated in 2 classes - positive or negative)

-- description : This dataset was extracted from Lakkaraju et. al (SDM 2011). It consists of reviews from varied domains like  laptops, camera and
printers. Each sentence is tagged with a feature and sentiment orientation of the sentence with respect to the feature. To use this dataset also cite the following paper.

--- Himabindu Lakkaraju, Chiranjib Bhattacharyya, Indrajit Bhattacharya and Srujana Merugu, "Exploiting Coherence for the simultaneous discovery of latent facets and
associated sentiments", SDM 2011 

-- schema : feature $ polarity $ review

-- example record : downloads $ neg $  you cannot download the pictures off the phone, motomixer is not on the phone either
_______________________________________________________________________________


BibTex Entry:

@inproceedings{mukherjee2012featurespecific,
  author = {Mukherjee, Subhabrata and Bhattacharyya, Pushpak},
  ee = {http://dx.doi.org/10.1007/978-3-642-28604-9_39},
  isbn = {978-3-642-28603-2},
  pages = {475-487},
  publisher = {Springer},
  series = {Lecture Notes in Computer Science},
  title = {Feature Specific Sentiment Analysis for Product Reviews.},
  volume = 7181,
  year = 2012
}

