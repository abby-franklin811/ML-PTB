# ML-PTB

This code was developed for the 2022 Synapse Pre-term birth prediction- Microbiome DREAM Challenge

basicModelsPTB.py generate basic random forest classfiers developed using the PTB Challenge's data

all modelTest files use their respective models to test on validation data

gan.py trains a GAN based on term data, preterm data, or both and outputs simulated samples as a .csv

feature_importance.py simply ranks importance of features for both basic models

ptb_gan_tsne.png is a quick TSNE plot roughly showing GAN simulated data's similarity to real data
