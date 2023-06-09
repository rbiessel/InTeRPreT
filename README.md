# InTeRPreT

InSAR Triplet Regression & Prediction Tools

This repository will provide a modular means of predicting, inverting, and analyzing systematic InSAR closure phases. The idea is that this repository may be used in conjunction with other time series estimators such as ML/EMI or even SBAS networks.

I also intend this repository to supersede the experimental code in `CovSAR`.

I will update this README with documentation and requirements once the code is in a mature and usable state.

## Goals:

- Provide a home for the intensity triplet first utilized in `CovSAR`.
- Implement additional triplets, providing a means of ingesting external data such as reanalysis or NDVI
- Be modular with time series packages such as MintPY, GREG, and more
- Implemented at pixel-by-pixel level but scalable to whole stacks (likely two different functions)
- Pixel-wise bootstrapping for significance testing

## Would be Nice:

- Both python and julia support
- Maximum Likelihood Integration
