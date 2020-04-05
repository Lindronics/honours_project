# Meeting notes

## Meeting 1

### Questions

May I add supervisor name to repository?
Which dataset to use?
How to proceed in problem/dataset selection?
FLIR camera?
Are the features truly hyperspectral images? (Might have to adapt description)
Relevant reading?

### Notes

* Use off-the-shelf data or create own (risky)
* Start with literature review/survey
* Pick baselines from data, look at existing code
* Propose improvements
* Predict visible -> infrared?
* Denoising
* Reasons for hyperspectral
* Integration of multiple sources? Joint learning or combining?
* 3D-imaging (depth perception)
* RGB to depth (example dataset from new York)
* Taking multiple sources and integrating them

## Meeting 2

### Agenda

* Very brief progress update
* Feature fusion
  * pixel-based
  * feature-level-based
  * score-level-based
* Dimensionality reduction
* Hyperspectral vs multispectral imaging

### Notes

* Convolutional filter
* Varying optimal complexity
* Flexible model preferred
* Keywords
  * Satellite imaging
  * RGB-D (RGB -> Depth)
  * NYU RGB-D
    * https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
  * Image colorization (RGB from greyscale)
* What problems are solved by hyperspectral (relationship between wavelengths)
* Concatenation of models (together or independently?)
* Resolution issues (noise characteristics)
* TF
* Grey -> RGB
* Input: RGB -> non-linear transformed RGB
  * Predict transformation
* Quantify extra performance added by extra layer
  * Test with RGB (start with 1 layer, add more)
* Priority: pick a problem
  * Follow up next week

## Meeting 3

### Notes

* Project topic: pedestrian detection
* Dive into image fusion techniques (different resolutions)
* Use previous project data as precursor task
* Maybe reproduce previous project exercise for testing
* FLIR One Pro sensor is ordered for creating custom dataset

## Meeting 4

### Agenda

* Object detection network architectures
  * YOLO
  * SSD
  * Other
* Dataset generation
* Using off-the-shelf vs self-made models

### Notes

* Benefits from thermal data
  * Pre-scanning (find hot patches)
  * Pixel segmentation (previous project?)
    * Download pre-trained network for bootstrapping
* Bootstrapping training set
* Pixel segmentation vs bounding boxes
* Calibration issues
  * Infrared offset from visible image
* RGB-Depth as starting point?

## Meeting 5

### Agenda

* Progress report
  * FLIR SDK
  * Raw data extraction from FLIR images

### Notes

* Data hygiene
  * Check alignment
* Applications
  * Biometric auth (hand thermal "footprint")
  * Keyboard trace

## Meeting 6

### Notes

* Bootstrapping, retraining classifier
* Checking alignment over time, verifying that calibration remains accurate

## Meeting 7

### Notes

* Image/Feature augmentation to upscale dataset

## Meeting 8

### Notes

* Start report structure
* Define expected results
  * Will help defining necessary work
* Quality of alignment -> quantify
* Limited scope is fine (time constraints)
* Evaluate possibilities of use

## Meeting 9

### Progress since last time

* Work on quantification of accuracy of image registration/alignment

### Notes

* Transfer learning? Images from internet?
* Autoencoder
* Mismatch thermal/RGB -> security applications

## Meeting 10

### Progress since last meeting

* Built a convolutional autoencoder that can reconstruct thermal signatures of humans from visible light
  *Data quality and overfitting?
  *Integration with classifier?
* Gathered classification dataset
  * 12 classes
  * ~700 samples
  * NN architecture to be determined
    * Try residual blocks?
    * Evaluate different architectures

### Notes

* Skip layers/ U-net for autoencoder
* Add bars/fences/foliage to unobstructed images
* Discuss limitations of dataset
* Variability in data (train/test split)* Sun/shade on walls
* Pre-made pre-trained network (VGG)
* Deep learning with python (chapter 5.3)

## Meeting 11

### Progress since last meeting

* Implemented data augmentation
  * Affine transformation
* Set up data loading pipeline for training
* Trained autoencoder on animals
  * Potential for generating new data samples from existing visible light dataset?
  * Necessary to expand to GAN?
* Attempts at classification

### Notes

* Augmentation
  * Shearing can be problematic
* Artificial data strategy:
  * might help with dark backgrounds, bad conditions
* Analysis
  * Literature survey
  * Research question
  * Rigour
* Evaluation
  * Testing/rigour
  * Mobile implementation (does the model work?)
* Maintaining data

## Meeting 12

### Progress since last meeting

* Captured more data (around 1100 more samples)
* Familiarised myself with the IDA cluster
* Classification experiments
  * Chicken and alpacas perform really poorly
  * Data too noisy?

### Notes

* Talk about bounding boxes, pixel segmentation
* Movement segmentation in real world
* Series of image- image difference
* Mobile app:
  * Basic architecture diagram
  * Battery use/performance stats

## Meeting 13

### Progress since last meeting

* Captured third batch of data
* Ran grid search on different model configurations
  * LWIR only
  * RGB only (worst results)
  * Stacked channels
  * CombSum
  * Late fusion (best results)
* Theory for poor accuracy on alpacas and chicken:
  * these are the only classes with variable colours (brown, white, black)
  * might highlight problems with dataset gathering
  * ethical implications for use on humans?
  * experiment still needs to be conducted

### Notes

* evaluation of light conditions (accuracy function of intensity?)
* mean appropriate?
  * at low intensity, higher weight to LWIR prediction?
* K-Fold CV
  * different parts of training data

## Meeting 14

### Progress since last meeting

* Attempt at transfer learning
  * Visible light
    * Off-the-shelf ResNet with ImageNet weights
    * Much better accuracy (around 0.85-0.9)
  * Infrared
    * ResNet that has been pre-trained on a dataset provided by FLIR One
    * Evaluation is not complete yet

### Notes

* Dissertation draft
  * Collecting dataset -> acquiring? collection is easy. using off-the-shelf is hard
  * Full stop at end of equation!
  * Check underscore
  * Fix references
  * Analyse individual images (dark background, obstruction)

## Meeting 15

### Progress since last meeting

* Deployed and evaluated multispectral ResNet to mobile app
  * Good performance (about 15FPS)
  * Model appears to be working (limited evaluation)
* Evaluated different batch sizes for training
  * I had issues with unstable losses. Dramatically reducing the learning rate helped stabilise the training process.
  * Results are still somewhat conflicting; will have to investigate a bit more.
* Extensive additions to dissertation
  * Sections for aforementioned points
  * Neural network designs
  * Background on deep learning, overfitting and regularisation
  * Evaluation of stratified train-validation-split
* Prepared visualisations and analysis tools
  * Class activation heatmap
  * Dimensionality reduced projection of inputs
  * Loss and accuracy history

### Questions

* Dissertation
  * Batch size / hyperparameter evaluation relevant enough?
  * How much basic background?
    * Machine learning, NNs, etc.
  * How specific should the literature review be?
  * Explanations and concept introductions in later chapters?
  * Mention epidemic?

### Notes

* Label smoothing
* Maybe put batch size, hyperparameters into appendices
* Reference to notebook?
* Positive examples for heat maps
* Weighting of heatmap contributions of branches
* Mobile app concerns
  * Alignment
  * Real-time
  * Sampling rate
* Mobile app evaluation
  * Add framerate discussion
  * Mean+STD, number of samples!
  * Moving device around
  * Screen recording for presentation

## Meeting 16

### Notes

* no individual neuron
* CNN discussion
  * why they do well
  * multiple layers
* overfitting figures maybe unnecessary
* integrate 6.2 and 6.3
* 6.4 K-fold CV
* 6.5.1 more details on alternative evaluation
  * detailed description
* 6.5.2 worst possible frame? histogram

## Meeting 17

### Notes

* Abstract
  * more detail about nature of architectures
  * describe FLIR camera
  * describe dataset size
* VIS - visible light
  * make sure abbreviation is explained
* Brightness x axis
* References
  * exclude URLs, DOIs, ISBNs
  * arXiv?
  * definitely exclude google books
  * IEE transactions is a journal, not a conference
  * CVPR venue in last reference
  * university physics name
* Motivation
  * paragraphs somewhat short, join them up
  * self-driving cars, image-based surveillance, autonomous systems, environmental angle
* Aim
  * explore different ways of incorporating LWIR
  * design options
  * discuss benefits and drawbacks
Move potential apps into motivation?
