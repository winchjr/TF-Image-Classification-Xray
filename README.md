# TF-Image-Classification-Xray

A project to classify NIH chest x-ray images by disease using neural nets.

Original:
https://www.kaggle.com/datasets/nih-chest-xrays/data

TFRecord:
https://www.kaggle.com/datasets/nickuzmenkov/nih-chest-xrays-tfrecords

# Purpose

X-rays are cheap to create and harm the patient less than other methods, but difficult to inspect at scale. Having automated, scalable machine learning models that can correctly predict disease classification would be ideal to make health diagnostics cheaper, faster, and less harmful to the patient.

# General Outline

The code utilizes two approaches: one of creating multiple nets (each with 2-3 diseases, plus the "no findings" category), and the other of creating a multi-label output for all 15 diseases. Tensorflow was used to train the net.

This project took advantage of having access to a supercompute environment, but even still training a net on 110k images is no small feat. Initially I broke it up into multiple models, but I learned how to use checkpointing and TFRecords to train the model which greatly reduced the RAM usage.

# Input

- 110k chest x-ray images (in both raw image form)
- Supplementary variables (patient age, disease(s), ID, follow up #, etc)
- TFRecord condensed both of the above into one data form

# Output

- 5 neural net models with binary output (yes/no this image has the labelled disease)
- 1 neural net with multi-label output, with 15 possible labels

# Future Improvements

- I did not finish the TFRecord sections yet. There is a memory leak error causing too much RAM to be used, although I've not found where it is at yet.
- Data augmentation 
- Optimization after doing the grid search
