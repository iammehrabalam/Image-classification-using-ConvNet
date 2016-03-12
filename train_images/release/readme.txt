This file gives documentation for the Fine-Grained Challenge 2013.
(http://sites.google.com/site/fgcomp2013/)


----------------------------------------
Downloading Training Data
----------------------------------------
The training data for the cars, dogs, and shoes domains is available at:
http://sites.google.com/site/fgcomp2013/training

Please download it to the same directory as this readme and extract it.

The training data for aircraft and birds cannot be included with the rest
of the training data due to licensing agreements, so you will have to
download them at their respective websites.

The aircraft data is available at:
http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

and the birds data is available at:
http://www.birds.cornell.edu/nabirds/

Please download both the aircraft and birds data to the same directory
as this readme and extract them. After doing so, they can be merged into
the same format as the rest of the training data by running:

$ python copy_licensed_train.py

If you have downloaded any of the training data in a different location than
specified, you will need to modify the paths in copy_licensed_train.py
accordingly.


----------------------------------------
Downloading Testing Data
----------------------------------------
To obtain the test data, register for the challenge (instructions are available
at the fgcomp2013 website). An email will be sent to you containing a link to
the cars, dogs, and shoes testing data. Please download it to the same
directory as this readme and extract it.

The testing data for aircraft and birds cannot be included with the rest
of the testing data due to licensing agreements, so you will have to
download them at their respective websites.

The aircraft data is available at:
http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
If you have already followed the instructions for downloading the aircraft
training data, you don't need to download it again.

The birds data is available at:
http://birds.cornell.edu/nabirds/testdata

Please download both the aircraft and birds data to the same directory
as this readme and extract them. After doing so, they can be merged into
the same format as the rest of the testing data by running:

$ python copy_licensed_test.py

If you have downloaded any of the testing data in a different location than
specified, you will need to modify the paths in copy_licensed_test.py
accordingly.


----------------------------------------
Metadata/Annotations
----------------------------------------
For all annotation files, .mat and .txt versions have been provided.
Descriptions of the files are as follows:

-class_meta.mat: contains a struct array of information about the classes.
  Each annotation includes three fields:
    class: an integral id for the class unique across the entire challenge
    class_name: a string giving a name or description of the class
    domain_index: the integral id of which domain the class belongs to.
      Each domain (aircraft, birds, cars, dogs, shoes) has a different index.

-class_meta.txt: a csv file, where each row is of the form:
    class,class_name,domain_index
  The format for each field is the same as in class_meta.mat

-domain_meta.mat: contains a struct array of metadata about the domains.
  Each domain annotation includes the fields:
    domain_name: a string giving a name or description of the domain
      (i.e. 'aircraft', 'birds', 'cars', 'dogs', 'shoes')
    domain_index: the integral id of the domain

-domain_meta.txt: a csv file, where each row is of the form:
    domain_name,domain_index
  The format for each field is the same as in domain_meta.mat

-train_annos.mat: contains a struct array of information about each
  training image. Each annotation includes the fields:
    image_index: index of the image (one-indexed).
    bbox: struct containing the fields:
      xmin: Min x-value of the bounding box, in pixels
      xmax: Max x-value of the bounding box, in pixels
      ymin: Min y-value of the bounding box, in pixels
      ymax: Max y-value of the bounding box, in pixels
    class: integral id of the class the image belongs to.
    domain_index: integral id of the domain the training images belongs to.
      Note that 'class' implies the value of 'domain_index'.
    rel_path: relative path to the image, assuming the images have been
      saved in the format as specified above.

-train_annos.txt: a csv file, where each row is of the form:
    image_index,rel_path,domain_index,class,xmin,xmax,ymin,ymax
  The format for each field is the same as in train_annos.mat

-test_annos_track1.mat: contains a struct array of information about each
  testing image for track 1. Each annotation includes the fields:
    image_index: index of the image (one-indexed).
    bbox: struct containing the fields:
      xmin: Min x-value of the bounding box, in pixels
      xmax: Max x-value of the bounding box, in pixels
      ymin: Min y-value of the bounding box, in pixels
      ymax: Max y-value of the bounding box, in pixels
    domain_index: integral id of the domain the training images belongs to.
    rel_path: relative path to the image, assuming the images have been
      saved in the format as specified above.

-test_annos_track1.txt: a csv file, where each row is of the form:
    image_index,rel_path,domain_index,xmin,xmax,ymin,ymax
  The format for each field is the same as in test_annos_track1.mat

-test_annos_track2.mat: contains a struct array of information about each
  testing image for track 2. The only difference between this file and
  test_annos_track1.mat is that this file does not include bounding boxes.
  Each annotation includes the fields:
    image_index: index of the image (one-indexed).
    domain_index: integral id of the domain the training images belongs to.
    rel_path: relative path to the image, assuming the images have been
      saved in the format as specified above.

-test_annos_track2.txt: a csv file, where each row is of the form:
    image_index,rel_path,domain_index
  The format for each field is the same as in test_annos_track2.mat


----------------------------------------
Submission file format
----------------------------------------
Files for submission should be .txt files with the class prediction for
image id M on line M. Examples of files in this format are
train_perfect_preds.txt and train_random_preds.txt

Included in the devkit are two scripts for evaluating total score in the
challenge, eval_train.m and eval_train.py. Examples:

(in MATLAB)
>> [domain_scores, overall_score] = eval_train('train_perfect_preds.txt')
or
(command line)
$ python eval_train.py train_perfect_preds.txt

Test submission files are in the same form.
