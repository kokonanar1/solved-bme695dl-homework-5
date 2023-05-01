Download Link: https://assignmentchef.com/product/solved-bme695dl-homework-5
<br>
This homework has the following goals:

<ol>

 <li>To create a YOLO-like network architecture for multi-instance object detection and localization in the COCO images.</li>

</ol>

As you saw in your previous homework, detecting and localizing objects is relatively straightforward when you can assume that an image contains only one object instance (or, when you can assume, that you need pay attention to only one object instance in an image). In such cases, you can associate the object label with the entire image.

But how does one solve the problem when an image is allowed to contain multiple object instances and you want to identify and localize all of them at the same time?

At the moment, there exist three major approaches for solving this problem: R-CNN, YOLO, and SSD. See the Preamble in my Week 9 lecture material for a brief introduction to these networks.

What you will program will not be identical to YOLO for the simple reason that the competition grade object detectors are first trained on very large image datasets just to make them aware of what different objects can look like and, subsequently, fine-tuned for object detection work. And even after that, a lot of work goes into optimizing all of the network parameters for maximal performance. Creating such an object detector would be beyond the scope of a homework.

Your goal is merely to demonstrate that you understand the basic logic of the YOLO approach to object detection.

<ol start="2">

 <li>To take your use of the COCO dataset to the next level — by using the bounding boxes for all the object instance in the images.</li>

</ol>

The following steps will prepare you to work with object detection, data loading with annotations, <em>e</em>.<em>g</em>., bounding boxes and labels, and so on.

<h1>2         Getting Ready for This Homework</h1>

Before embarking on this homework, do the following:

<ol>

 <li>Your first step would be to come to terms with the basic concepts of YOLO:</li>

</ol>

Compared to everything you have done so far in our DL class, the YOLO logic is very complex. As I will explain in class, it uses the notion of Anchor Boxes. You divide an image into a grid of cells and you associate N anchor boxes with each cell in the grid. Each anchor box represents a bounding box with a different aspect ratio.

Your first question is likely to be: Why divide the image into a grid of cells? To respond, the job of estimating the exact location of an object is assigned to that cell in the grid whose center is closest to the center of the object itself. Therefore, in order to localize the object, all that needs to be done is to estimate the offset between the center of the cell and the center of true bounding box for the object.

But why have multiple anchor boxes at each cell of the grid? As previously mentioned, anchor boxes are characterized by different aspect ratios. That is, they are candidate bounding boxes with different heightto-width ratios. In my implementation in the RegionProposalGenerator module, I create five different anchor boxes for each cell in the grid, these being for the aspect ratios: 1<em>/</em>5<em>,</em>1<em>/</em>3<em>,</em>1<em>/</em>1<em>,</em>3<em>/</em>1<em>,</em>5<em>/</em>1. The idea here is that the anchor box whose aspect ratio is closest to that of the true bounding box for the object will speak with the greatest confidence for that object.

<ol start="2">

 <li>You can deepen your understanding the YOLO logic by looking at my implementation of image gridding and anchor boxes in Version 2.0.1 of my RegionProposalGenerator module:</li>

</ol>

<a href="https://engineering.purdue.edu/kak/distRPG/RegionProposalGenerator-2.0.1.html">https://engineering.purdue.edu/kak/distRPG/ </a><a href="https://engineering.purdue.edu/kak/distRPG/RegionProposalGenerator-2.0.1.html">RegionProposalGenerator-2.0.1.html</a>

Go to the Example directory and execute the script: multi_instance_object_detection.py

and work your way backwards into the module code to see how it works. In particular, I’d like you to see how I have implemented the notion of anchor boxes in the function

run_code_for_training_multi_instance_detection()

To execute the script multi instance object detection.py, you will need to download the install the following datasets:

Purdue_Dr_Eval_Multi_Dataset-clutter-10-noise-20-size-10000-train.gz

Purdue_Dr_Eval_Multi_Dataset-clutter-10-noise-20-size-1000-test.gz

Note that these datasets need to be installed from the RegionProposalGenerator webpage.

In the dataset names, a string like ’size-10000’ indicates the number of images in the dataset, the string ’noise-20’ means 20% added random noise, and the string ’clutter-10’ means a maximum of 10 background clutter objects in each image.

Follow the instructions on the main webpage for RegionProposalGenerator on how to unpack the image data archive that comes with the module and where to place it in your directory structure. These instructions will ask you to download the main dataset archive and store it in the Examples directory of the distribution.

<h1>             3       Special Note</h1>

For all the previous homework assignments, you could look at the DLStudio code for a reference implementation. But that’s not the case with this homework. The code in the RegionProposalGenerator is NOT a full-blown implementation of the YOLO object detector. Your instructor’s goal in that implementation was simply to show with actual code his best understanding of how anchor boxes are used in YOLO and how one assembles the 8-element encodings for each anchor box for every cell into a 1440-element vector for the target for a 128 × 128 training image. That is, for each input image, you would compare the output of the neural network with this 1440-element tensor.

Therefore, you are likely to run into a larger number of challenges as you do this homework. If you get an early start on this homework, your instructor will be glad to work with you on resolving those challenges. If you seek your instructor’s help regarding this homework, send him an email with the string “ECE695 YOLO” in the subject line. Remember, seeking help from your instructor does not imply getting him to write code for you.

<h2>4     How to Create a COCO Multi-Instance Subset</h2>

Recall that in your HW05, you picked the largest bounding box for choosing a label for any given image. For this homework, you could create a COCO subset similar to Dr. Eval multi-instance dataset, by picking three categories in COCO dataset.

For example, let’s pick the following three COCO classes, bus, truck, and car. One way to create a COCO multi-instance subset is using the following

criteria,

<ol>

 <li>Select subset of COCO images such that all the above three objects are present in them.</li>

 <li>Augment these images such that at least two objects are present.</li>

 <li>For each individual category, <em>i</em>.<em>e</em>., bus, truck and car, select only those images that have at least two instances for that category.</li>

</ol>

Obviously, additional conditions are required to avoid picking overlapping set of images in the above selection criteria.