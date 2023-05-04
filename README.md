Download Link: https://assignmentchef.com/product/solved-cs6476-project6-semantic-segmentation-deep-learning
<br>



<strong>Overview</strong>

In this project, you will design and train deep convolutional networks for semantic segmentation.

<h1>Setup</h1>

<ol>

 <li>Install <a href="https://conda.io/miniconda.html">Miniconda</a><a href="https://conda.io/miniconda.html">.</a> It doesn’t matter whether you use Python 2 or 3 because we will create our own environment that uses python3 anyways.</li>

 <li>Download and extract the project starter code.</li>

 <li>Create a conda environment using the appropriate command. On Windows, open the installed “Conda prompt” to run the command. On MacOS and Linux, you can just use a terminal window to run the command, Modify the command based on your OS (linux, mac, or win): conda env create -f</li>

</ol>

proj6_env_&lt;OS&gt;.yml

<ol start="4">

 <li>This will create an environment named “cs6476 proj6”. Activate it using the Windows command, activate cs6476_proj6 or the MacOS / Linux command, conda activate cs6476_proj6 or source activate cs6476_proj6</li>

 <li>Install the project package, by running pip install -e . inside the repo folder. This might be unnecessary for every project, but is good practice when setting up a new conda environment that may have pip</li>

 <li>Run the notebook using jupyter notebook ./proj6_code/proj6.ipynb</li>

 <li>After implementing all functions, ensure that all sanity checks are passing by running pytest proj6_unit_tests inside the repo folder.</li>

 <li>Generate the zip folder for the code portion of your submission once you’ve finished the project using python zip_submission.py –gt_username &lt;your_gt_username&gt;</li>

</ol>

<h1>Dataset</h1>

The dataset to be used in this assignment is the Camvid dataset, a small dataset of 701 images for self-driving perception. It was first introduced in 2008 by researchers at the University of Cambridge [1]. You can read more about it at the <a href="http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/">original dataset page</a> or in the <a href="http://www0.cs.ucl.ac.uk/staff/G.Brostow/papers/Brostow_2009-PRL.pdf">paper</a> describing it. The images have a typical size of around 720 by 960 pixels. We’ll downsample them for training though since even at 240 x 320 px, most of the scene detail is still recognizable.

Today there are much larger semantic segmentation datasets for self-driving, like Cityscapes, WildDashV2, Audi A2D2, but they are too large to work with for a homework assignment.

The original Camvid dataset has 32 ground truth semantic categories, but most evaluate on just an 11class subset, so we’ll do the same. These 11 classes are ‘Building’, ‘Tree’, ‘Sky’, ‘Car’, ‘SignSymbol’, ‘Road’, ‘Pedestrian’, ‘Fence’, ‘Column Pole’, Sidewalk’, ‘Bicyclist’. A sample collection of the Camvid images can be found below:

(a) Image A, RGB              (b) Image A, Ground Truth               (c) Image B, RGB              (d) Image B, Ground Truth

Figure 1: Example scenes from the Camvid dataset. The RGB image is shown on the left, and the corresponding ground truth “label map” is shown on the right.

<h1>1           Implementation</h1>

For this project, the majority of the details will be provided into two separate Jupyter notebooks. The first, proj6_local.ipynb includes unit tests to help guide you with local implementation. After finishing that, upload proj6_colab.ipynb to Colab. Next, zip up the files for Colab with our script zip_for_colab.py, and upload these to your Colab environment.

We will be implementing the PSPNet [3] architecture. You can read the original paper <a href="https://arxiv.org/pdf/1612.01105.pdf">here</a><a href="https://arxiv.org/pdf/1612.01105.pdf">.</a> This network uses a ResNet [2] backbone, but uses <em>dilation </em>to increase the receptive field, and aggregates context over different portions of the image with a “Pyramid Pooling Module” (PPM).

Figure 2: PSPNet architecture. The Pyramid Pooling Module (PPM) splits the <em>H</em>×<em>W </em>feature map into KxK grids. Here, 1×1, 2×2, 3×3, and 6×6 grids are formed, and features are average-pooled within each grid cell. Afterwards, the 1 × 1, 2 × 2, 3 × 3, and 6 × 6 grids are upsampled back to the original <em>H</em>×<em>W </em>feature map resolution, and are stacked together along the channel dimension.

You can read more about dilated convolution in the Dilated Residual Network <a href="https://arxiv.org/pdf/1511.07122.pdf">here</a><a href="https://arxiv.org/pdf/1511.07122.pdf">,</a> which PSPNet takes some ideas from. Also, you can watch a helpful animation about dilated convolution <a href="https://github.com/vdumoulin/conv_arithmetic#dilated-convolution-animations">here</a><a href="https://github.com/vdumoulin/conv_arithmetic#dilated-convolution-animations">.</a>

(a)                                                            (b)                                                            (c)

Figure    3:             Dilation convolution.         Figure    source: <a href="https://github.com/vdumoulin/conv_arithmetic##dilated-convolution-animations">https://github.com/vdumoulin/conv_ </a><a href="https://github.com/vdumoulin/conv_arithmetic##dilated-convolution-animations">arithmetic#dilated-convolution-animations</a>

<h1>Suggested order of experimentation</h1>

<ol>

 <li>Start with just ResNet-50, without any dilation or PPM, end with a 7×7 feature map, and add a 1×1 convolution as a classifier. Report the mean intersection over union (mIoU).</li>

 <li>Now, add in data augmentation. Report the mIoU. (you should get around 48% mIoU in 50 epochs, or 56% mIoU in 100 epochs, or 58-60% in 200 epochs).</li>

 <li>Now add in dilation. Report the mIoU.</li>

 <li>Now add in PPM module. Report the mIoU.</li>

 <li>Try adding in auxiliary loss. Report the mIoU (you should get around 65% mIoU over 100 epochs, or 67% in 200 epochs).</li>

</ol>