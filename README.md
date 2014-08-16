unshred
=======
Experiments on [Darpa Shredder Challenge](http://archive.darpa.mil/shredderchallenge/) tasks using Python &amp; OpenCV.

### [Demo](http://dchaplinsky.github.io/unshred/)

### Installation
 * Install OpenCV using your favorite package manager or by following [instructions from openCV site](http://opencv.org/downloads.html). Make sure that you have 2.4.x. This code doesn't work with 2.3.x
 * [Install pip](http://pip.readthedocs.org/en/latest/installing.html#install-pip)
 * Checkout codebase from github.
 * Run ```pip install -r requirements.txt``` (if you are using virtualenv, make sure you've created your env with ```--system-site-packages``` option so you'll have an access to openCV bindings which are installed globally)
 * Run it on test file: ```python split.py ../src/puzzle.tif```
 * If everything was ok it should create bunch of files in out/ dir, similar to those from the demo link above.

### What it can (at the moment)
 * Detect and remove background (also background of the scanner)
 * Find pieces and separate them
 * Ignore crapy pieces (too small at the moment)
 * Detect pieces orientation and unify them (straighten and make vertical)
 * Save pieces with alpha channel
 * Detects some features to be used for future matching: top side of the piece, corners of contour, top/bottommost points, palette, geometry
 * Suggest tags (like “has blue ink”)
 * Can process files in batch mode
 * Output debug info in nifty html with ability to review each detected piece in great detail.

### Development
Check features subdir for examples of feature detectors and interfaces they are using. If you have an idea or implementation of good features — contact me!

### Requirements
 * [NumPy 1.7.1](http://www.numpy.org/)
 * [OpenCV 2.4.9](http://opencv.org/)
 * [Jinja 2](http://jinja.pocoo.org/)

I've used OpenCV/NumPy from [homebrew](http://brew.sh/). I think any 2.* OpenCV version will work as well 

### Source files
I've included cropped version of task #1 of Darpa Shredder Challenge for the reference. For original files please visit [their website](http://archive.darpa.mil/shredderchallenge/Download.html)

### Speed
On my oldie MBP 17" (2.66 Core I7, 8GB, SSD) it's roughly 10 seconds on first task from DARPA (4600x3600 px), 455 pieces.

### Further reading
I've grabbed and implemented first steps [of solution](http://www.marcnewlin.me/2011/12/you-should-probably-start-burning-your_02.html) from wasabi team.
