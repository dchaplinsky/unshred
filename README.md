unshred
=======
Experiments on [Darpa Shredder Challenge](http://archive.darpa.mil/shredderchallenge/) tasks using Python &amp; OpenCV.

### What it can (at the moment)
 * Detect and remove background (also background of the scanner)
 * Find pieces and separate them
 * Ignore crapy pieces (too small at the moment)
 * Detect pieces orientation and unify them (straighten and make vertical)
 * Save pieces with alpha channel
 * Detects some features to be used for future matching: top side of the piece, corners of contour, top/bottommost points
 * Can process files in batch mode
 * Output debug info in nifty html with ability to review each detected piece in great detail.

### Requirements
 * [NumPy 1.7.1](http://www.numpy.org/)
 * [OpenCV 2.4.9](http://opencv.org/)
 * [Jinja 2](http://jinja.pocoo.org/)

I've used OpenCV/NumPy from [homebrew](http://brew.sh/). I think any 2.* OpenCV version will work as well 

### Source files
I've included cropped version of task #1 of Darpa Shredder Challenge for the reference. For original files please visit [their website](http://archive.darpa.mil/shredderchallenge/Download.html)

### Speed
On my oldie MBP 17" (2.66 Core I7, 8GB, SSD) it's oughly 6 minutes on 10 jpegs (300dpi), 200-250 pieces each.

### Further reading
I've grabbed and implemented first steps [of solution](http://www.marcnewlin.me/2011/12/you-should-probably-start-burning-your_02.html) from wasabi team.
