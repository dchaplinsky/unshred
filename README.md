unshred
=======
Experiments on [Darpa Shredder Challenge](http://archive.darpa.mil/shredderchallenge/) tasks using Python &amp; OpenCV.

### What it can (at the moment)
 * Detect and remove background
 * Find pieces and separate them
 * Ignore crapy pieces (too small at the moment)
 * Detect pieces orientation and unify them (straighten and make vertical)
 * Save pieces with alpha channel
 * Output some useful debug info

### Requirements
 * [Numpy 1.7.1](http://www.numpy.org/)
 * [OpenCV 2.4.9](http://opencv.org/)

I've used versions from [homebrew](http://brew.sh/). I think any 2.* OpenCV version will work as well 

### Source files
I've included cropped version of task #1 of Darpa Shredder Challenge for the reference. For original files please visit [their website](http://archive.darpa.mil/shredderchallenge/Download.html)

### Further reading
I've grabbed and implemented first steps [of solution](http://www.marcnewlin.me/2011/12/you-should-probably-start-burning-your_02.html) from wasabi team.
