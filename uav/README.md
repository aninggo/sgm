# TULIPP UAV Use Case - Reference Code#

&copy; Fraunhofer Institute of Optronics, System Technologies and Image Exploitation, 2016

<b>Contact:</b>
 - Ruf, Boitumelo (<boitumelo.ruf@iosb.fraunhofer.de>)
 - Schuchert, Tobias (<tobias.schuchert@iosb.fraunhofer.de>)

### Brief Use Case Description ###

Small UAVs have entered a large range of applications as their underlying technology has improved and more avenues for use have been explored.
Now, applications such as surveillance, search and rescue, video production, logistics and research are just a small subset of their uses.
Their use in the entertainment domain is rapidly growing as the result versus cost ratio becomes more competitive.
However, with the growing number of UAVs in use the number of crashes and problems with their control are also increasing.
These problems can be caused by operator error or malfunction.
In the worst-case scenario these errors can cause damage to more than just the UAV involved and end up harming people, goods or infrastructure.
Therefore, UAVs need more intelligent control and interaction systems, such as automatic collision avoidance or more robust pose estimation, to minimize risks of failure.
The problem is that more intelligence needs more computing power, which is very limited especially on small UAVs.

The Tulipp solution aims to fill this processing gap by using its good performance-to-weight and power-consumption-to-weight figures.
Similar to [3] we aim to use computer vision algorithms such as stereo and depth estimation to detect obstacles and evaluate the surroundings in order to make the UAV more intelligent.
For this purpose, we attach the Tulipp-board with a stereo camera setup orientated in direction of flight to a UAV.
Our goal is to use stereo algorithms to detect obstacles automatically that are within a dangerous vicinity in front of the UAV and to avoid a collision.
Amongst others [1], [4] and [5] describe popular stereo algorithms that are able to be run in real-time.

### Code ###

As reference code for this use case we implemented a disparity computation algorithm [1] to compute the pixel displacements between two stereo images.
For each pixel in the reference image a matching cost [2] is computed against a corresponding pixel in the matching image at a certain displacement along the scanline.
This is followed by a cost aggregation along eight paths centered in each reference pixel.
For the final disparity image the displacement is chosen which corresponds to the lowest costs.

For more information on the internals of the algorithm please consult the references [1] and [2].

Run Doxygen for complete documentation.

#### System Requirements ####

The reference code is tested and compiled on a <b>Ubuntu 15.04.</b> and requires:

- C++ 11
- CMake (2.8.11 or higher)
- OpenCV (3.0 or higher)
- OpenMP

#### Build ####

Create a seperate build directory and generate a `Makefile`.
Running `make` on the generated `Makefile` builds the applicaton and adds it to the `bin` directory.

```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
$ cd ../bin
```

#### Synopsis ####

The sample application requires two rectified stereo images. Image 2 is to be located on the left hand side of image 1:

<code>
./app <i>frame1 frame2 imgScale</i>
</code>

&nbsp;&nbsp;&nbsp;&nbsp;<b>frame1</b>:&nbsp;&nbsp;Input frame 1&nbsp;&nbsp;<i>Required</i><br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>frame2</b>:&nbsp;&nbsp;Input frame 2 of same size as frame1 &nbsp;&nbsp;<i>Required</i><br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>imgScale</b>:&nbsp;&nbsp;Factor to scale input iamges. <i>Optional</i><br>

#### Configuration ####

The application can be configured setting the corresponding values in config.hpp. NOTE, that the application needs to be recompiled in order for the configurations to take effect.

### Reference ###

[1] <br> Hirschmüller, Heiko. "Accurate and efficient stereo processing by semi-global matching and mutual information". IEEE Computer Society Conference on Computer Vision and Pattern Recognition, Vol. 2., 2005.

[2] <br> Birchfield, S. and Tomasi, C. "Depth discontinuities by pixel-to-pixel stereo". International Journal of Computer Vision, Springer, 1999

[3] <br> Barry, Andrew J., and Russ Tedrake. "Pushbroom stereo for high-speed navigation in cluttered environments." IEEE International Conference on Robotics and Automation (ICRA), 2015.

[4] <br> Hrabar, Stefan, et al. "Combined optic-flow and stereo-based navigation of urban canyons for a UAV". 2005 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, 2005.

[5] <br> Yang, Ruigang, and Marc Pollefeys. "Multi-resolution real-time stereo on commodity graphics hardware". IEEE Computer Society Conference on Computer Vision and Pattern Recognition, Vol. 1. IEEE, 2003.
