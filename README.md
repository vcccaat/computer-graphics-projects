This repo demonstrates my assignments and projects in computer graphics.

# Final Assignment

Built a fully functional Kirby game with camera control and different shading effects using three.js and applied basic animation principles on Kirby, Kirby can fly around, walk around and inhale bombs in the scene. Implemented particle system to mimic a waterfall, and used sprite model to design a river on a mountain. 

## Demo video

<a href="https://graphics-kirby-demo.vercel.app">
<img src="inputs/kirbyGame.png" alt=" " style="width:400px;"/></a>

Try out here: https://cs5620-computer-graphic.vercel.app/
Source Code: https://github.com/vcccaat/kirby-game

# Assignment 4: Ray Tracing

Implemented ray tracing in Python with diffuse shading, specular shading, shadow test, and mirror reflection. The result image: 

![step7](inputs/step7.png)

Source Code: [ray.py](./ray.py)

## Creative part

Our group used the sphere class for building our 3d objects in the rendered image. A Kirby character was built by overlapping spheres. There are also many spheres of different color and size floating in the air. Each of the spheres applied the Blinn-Phong shading. We used five light sources each with different color to give our image more complex and colorful lighting.

DEMO:

​                         <img src="inputs/Creative.png" alt="Creative" style="width:400px;" />      

<img src="inputs/demo1.png" alt="demo1 " style="width:400px;" />

<img src="inputs/demo2.png" alt="demo2 " style="width:400px;" />

# Assignment 2

2D transformation 

- Setting PRSA with a matrix and a position vector

- Basic 2D shape manipulation controls 

- Implement basic Subdivision

<img src="images/cs4620iscool.jpg" alt=" " style="width:400px;"/>

## Demo video:

<a href="https://graphics-emoji-demo.vercel.app/">
<img src="inputs/emoji.png" alt=" " style="width:400px;"/></a>

# Assignment 3: Imaging

Implemented:

* [Photography.ipynb](./Photography.ipynb) Pointwise transformations apply to every pixel separately.  We will use them to compute good-looking 8-bit color images from raw camera sensor data. The curve adjustments we learned about in class fall under this category.

DEMO of white balancing:

<img src="inputs/whiteblance.png" alt="whiteblance" style="width:400px;" />

<img src="inputs/whitebalanceAfter.png" alt="whitebalanceAfter " style="width:400px;" />

* [Filtering.ipynb](./Filtering.ipynb) Convolution filters involve computing weighted sums over a local area of the image.  We will use them to simulate out-of-focus blur and to sharpen images.

DEMO of gaussian filter and sharpen filter:

<img src="inputs/convolution-reference.png" alt="convolution-reference " style="width:200px;" />

<img src="inputs/convolution-reference-medium.png" alt="convolution-reference-medium " style="width:200px;"  />

<img src="inputs/sharpened-reference-medium.png" alt="sharpened-reference-medium" style="width:200px;" />

* [Distortion.ipynb](Distortion.ipynb) Image warping involves moving content around in an image.  We will apply a simple image warp that corrects for the distortion in a wide angle lens.
  
  DEMO:
  
  <img src="inputs/room.jpg" alt="room" style="width:400px;" />

<img src="inputs/distortion-img-lin-reference.png" alt="distortion-img-lin-reference " style="width:400px;" />

- [CreativePart.ipynb](./CreativePart.ipynb) in the creative part, I implemented edge detection using sobel kernel and set obvious edges with a RGB value. I built this filter to mimic the effect of a color pencil sketch
  
  <img src="inputs/1.png" alt="1 " style="width:400px;" />

<img src="inputs/cornell.png" alt="cornell" style="width:400px;" />

# Assignment 1: Point-Line Duality

In this assignment you will complete the basic mathematical functions underlying an application for visualizing Point Space and Line Space. When you click inside the Point Space window, it will create a draggable green point in Point Space. At the same time, in Line Space, we will see the coordinates of any line that contains our specified point turn green as well. Similarly, if we click in the Line Space window, it will lay down a draggable blue dot in line space (remember, a location in line space is actually a line in Point Space, where we live...). At the same time, we will see the set of points in point space that rest on our selected line turn blue as well. In short: click on point space to put down a point, which is a line in line space; click on line space to put down a line in point space, which is a point in line space...

### Completed Demo:

<img src="images/WithAxesLabels.png" alt=" " style="width:400px;" />

<img src="images/FinishedCodeImage.png" style="width:400px;" />
