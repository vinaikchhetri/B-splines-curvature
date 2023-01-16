# B-spline-curvature-
This work visualises the curvature surface plot of bsplines across knots via Deep Learning vs centripetal heuristic. The videos below correspond to clamped B-splines with 4 points. The training file 2D-unsupervised-optimisation.ipynb deals with unsupervised traninng for B-splines interpolation for only 4 points. Training noteooks and neural nets for 5,6,7,8 points will be uploaded soon. 

#To only see the B-spline curve change do (1st video),
"""
python bsp_model.py
"""

#To see the B-spline curve as well as curvature surface plot change do (2nd video),
"""
python ml_viz.py
"""
#Note the surface plot(video2) file takes a little bit of time because 3d plotting is a bit expensive in this case.
For the ease of use, while running ml_viz.py, click on the curves window first, then click and hold only one of the 2 interior crosses/points. Move the point and let go. Now you will need to wait a little to see the changes as shown in the 2nd video.

#bsp_model.py is very fast as shown in video 1.

https://user-images.githubusercontent.com/28961441/212672261-b6e81a4a-4b91-4ddd-a082-1dae9e9f65df.mov



https://user-images.githubusercontent.com/28961441/212676631-cb3d84ac-7e69-41cb-8369-ea4709e86f01.mov

