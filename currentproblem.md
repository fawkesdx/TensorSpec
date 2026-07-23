updates for the viewer: pease do the following

1. I dont understand why there is the option of XY scan and Fermimap beside the loader button. the user may not know what kind of data they are loading. the load button should just stand alone.

2. when loaded, I want the data to be directly stored in the workspace. I know this has been done. but also, I want the user to be able to choose couple of files together.

3. that means, loading a file doesnt mean we need to directly open it

4. I want to be able to "fetch" the data from the workspace where there is another column available in the data_viewer tab only listing the data feasible to be drawn with data viewer like arpes data or simulated data. and since this is the tab inside the arpes tab so only arpes related data is allowed.

I understand that the mechanism can be called in XMCD data viewer tab later. but I want the general feature to be callable later in other tab but for this one is strictly arpes. so does the data in other tab later.

5. I want the tensorspec to have a generic data viewer available in the main browser. because we can know what kind of data available there anyway. if it is the crystal structure so it can call the crystal viewer to view it in their panel. if it is arpes then view it in arpes data viewer panel. if it is xmcd and so on and so forth. I want this to be also avialable in many panels. which means I want to be able to draw two different crystal structure stored in workspace from this general viewer button call. basically what I am thinking is that this is like the copy of the viewer and the features available in its suite but just a stand alone window for user to play around. and user can compare two things at the same time

6. with multi windows to be open like this, I want the main browser to keep track on the opened windows too so that the user can choose the window directly from the panel available in main browser to bring it to the front

7. with this multi viewer windows opened, I want to have ability to snap dimension too where the crosshair on two or more different images that come from different files can be explored together by snapping their dimension to follow if they have same dimension axis so that user can follow higher dimension data viewed at the same time

8 I want to have the deltaX deltaY integration option to have slider beside the manual input

9. I want the remainder dimension slider at the bottom of it to have manual input too!

10. the snap function is great, but I also want to have some kind of positioning too. if I expand the snap function on the vertical axis, then it make sense to draw it on the side. but if it is click on the top then the extra viewer is on the top. bottom at bottom, left to left, and so on. this extra window can have ability to detach and reattach too. regarding positioning, it is better that the code recognize in which quadrant I am clicking so that if I am in Q1, then I want to snap the Y axis so it will show on the right side (vertical y axis), if I snap the X axis then it appears on the top panel. this snapping function act like a puzzle too so that user can expand their snapping to the newly built figures so if their screen is big enough, they can snap whatever pictures they want to do and they are all connected in their crosshair. a right click on a panels can have option to de-snap it from the configuration. Give option to desnap "up, bot, left, right" connection so it is very modular.

11. the 1D toggle is good which is to draw "EDC/MDC" for arpes for example. But I also want to have another 1D toggle button which will plot the data along the "other" dimension where user have the dropdown option next to it which will populate the remainder dimension label. SO the use case here is that if it is a PEEM spectra image stacks, the plot of XY will show how the spatial intensity looks like. and the other dimension is the energy. so if user click the XY point and toggle on this button, a new 1D plotter will appear on the side and show how the intensity vs energy looks like (because energy is the only remainder dimension left)

12.I like the sliders shown for the remainder dimension. I also want to have manual input on the number there not just a static number showing where the slider is.