# vid_annotation_tool
Stand-alone version of the annotation tool from the original AccessMath repo.

## Code
This tool is tested and intended for usage with **Python 3.6+**.

Main Library Requirements:
 - Pygame
 - OpenCV
 - Numpy 
 - Shapely

To use the annotation tool, please extract all folders and files from `Extract_this.zip` and put them in the root directory.

## Video Annotation
#### Export Frames(required for annotation tools)
It exports frames from original videos for video annotation. To ensure the annotator work correctly, we suggest using `image based annotation mode` instead of `video based annotation mode`. Please export `FRAME_EXPORT_FPS` frames per second given in the [config] 

       Command: 
       > python pre_ST3D_v2.0_00_export_frames.py [config] [mode] [parameters]  

       Examples:
       For one specific lecture:
       > python pre_ST3D_v2.0_00_export_frames.py conf_test.conf -l video_name

       Similarly, for a set of lectures: 
       > python pre_ST3D_v2.0_00_export_frames.py conf_test.conf -l "video_name_01 video_name_02 ..."

#### Video Annotation Tool
![alt text](https://github.com/adaniefei/Other/blob/images/img_gt_annotator_v2.png?raw=true "gt_annotator")

This annotator is used to label the video objects, video segments and key-frames, and annotation data can be exported for further analysis. 

       Command:
       > python gt_annotator.py [config] [lecture_name]

       Examples:
       For one specific lecture:
       > python gt_annotator.py conf_test.conf video_name

The tool also has keyboard shortcuts available to make your annotation easier.
![alt text](https://github.com/adaniefei/Other/blob/images/img_gt_annotator_shortcuts_v2.png?raw=true "shortcuts")
       
- Video Playing 
       
       [SPACE]: Play/Pause the video stream
       [Z]/[X]: Jump backward/forward for 1 frame
       [A]/[S]: Jump backward/forward for 10 frames
       [Q]/[W]: Jump backward/forward for 100 frames
       [1]/[2]: Jump backward/forward for 1000 frames
       [C]/[V]: Speed slower/faster for 50%
       
- Annotation
       
       [K]: Add a new object key-frame on the current segment "S_cur". This action will split the current segment into two
       subsegments "S_1" and "S_2". The label of either subsegments will be copied from "S_cur" directly unless it is updated. 
       The video stream will keep playing.  
       
       [E]: Add a new object key-frame on the current segment "S_cur" (similar to [K]). In addition, the tool will show the 
       section for inputting segment label (pointed by the green arrow in the figure above) while pausing the video stream. 

- Zoom In/Out

       The [0.5x]/[2.0x] buttons aside of the "Zoom:[Percentage]" status are used to zoom in/out to show the visual detail of the 
       video. The zooming percentage can be "100%", "200%" or "400%". 
       
       When the zoom status is not "100%", a vertical/horizontal scroll bar will appear on the right side of/below the display 
       region. These two scroll bars are used to access the labeling region under the current zooming setting.
       
- Shortcuts not included as buttons on the tool's interface
       
       [<]/[>]: Copy the position of the current video object from previous/next keyframe
       
- "Split" feature for "QUAD" object
![alt text](https://github.com/adaniefei/Other/blob/images/Quad_split.png?raw=true "split-quad")
       
       For the quadrilateral (QUAD) object, a "Split" button (see the figure above) is used to add an extra node on every edge of 
       the object. This feature provides the flexibility to create a polygon instead of a quadrilateral for annotating in more complex
       scenarios. For example, when two ROIs are too close on the screen, using two QUAD objects can not fully annotate each 
       individual region without significantly overlapping the other object.
       
       To use the "Split" feature:
       - (1): Select a QUAD object. 
              The "Split" button for the QUAD object will only appear on the interface when the QUAD object is selected.
       - (2): Click the "Split" button.
              After this action, the QUAD object will be "unselected" in the display region. The object list will be refreshed, and the 
              thumb of its vertical scrollbar will move to the top.
       - (3): Select the QUAD object.
              The additional nodes can be seen and modified as the object is selected.
              
              
------

## Citation
Please cite the following paper in your publication if this tool helps your research :)

    @inproceedings{davila2017whiteboard,
        title={Whiteboard video summarization via spatio-temporal conflict minimization},
        author={Davila, Kenny and Zanibbi, Richard},
        booktitle={2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR)},
        volume={1},
        pages={355--362},
        year={2017},
        organization={IEEE}
     }
