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
![alt text](https://github.com/adaniefei/Other/blob/images/gt_annotator.png?raw=true "Logo Title Text 1")

This annotator is used to label the video objects, video segments and key-frames, and annotation data can be exported for further analysis. 

       Command:
       > python gt_annotator.py [config] [lecture_name]

       Examples:
       For one specific lecture:
       > python gt_annotator.py conf_test.conf video_name

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
