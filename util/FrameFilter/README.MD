# FrameFilter

The FrameFilter goes through all scan directories of the 3RScan dataset and creates one file 2DInstances.txt.
This file is a list of all object instances from all frames in all scans that fulfill a minimum amount of filtering options.
These filters are:

- Minimum bounding-box size (in pixels)
- Minimum segmentation-mask size (in pixels)
- Minimum visibility scores (truncation and occlusion)
    - Truncation: 0.0 < t < 1.0 specifies how much of an object is visible in the frame, i.e. not cut off at image edges
    - Occlusion: 0.0 < o < 1.0 specifies how much of an object is visible in front of objects, i.e. not occluded by other objects
- Class Filter: the file filter_labels.txt specifies all classes that are valid.

If an instance passes the filters it is added as a row of the file 2DInstances.txt.
Each row has the following format:
          
    reference	scan_id	frame_id	instance_id	semantic_label	bounding box	visibility	other_instance_ids_and_bboxes
    
The FrameFilter requires each scan directory to have rendered all frames with the Rio_Renderer (https://github.com/WaldJohannaU/3RScan/tree/master/c%2B%2B/rio_renderer).
In detail, the following files are required:
- Bounding-Box file: frame-*.bb.txt
- Visibility file: frame-*.visibility.txt

The FrameFilter can be used as follows:

- mkdir build
- cd build
- cmake ..
- make
- ./FrameFilter <path/to/3RScan>

The filters currently need to be set manually in the code by modifying main.cc at the corresponding locations.