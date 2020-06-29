# [G05] Computer Vision Project 2020 
To run use the command `python proj.py`

Options:
 - _[Required]_ `-i` or `--input` followed by the `<filename>` of the input file
 - `-o` or `-output` followed by the `<filename>` of the desired otput file without extension. 
 If it is not defined the result will be shown in a window.
 - `-v` or `--verbose`, if defined will run the project in verbose mode
 - `-s` or `--skip` followed by a number defines the number to frame to skip, for example if value is 3 it will analyze 1 frame each 3.
 - `-d` or `--debug` will enable debug-mode

## Setup
Require python modules are found in `requirements.txt` and therefore can be installed with `pip install -r requirements.txt` 

The only file that we were not able to re-create or retrieve during runtime are YOLO network weights,
that must be downloaded [here](http://tette.org/) and put into `detection/people/yolo-coco/`.

## Paper
Detail of the pipline can be found inside the [paper document](./Gambelli_Gavioli_Glorio_g05_project_paper.pdf)