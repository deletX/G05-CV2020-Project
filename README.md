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
that must be downloaded [here](https://drive.google.com/file/d/1kQ29lFhHEGQrKqlUU0BNB1cIyoTHkwI6/view?usp=sharing) and put into `detection/people/yolo-coco/`.

## Output
The output of the project consists in:
    1 - an output video in which we annotate:
        - For each painting bbox, the ROI containing the rectified version of the original painting and the indication of the first retrieved db painting
        - For each person, the ROI with the indication of the relative room
    2 - A terminal output in which we print:
        - The description of what the program is processing(ONLY IF VERBOSE MODE IS ACTIVATED)
        - For each frame, and for each painting detected, the indication of the entire result of the ranked list retrieved(ALWAYS) 
 
## Paper
Detail of the pipline can be found inside the [paper document](./Gambelli_Gavioli_Glorio_g05_project_paper.pdf)

## Example videos
These will open youtube videos
### Example video #1
[![Example Video 1](https://img.youtube.com/vi/76wbWCvVGRY/0.jpg)](https://www.youtube.com/watch?v=76wbWCvVGRY)

### Example video #2
[![Example Video 2](https://img.youtube.com/vi/aPo2egg-6tc/0.jpg)](https://www.youtube.com/watch?v=aPo2egg-6tc)

### Example video #3
[![Example Video 2](https://img.youtube.com/vi/W2iwnYzL37U/0.jpg)](https://www.youtube.com/watch?v=W2iwnYzL37U)

### Example video #4
[![Example Video 2](https://img.youtube.com/vi/VaQ0EZ9viMM/0.jpg)](https://www.youtube.com/watch?v=VaQ0EZ9viMM)

### Example video #5
[![Example Video 2](https://img.youtube.com/vi/BMh_mPzDwG4/0.jpg)](https://www.youtube.com/watch?v=BMh_mPzDwG4)
