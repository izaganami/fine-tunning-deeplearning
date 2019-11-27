import json
import os
from os import listdir
from os.path import isfile, join
import subprocess
from moviepy.video.io.VideoFileClip import VideoFileClip
from collections import *
import os



##Entry will be like : A={"F01":(p1,q1),"F02":(p2,q2),...}
##path: path for the directory
def extract(path):
    A=OrderedDict(("F01",[p1,q1]),("F02",[p2,q2]),...)
    files = [f for f in listdir(path) if isfile(join(path, f)) and f.split(".")[1] == "json" ]
    for file in files:
        with open(file) as f:
            data=json.load(f)
            name=data['dataFilename']+'.mp4'
            for clicks in data['clicks']:
                click=clicks['click']
                frame=clicks['time']
                pframe=A[click][0]
                qframe=A[click][1]
                cmd = "ffmpeg -ss {} -i {} -t {} -c copy output/extract_{}.mp4".format(frame-pframe,name,pframe+qframe,str(A.keys().index(click))+"_"+click)
                process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
                os.system(cmd)





