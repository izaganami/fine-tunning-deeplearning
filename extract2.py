"""
DEPENDENCIES : ffmpeg
"""
from collections import *

import json
import os
from os import listdir
from os.path import isfile, join
import subprocess
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.VideoFileClip import VideoFileClip
import matplotlib.pyplot as plt
import time
##import data_augmentation
import numpy as np


def get_duration(file):
    """
        prend en entrée le nom du fichier (ou son chemin ?) et renvoie la durée
    """
    print(file)
    #Get the duration of a video using ffprobe.
    cmd = 'ffprobe -show_entries format=duration -v quiet -of csv="p=0" -i {}'.format(file)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, shell=True)
    out = p.stdout.decode('utf-8')
    #out = p.stdout.read()
    print(out)
    #return float(out)


def get_duration_linux(file):
        clip = VideoFileClip(file)
        return(clip.duration)


def decoupe_multiple_vid():
    """
        On considere que les video sont dans le dossier videos et les json dans le dossier json
        les videos suivent la convention suivante : pXXXvariable
        les json suivent la convention suivante : CC-pXXXvariable
    """
    workplace = os.path.dirname(os.path.abspath(__file__))
    workplace = workplace + "/json"
    #on recupere tous les JSON
    onlyfiles = [f for f in listdir(workplace) if isfile(join(workplace, f))]
    #pour chaque JSON, on appelle decoupe
    for file in onlyfiles:
        decoupe("json/" + file)

def decoupe(json_filename):
    """
        Prend en entrée le nom (ou le chemin ?) du fichier json et découpe en sous extraits vidéo la vidéo associée
    """
    A=OrderedDict(("F01",['p1','q1']),("F02",['p2','q2']),...)

    __file__ = "data_creationV2.py"
    workplace = os.path.dirname(os.path.abspath(__file__))
    print
    with open(json_filename) as json_file:

        video_filename = "init"
        workplace = workplace + "/videos"
        #on recupere le nom de toutes les videos du dossier video
        onlyvideos = [f for f in listdir(workplace) if isfile(join(workplace, f))]
        for video in onlyvideos:
            #on cherche la video dont le nom contient le même identifiant pXXX
            if json_filename.split("/")[1][3:7] in video:
                video_filename = video
                break

        data = json.load(json_file)

        print("video nom : " + video_filename)

        #On recupere la duree de la video
        duration = get_duration_linux("/home/formation/Documents/data_creatin_hugo/videos/" + video_filename)
        i = 0

        #on cree un dossier pour ranger les extraits, en fait non, un dossier extract pour tout
        #os.makedirs("extract_" + video_filename[0:-4])

        #pour chaque clic, on va decouper la video en sous extrait grâce a la fonction Ellie
        for p in data['clicks']:
            time = p["time"]
            eventType = p["click"]
            pframe=A[eventType][0]
            qframe=A[eventType][1]
            t_begin = max(int(time) - pframe, 0)
            duration = min(pframe+qframe,abs(int(time)-duration))
            Elie(video_filename,t_begin,duration, str(i)+'_'+eventType)
            i += 1

def Elie(video_filename,time_begin,duration, event_id):
    """
        Découpe une vidéo suivant le time begin et duration donne
    """

    #solution 1 : moyennement rapide --- 60.02729105949402 seconds ---
    ffmpeg_extract_subclip("videos/" + video_filename, time_begin, time_begin+duration, targetname="extract/"+video_filename[0:-4]+"_{}.mp4".format(event_id))
    """

    #solution 2 : lent
    p = subprocess.Popen(['ffmpeg', '-ss', str(time_begin), '-i', video_filename, '-t', str(duration), 'extract_'+str(event_id)+'.mp4'], stdout=subprocess.PIPE, shell = True)


    #solution 3 : 65.6761155128479 seconds ---
    bashCommand = "ffmpeg -ss {} -i {} -t {} -c copy outputextract_{}.mp4".format(time_begin, video_filename, duration, event_id)
    os.system(bashCommand)
    """

def get_frames(video_filename):
    """
        enregistre dans le meme dossier les frames de la video donnee en entree
        Le nombre de frames en sortie peut etre parametree : https://ffmpeg.org/ffmpeg-filters.html#Examples-136
    """
    #cmd = 'ffmpeg -i {} -vf "select=eq(pict_type\,I)" -vsync vfr {}_frame-%02d.png'.format(video_filename,video_filename)
    print(video_filename)
    os.makedirs("frames/frames_" + video_filename[0:-4])
    cmd = 'ffmpeg -i {} -vf select="not(mod(n\,8))" -vsync vfr frames/frames_{}/{}_frame-%02d.png'.format("extract/"+video_filename,video_filename[0:-4],video_filename[0:-4])
    os.system(cmd)

def get_mosaique(video_filename):
    """
        cree automatiquement une mosaique a partir d'une video (a ne pas utiliser donc)
    """
    cmd = 'ffmpeg -skip_frame nokey -i {} -vf scale=128*2:72*2,tile=3x2 -an -vsync 0 {}_keyframes%03d.png'.format(video_filename,video_filename)
    os.system(cmd)

def concatenate(dossier):
    """
        Prend en entree le dossier où sont contenues les images a concatener et renvoie une image concatenenee sans l'enregistrer
    """
    #on recupere le chemin aboslu
    workplace = os.path.dirname(os.path.abspath(__file__))
    if dossier != "":
        workplace = workplace + "/" + dossier
    #on recupere les images du dossier
    onlyimgs = [f for f in listdir(workplace) if isfile(join(workplace, f))]
    imgs = [plt.imread(dossier + "/" + i) for i in onlyimgs]
    return np.concatenate(imgs)

def main():
    decoupe_multiple_vid()


if __name__ == "__main__":
    start_time = time.time()
    #main()
    #get_frames("extract_1_U00.mp4")
    #get_mosaique("extract_1_U00.mp4")
    get_duration("videos/p346_Sebastian_ind_adu.mp4")
    print("--- %s seconds ---" % (time.time() - start_time))

    decoupe_multiple_vid()

    workplace = os.path.dirname(os.path.abspath(__file__))
    workplace = workplace + "/extract"
    onlyfiles = [f for f in listdir(workplace) if isfile(join(workplace, f))]
    for f in onlyfiles:
        get_frames(f)

    workplace = os.path.dirname(os.path.abspath(__file__))
    workplace = workplace + "/frames"
    onlydossiers = [d for d in listdir(workplace) if not isfile(join(workplace, d))]
    for d in onlydossiers:
        os.makedirs("frames_augmented/" + d)
        onlyfiles = [plt.imread("frames/"+ d + "/" + f[0:-4] + ".png") for f in listdir(workplace + "/"+d) if isfile(join(workplace + "/" + d, f))]
        ##data_augmentation.data_augmentation(np.array(onlyfiles),d)
        print("CHECKPOINT")
        plt.imsave("mosaiques/mosaique_{}.png".format(d),concatenate("frames_augmented/"+d))
    #Etapes donc :
    #   - appeler mutliple-vid()
    #   - get_frames pour chaque sous video extrait
    #   - data augmentation d'un groupe de frame (c'est dans un autre fichier)
    #   - concatenate pour obtenir la mosaique d'un groupe de frames


