import argparse
import logging
import os
import os.path
import shutil
import sys
import time
from deepface import DeepFace
from joblib import Parallel, delayed

COPY_DIR = "found"

def check_dir(aDir):
    if(os.path.isdir(aDir)):
        return aDir
    else:
        raise argparse.ArgumentTypeError(f"Directory '{aDir}' does not exist")

def check_exists(aFile):
    if(os.path.exists(aFile)):
        return aFile
    else:
        raise argparse.ArgumentTypeError(f"File path '{aFile}' does not exist")

def is_image(aFile):
    _, ext = os.path.splitext(aFile)
    return ext.lower() in [".jpg", ".jpeg"] and not aFile.startswith('.')

class ImageCompare:
    model = None
    detector = None

    def __init__(self, model, detector):
        self.model = model
        self.detector = detector

    def compare_image(self, image_file, known_faces):
        result = {"image": image_file, "result": False}

        if(os.path.exists(image_file)):
            test_encoding = DeepFace.find(img_path = image_file, db_path = known_faces, model_name=self.model, detector_backend=self.detector, silent=True, enforce_detection=False)

            for df in test_encoding:
                distance = df.loc[:,f"{self.model}_cosine"].min()
                if(distance < .20):
                    result['result'] = True
                    break
        else:
            logging.debug(f"skipping {image_file}")

        return result

# parse the arguments
parser = argparse.ArgumentParser(description='Face Finder')
parser.add_argument('-k', '--known', default='known/', type=check_dir,
                    help="Directory of known faces to compare against, default is '%(default)s'")
parser.add_argument("-i", "--input", type=check_exists, required=True,
                    help="A single image, or a directory of images, to compare against")
parser.add_argument("-n", "--name", type=str, default="Known face",
                    help="The name of who you're looking for, for output")
parser.add_argument("-p", "--parallel", type=int, default=3,
                    help="The number of image processing jobs to run at the same time, default is %(default)i")
parser.add_argument("-C", "--copy", action="store_true", help="copies found images to a directory")
parser.add_argument('-D', '--debug', action='store_true', help='If the program should run in debug mode')

deepface_args = parser.add_argument_group("deepface options", "arguments for the deepface library")
deepface_args.add_argument("-m", "--model", default="Facenet512", help="The face recognition model, default is %(default)s",
                          choices=["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace",])
deepface_args.add_argument("-d", "--detector", default="mtcnn", help="The face detection backend, default is %(default)s",
                          choices=['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe'])
args = parser.parse_args()

logLevel = 'INFO' if not args.debug else 'DEBUG'
logging.basicConfig(datefmt='%m/%d %H:%M',
                    format="%(levelname)s: %(message)s",
                    level=getattr(logging, logLevel))
logging.debug('Debug Mode On')

# load the known face images
start_time = time.perf_counter()
logging.info(f"Loading faces from '{args.known}'")

comparator = ImageCompare(args.model, args.detector)
# test against unknown
if(os.path.isfile(args.input)):
    if(comparator.compare_image(args.input, args.known)):
        logging.info(f"{args.name} found in '{os.path.join(args.input)}'")
    else:
        logging.info("No known faces found")
else:
    search_images = []
    for root, dirs, files in os.walk(args.input):
        for f in files:
            if(is_image(f)):
                search_images.append(os.path.join(root, f))
    logging.info(f"Found {len(search_images)} images to process")

    processed_images = Parallel(n_jobs=args.parallel)(delayed(comparator.compare_image)(f, args.known) for f in search_images)

    found_images = []
    for i in processed_images:
        if(i['result']):
            found_images.append(i['image'])

    logging.info(f"Found {len(found_images)} images")
    for i in found_images:
        logging.info(f"{args.name} found in '{i}'")
finish_time = time.perf_counter()
print(f"Program ran in {finish_time-start_time} seconds")

if(args.copy):
    print(f"Copying found images to {COPY_DIR}")

    # create directory if not there
    if(not os.path.isdir(COPY_DIR)):
        os.mkdir(COPY_DIR)

    for f in found_images:
        shutil.copy(f, os.path.join(COPY_DIR, os.path.basename(f)))
