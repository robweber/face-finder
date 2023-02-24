import argparse
import os
import os.path
import sys
from deepface import DeepFace

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

def compare_image(image_file, known_faces):
    result = False
    #print(image_file)
    if(os.path.exists(image_file)):
        test_encoding = DeepFace.find(img_path = image_file, db_path = known_faces, model_name="Facenet512", detector_backend="mtcnn", silent=True, enforce_detection=False)

        for df in test_encoding:
            distance = df.loc[:,"Facenet512_cosine"].min()
            #print(distance)
            if(distance < .20):
                result = True
                break
    else:
        print(f"skipping {image_file}")

    return result

# parse the arguments
parser = argparse.ArgumentParser(description='Face Finder')
parser.add_argument('-k', '--known', default='known/', type=check_dir,
                    help='Directory of known faces to compare against')
parser.add_argument("-i", "--input", type=check_exists, required=True,
                    help="A single image, or a directory of images, to compare against")
parser.add_argument("-n", "--name", type=str, default="Known face",
                    help="The name of who you're looking for, for output")
args = parser.parse_args()

# load the known face images
print(f"Loading faces from '{args.known}'")

#print(f"Found {len(known_face_encodings)} faces in known images")

# test against unknown
if(os.path.isfile(args.input)):
    if(compare_image(args.input, args.known)):
        print(f"{args.name} found in '{os.path.join(args.input)}'")
    else:
        print("No known faces found")
else:
    found_images = []
    for root, dirs, files in os.walk(args.input):
        for f in files:
            if(is_image(f)):
                if(compare_image(os.path.join(root, f), args.known)):
                    found_images.append(f"{args.name} found in '{os.path.join(root, f)}'")

    print(f"Found {len(found_images)} images")
    for i in found_images:
        print(i)
