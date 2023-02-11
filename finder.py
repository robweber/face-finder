import argparse
import face_recognition
import os
import os.path
import sys

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

def compare_image(image_file, known_face_encodings):
    result = False
    test_image = face_recognition.load_image_file(image_file)
    test_encoding = face_recognition.face_encodings(test_image)

    for e in test_encoding:
        results = face_recognition.compare_faces(known_face_encodings, e)
        if(results[0] == True):
            result = True
            break

    return result

# parse the arguments
parser = argparse.ArgumentParser(description='Face Finder')
parser.add_argument('-k', '--known', default='known/', type=check_dir,
                    help='Directory of known faces to compare against')
parser.add_argument("-i", "--input", type=check_exists, required=True,
                        help="A single image, or a directory of images, to compare against")
args = parser.parse_args()

# load the known face images
print(f"Loading known faces from '{args.known}'")

known_face_encodings = []
for file in os.listdir(args.known):
    k_image = face_recognition.load_image_file(os.path.join(args.known, file))
    known_face_encodings.append(face_recognition.face_encodings(k_image)[0])

print(f"Found {len(known_face_encodings)} faces in known images")

# test against unknown
if(os.path.isfile(args.input)):
    if(compare_image(args.input, known_face_encodings)):
        print(f"Known face found in '{os.path.join(root, f)}'")
    else:
        print("No known faces found")
else:
    for root, dirs, files in os.walk(args.input):
        for f in files:
            if(is_image(f)):
                if(compare_image(os.path.join(root, f), known_face_encodings)):
                    print(f"Known face found in '{os.path.join(root, f)}'")
