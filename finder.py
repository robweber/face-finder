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

# parse the arguments
parser = argparse.ArgumentParser(description='Face Finder')
parser.add_argument('-k', '--known', default='known/', type=check_dir,
                    help='Directory of known faces to compare against')

args = parser.parse_args()

print(f"Loading known faces from '{args.known}'")

known_image_encodings = []
for file in os.listdir(args.known):
    k_image = face_recognition.load_image_file(os.path.join(args.known, file))
    known_image_encodings.append(face_recognition.face_encodings(k_image)[0])
print(f"Found {len(known_image_encodings)} faces in known images")

test_image = face_recognition.load_image_file("pics/training/2022-11-21 - Magic Kingdom Park - Main Street USA_7.jpeg")
test_encoding = face_recognition.face_encodings(test_image)

for e in test_encoding:
    results = face_recognition.compare_faces(known_image_encodings, e)
    print(results)
