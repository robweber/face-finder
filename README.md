# Face Finder

Combines the Python [deepface library](https://github.com/serengil/deepface) with embedded Python to create a portable face recognition system to scan folders full of images. The goal is to set this up so that it can be easily transported, via USB or other means, to different computers and run the face recognition algorithms.

## Install

The target system here is Windows. If you want to run on other systems where Python is already installed simply clone the repo and install the dependencies. For Windows download [embedded Python](https://www.python.org/downloads/windows/) (tested on 3.10) and clone a copy of the repository. Extract the full embedded Python source code into the `face-finder` folder in a sub folder named `python`. You should have a folder structure that looks like the following:

```
python/
finder.py
get_pip.py
README.md
requirements.txt
```

To install the required libraries PIP also needs to be available. To install PIP do the following:

1. Copy the `get_pip.py` file inside the `python/` folder.
2. From the command line run `python.exe get_pip.py` from within the folder. This will install Pip into the `Scripts/` directory.
3. Open and edit the file `python310._pth`
4. Uncomment the `import` line.
5. Download the required Python libraries. `Scripts\pip.exe install -r ..\requirements.txt`

When this is complete you should have embedded Python ready to go, along with the required Python libraries. You can now move the `face-finder` directory wherever you want and run the script wherever you want.

## Usage

To use the program you need to have 2 things. The first is a folder where the known face images will go. These should be of a single person you want to recognize in your input images. The more you have the better matching the models can do. Each image should contain only the face of the person you're trying to match. The second thing you need is a single image or a folder of images you want to search. The program will scan folders recursively you so can do a full directory full of sub folders.

I usually put the known images folder right inside the script directory so everything will stay portable. You can run the program off the command line with:

```

python\python.exe finder.py -k known_images\ -i photos_dir\

```

### Advanced

The deepface library can use different facial recognition models and face detection backends. The default ones for this script are __Facenet512__ and __mtcnn__. This was based on my own experience playing around for accuracy. You can use the `-m` and `-d` arguments to specify different ones. Reading the [deepface docs](https://github.com/serengil/deepface) you can see which are better for speed vs accuracy.

## Acknowledgements

All the heavy lifting is accomplished by the [deepface](https://github.com/serengil/deepface) by [@serengil](https://github.com/serengil)
