# face verification with the VGGFace2 model

import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input


def main(reference, candidate):
    # define filenames
    filenames = [reference, candidate]
    # get embeddings file filenames
    embeddings = get_embeddings(filenames)
    # define sharon stone
    sharon_id = embeddings[0]
    # verify known photos of sharon
    print("Positive Tests")
    matched = is_match(embeddings[0], embeddings[1])
    if matched:
        result_text = "Matched"
    else:
        result_text = "Unmatched"

    # figure size in inches optional
    rcParams["figure.figsize"] = 11, 8

    # read images
    img_a = mpimg.imread(reference)
    img_b = mpimg.imread(candidate)

    # display images
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img_a)
    ax[1].imshow(img_b)
    plt.title(result_text, loc="left")
    plt.show()


# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = plt.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]["box"]
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array


# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames):
    # extract faces
    faces = [extract_face(f) for f in filenames]
    # convert into an array of samples
    samples = np.asarray(faces, "float32")
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    # create a vggface model
    model = VGGFace(model="resnet50", include_top=False, input_shape=(224, 224, 3), pooling="avg")
    # perform prediction
    y_pred = model.predict(samples)
    return y_pred


# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        print(">face is a Match (%.3f <= %.3f)" % (score, thresh))
        return True
    else:
        print(">face is NOT a Match (%.3f > %.3f)" % (score, thresh))
        return False


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Verification.")
    parser.add_argument("-i",
                        "--input",
                        type=str,
                        help="Reference image")
    parser.add_argument("-c",
                        "--candidate",
                        type=str,
                        help="Candidate image")
    args = parser.parse_args()
    reference_img = args.input
    candidate_img = args.candidate
    main(reference_img, candidate_img)
