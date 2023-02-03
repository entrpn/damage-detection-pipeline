import base64
from threading import Thread
import os
from PIL import Image
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict

def detect_landmarks(path, return_val):
    """Detects landmarks in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.landmark_detection(image=image)
    landmarks = response.landmark_annotations
    print('Landmarks:',landmarks)
    retval = {}
    for landmark in landmarks:
        retval[landmark.description] = landmark.score
        print(landmark.description)
        for location in landmark.locations:
            lat_lng = location.lat_lng
            print('Latitude {}'.format(lat_lng.latitude))
            print('Longitude {}'.format(lat_lng.longitude))
    
    return_val[5] = retval

    

def detect_text(path, return_val):
    """Detects text in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')
    texts = [text.description for text in texts]
    if len(texts) > 0:
        return_val[2] = "\n".join(texts)
    else:
        return_val[2] = "None found"

def detect_labels(
    path: str,
    return_val : list    
):
    """Detects labels in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.label_detection(image=image)
    print(response)
    labels = response.label_annotations
    retval = {}
    for label in labels:
        retval[label.description] = label.score
    
    return_val[1] = retval

def predict_image_classification_sample(
    filename: str,
    return_val: list
):
    project=PROJECT_ID
    endpoint=ENDPOINT_ID
    location=REGION
    api_endpoint=f"{location}-aiplatform.googleapis.com"
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

    image = Image.open(filename)
    w, h = image.size
    while w * h > 1000000:
        tmp_path = "/tmp/img.png"
        max_size = max(w*0.9,h*0.9)
        image.thumbnail((max_size,max_size))
        image.save(tmp_path)
        w, h = image.size
        print("w:",w,"h:",h,"w*h:",(w*h))
        filename = tmp_path
    
    with open(filename, "rb") as f:
        file_content = f.read()

    print("file_content_length: ", len(file_content))

    # The format of each instance should conform to the deployed model's prediction input schema.
    encoded_content = base64.b64encode(file_content).decode("utf-8")
    print("encoded_content length: ",len(encoded_content))
    instance = predict.instance.ImageClassificationPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]
    # See gs://google-cloud-aiplatform/schema/predict/params/image_classification_1.0.0.yaml for the format of the parameters.
    parameters = predict.params.ImageClassificationPredictionParams(
        confidence_threshold=0.0, max_predictions=5,
    ).to_value()
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    # See gs://google-cloud-aiplatform/schema/predict/prediction/image_classification_1.0.0.yaml for the format of the predictions.
    predictions = response.predictions
    retval = {}
    for label, score in zip(predictions[0]['displayNames'],predictions[0]['confidences']):
        retval[label] = score
    print(retval)
    return_val[0] = retval

def detect_web(
    path : str,
    return_val : str
):
    """Detects web annotations given an image."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    response = client.web_detection(image=image)
    annotations = response.web_detection

    images = []
    description = ""
    if annotations.best_guess_labels:
        for label in annotations.best_guess_labels:
            description = '\nBest guess label: {}'.format(label.label)

    if annotations.pages_with_matching_images:
        description = '\n{} Pages with matching images found:'.format(
            len(annotations.pages_with_matching_images))

        for page in annotations.pages_with_matching_images:
            description += '\n\tPage url   : {}\n'.format(page.url)

            if page.full_matching_images:
                description += '\t{} Full Matches found: \n'.format(
                       len(page.full_matching_images))

                for image in page.full_matching_images:
                    description += '\t\tImage url  : {}\n'.format(image.url)

            if page.partial_matching_images:
                description += '\t{} Partial Matches found: \n'.format(
                       len(page.partial_matching_images))

                for image in page.partial_matching_images:
                    description += '\t\tImage url  : {}\n'.format(image.url)

    if annotations.web_entities:
        # print('\n{} Web entities found: '.format(
        #     len(annotations.web_entities)))
        description += f'\n{len(annotations.web_entities)} Web entities found: '

        for entity in annotations.web_entities:
            description += '\n\tScore      : {}\n'.format(entity.score)
            description += u'\tDescription: {}\n'.format(entity.description)
            # print('\n\tScore      : {}'.format(entity.score))
            # print(u'\tDescription: {}'.format(entity.description))

    if annotations.visually_similar_images:
        description += '\n{} visually similar images found:\n'.format(
            len(annotations.visually_similar_images))

        for image in annotations.visually_similar_images:
            description += '\tImage url    : {}\n'.format(image.url)
            images.append(image.url)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    print(len(images))
    print(images)
    images = filter(lambda x:x.startswith(("http", "https")), images)
    return_val[3] = images
    return_val[4] = description

def run_flow(image_path):
    # poor attemt to mult-thread but works
    retval = [None,None,None,None,None, None]
    
    classification_labels_t = Thread(target=predict_image_classification_sample, args=(image_path, retval))
    image_objects_t = Thread(target=detect_labels,args=(image_path,retval))
    detect_web_t = Thread(target=detect_web,args=(image_path,retval))
    detect_text_t = Thread(target=detect_text, args=(image_path, retval))
    detect_landmarks_t = Thread(target=detect_landmarks, args=(image_path,retval))

    detect_text_t.start()
    classification_labels_t.start()
    image_objects_t.start()
    detect_web_t.start()
    detect_landmarks_t.start()


    classification_labels_t.join()
    image_objects_t.join()
    detect_web_t.join()
    detect_text_t.join()
    detect_landmarks_t.join()
    
    return retval[0], retval[1],retval[5], retval[2], retval[3], retval[4]