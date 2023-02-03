from google.cloud import aiplatform

endpoints = aiplatform.Endpoint.list()

for endpoint in endpoints:
    if endpoint.display_name == "car-images-damage-detection-endpoint":
        resource_name = endpoint.resource_name
        region = resource_name.split("/")[3]
        print(resource_name.split("/")[-1])
        exit() 

print("ERROR: Endpoint was not found!!!")