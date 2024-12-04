import gradio as gr
import argparse
from inference_module import InferenceCustomIsIcModule
import torch
from torchvision import transforms as T
from custom_transform import remove_hair
import numpy as np
from PIL import Image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu') 
model = None
transform = None

def classify_image(image):
    global model
    global device
    global transform

    image = np.array(image)
    image = remove_hair(image)
    image = Image.fromarray(image)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
    
    pred = output.detach().cpu()[0][0]
    probabilities = [1 - pred, pred]
    class_names = ["benign", "malignant"]
    predictions = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    return predictions


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-path', type=str)
    parser.add_argument('-model', type=str)

    args = parser.parse_args()

    model_path = f'{args.path}/{args.model}.ckpt'

    model = InferenceCustomIsIcModule.load_from_checkpoint(model_path, strict=False)
    model.eval()

    transform = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.ToTensor(),
            T.Lambda(lambda x: (x * 2) - 1),
        ])

    iface = gr.Interface(
        fn=classify_image, 
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=2),
        title="ISIC2024 Classifier",).launch()