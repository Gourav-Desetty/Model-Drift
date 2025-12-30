import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from models.image_classifier.model_architecture import LeukemiaCNN

class ImageClassifier:
    def __init__(self, model_path='leukemia_model_densenet_121_01.pth') -> None:

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = LeukemiaCNN(num_classes=2, pretrained=False)

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image, class_name=["hem", "all"]):
        target_image = Image.open(image).convert('RGB')
        transformed_image = self.transform(target_image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output = self.model(transformed_image)
            probability = torch.softmax(output, dim=1)
            label = probability.argmax(dim = 1).item()
            confidence = probability.max().item()
        
        return {'class' : class_name[label], 'confidence' : confidence}

    def embeddings(self, image):
        img = Image.open(image).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            f = self.model.model.features(x)
            f = torch.relu(f)
            f = torch.nn.functional.adaptive_avg_pool2d(f, (1,1))

        return f.flatten(1).cpu().numpy().tolist()

if __name__ == "__main__":
    image_classifier = ImageClassifier()
    result = image_classifier.predict('11.bmp')
    print(result)