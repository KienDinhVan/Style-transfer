import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import os

class StyleTransferModel:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.imsize = 256
        self.content_weight = 1
        self.style_weight = 1e6
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
        # Initialize VGG19
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad_(False)
            
        # Loss functions
        self.content_loss_fn = nn.MSELoss()
        self.style_loss_fn = nn.MSELoss()
        
        # Image transforms
        self.img_transforms = transforms.Compose([
            transforms.Resize((self.imsize, self.imsize)),
            transforms.ToTensor()
        ])

    def load_image(self, image_path):
        if isinstance(image_path, str):
            image = Image.open(image_path)
        else:
            image = Image.open(image_path)
        image = self.img_transforms(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def gram_matrix(self, tensor):
        a, b, c, d = tensor.size()
        tensor = tensor.view(a * b, c * d)
        G = torch.mm(tensor, tensor.t())
        return G.div(a * b * c * d)

    def get_features(self, image):
        layers = {
            '0': 'conv_1',
            '5': 'conv_2',
            '10': 'conv_3',
            '19': 'conv_4',
            '28': 'conv_5'
        }
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def transfer_style(self, content_img, style_img, num_steps=1000, save_path=None):
        # Get features
        content_features = self.get_features(content_img)
        style_features = self.get_features(style_img)
        
        # Initialize target image
        target_img = content_img.clone().requires_grad_(True).to(self.device)
        optimizer = torch.optim.Adam([target_img], lr=0.02)

        for step in range(num_steps):
            optimizer.zero_grad()
            target_features = self.get_features(target_img)

            # Content loss
            content_loss = self.content_loss_fn(
                content_features['conv_1'], 
                target_features['conv_1']
            )

            # Style loss
            style_loss = 0
            for layer in self.style_layers:
                target_gram = self.gram_matrix(target_features[layer])
                style_gram = self.gram_matrix(style_features[layer])
                style_loss += self.style_loss_fn(style_gram, target_gram)

            # Total loss
            total_loss = self.content_weight * content_loss + self.style_weight * style_loss
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                target_img.clamp_(0, 1)

            if step % 100 == 0:
                print(f"Step [{step+1}/{num_steps}] total loss: {total_loss.item():.8f}")

        if save_path:
            self.save_model(target_img, save_path)
            
        return target_img

    def save_model(self, target_img, path):
        """Save the model state"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'target_img': target_img.detach(),
            'content_weight': self.content_weight,
            'style_weight': self.style_weight,
        }, path)

    def load_model(self, path):
        """Load the model state"""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            return checkpoint['target_img'].to(self.device)
        return None

    def transform_image(self, content_path, style_path, checkpoint_path=None):
        """Main method to be called from the Streamlit app"""
        content_img = self.load_image(content_path)
        style_img = self.load_image(style_path)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            return self.load_model(checkpoint_path)
        
        result = self.transfer_style(
            content_img, 
            style_img, 
            num_steps=2000,
        )
        
        return result