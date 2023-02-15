import torchvision
import torch.nn as nn

## LocNet Model with Resnet backbone with pretrained
class LocNet(nn.Module):
    def __init__(self,num_layers,image_channels,num_classes,bb_points,pretrained):
        super(LocNet,self).__init__()
        self.num_layers     = num_layers
        self.image_channels = image_channels
        self.num_classes    = num_classes
        self.bb_points      = bb_points

        self.backbone           = self.get_backbone(pretrained)
        self.in_features        = 512 if num_layers < 50 else 2048
        self.fc_classification  = self.get_classification_head()
        self.fc_bounding_box    = self.get_bbox_head()

    def forward(self,x):
        x = self.backbone(x)
        y_classification = self.fc_classification(x)
        y_bounding_box = self.fc_bounding_box(x)
        return y_classification,y_bounding_box
    
    def get_classification_head(self):
        classifiaction_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(in_features=1* 1* self.in_features, out_features=1 * 1 * self.in_features),
            nn.Dropout(p=0.2,inplace=True),
            nn.Linear(in_features=1 * 1 * self.in_features, out_features=self.num_classes)
        )
        return classifiaction_head
    
    def get_bbox_head(self):
        bbox_head = nn.Sequential(
            nn.Conv2d(in_channels=self.in_features,out_channels=2048,
                               kernel_size=(3,3), stride=1, padding=1,bias=True),
            nn.BatchNorm2d(num_features=2048),
            nn.ReLU(),
            nn.Conv2d(in_channels=2048,out_channels=2048,
                      kernel_size=(3,3), stride=1, padding=1,bias=True),
            nn.BatchNorm2d(num_features=2048),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),

            nn.Conv2d(in_channels=2048,out_channels=1024,
                               kernel_size=(3,3), stride=1, padding=1,bias=True),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),

            nn.Conv2d(in_channels=1024,out_channels=self.in_features,
                      kernel_size=(3,3), stride=1, padding=1,bias=True),
            nn.BatchNorm2d(num_features=self.in_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),

            nn.AdaptiveAvgPool2d((1,1)),

            nn.Flatten(),
            nn.Linear(in_features=1 * 1 * self.in_features, out_features=1 * 1 * self.in_features),
            nn.Dropout(p=0.2,inplace=True),
            nn.Linear(in_features=1 * 1 * self.in_features, out_features=self.bb_points)
        )
        return bbox_head

    def get_backbone(self,pretrained):
        if pretrained:
            weights,model = self.get_model_details(self.num_layers)
            pretrained_model = model(weights=weights.DEFAULT)
            for param in pretrained_model.parameters():
                param.requires_grad = False 
            layers = list(pretrained_model.children())[:8]
            backbone = nn.Sequential(*layers)
            return backbone
        else:
            pass
            # backbone =  self.custom_resnet(num_layers=self.num_layers,
            #                           block=ResidualBlock,
            #                           image_channels=self.image_channels,
            #                           num_classes=self.num_classes)
            # for c in backbone.children():
            #     if isinstance(c, nn.Conv2d):
            #         nn.init.xavier_uniform_(c.weight)
            # return backbone
        
    def get_model_details(self,num_layers):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                                     f'to be 18, 34, 50, 101, or 152 '
        resnet_models = {
            18 : [torchvision.models.ResNet18_Weights,torchvision.models.resnet18],
            34 : [torchvision.models.ResNet34_Weights,torchvision.models.resnet34],
            50 : [torchvision.models.ResNet50_Weights,torchvision.models.resnet50],
            101: [torchvision.models.ResNet101_Weights,torchvision.models.resnet101],
            152: [torchvision.models.ResNet152_Weights,torchvision.models.resnet152]
        }
        return resnet_models[num_layers]
