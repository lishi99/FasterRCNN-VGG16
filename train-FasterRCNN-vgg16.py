import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import vgg16
from torch.utils.data import DataLoader

# 加载 VGG16 模型并移除分类头
backbone = vgg16(weights='IMAGENET1K_V1').features
backbone.out_channels = 512  # VGG16 的输出通道数

# 创建 RPN anchor 生成器（用于不同尺寸的 anchors）
rpn_anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5
)

# 创建 ROI pooler
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

# 定义 Faster R-CNN 模型
model = FasterRCNN(
    backbone,
    num_classes=2,  # 示例中的二分类任务（背景 + 1 个目标类）
    rpn_anchor_generator=rpn_anchor_generator,
    box_roi_pool=roi_pooler
)

# 设置设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"运行设备：{device}")
model.to(device)

# 设置优化器
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# 生成固定输入数据
def generate_fixed_data(batch_size, image_size=(3, 224, 224), num_boxes=5):
    images = [torch.rand(image_size) for _ in range(batch_size)]
    targets = []
    for _ in range(batch_size):
        boxes = torch.rand((num_boxes, 4)) * image_size[1]
        boxes[:, 2:] += boxes[:, :2]
        labels = torch.ones((num_boxes,), dtype=torch.int64)
        targets.append({"boxes": boxes, "labels": labels})
    return images, targets


def train():
    # 训练循环
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        batch_size = 4

        # 使用固定的数值输入进行训练
        for _ in range(2):  # 假设每个 epoch 有 100 个批次
            images, targets = generate_fixed_data(batch_size)
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 前向传播与计算损失
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            # 反向传播
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/100}")  # 输出平均损失



train()

