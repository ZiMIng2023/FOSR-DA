from source_data import source_loader
from distance_based_target_data import target_loader
from SR2CNN import *
from discriminator import *


version='RELEASE'
feature_dim=3*128
num_class=4
model = getSR2CNN(num_class,feature_dim)
model_path='./complexmodel15.pth'.format(version)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))


# 初始化鉴别器
discriminator = LargeAdversarialNetwork(3*128).to(device)
# 定义优化器
scheduler = lambda step, initial_lr : inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)

optimizer_discriminator = OptimWithSheduler(optim.SGD(discriminator.parameters(), lr=5e-4, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)
optimizer_feature_extractor = OptimWithSheduler(optim.SGD(model.parameters(), lr=5e-5, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)


# 定义损失函数
criterion = BCELossForMultiClassification
num_epochs=500
# 对抗学习的训练循环
for epoch in range(num_epochs):
    for source_data,target_data in zip(source_loader,target_loader):
        # 获取数据，并将其送入GPU
        im_source, label_source = source_data
        im_target, label_target = target_data
        source_inputs, source_labels = im_source.to(device), label_source.to(device)
        target_inputs, target_labels = im_target.to(device), label_target.to(device)

        # 生成特征
        source_features = model.getSemantic(source_inputs)
        target_features = model.getSemantic(target_inputs)

        # 使用鉴别器对特征进行分类
        domain_prob_discriminator_1_source = discriminator.forward(source_features)
        domain_prob_discriminator_1_target = discriminator.forward(target_features)

        # 计算损失

        adv_loss = BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_source),
                                                 predict_prob=domain_prob_discriminator_1_source)
        adv_loss += BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_target),
                                                  predict_prob=1 - domain_prob_discriminator_1_target)
        # 更新鉴别器
        with OptimizerManager([optimizer_feature_extractor, optimizer_discriminator]):
            loss = adv_loss
            loss.backward(retain_graph=True)
            # 打印损失
        print(f'Epoch [{epoch + 1}/{num_epochs}], adv_Loss: {adv_loss.item()}')

torch.save(model.state_dict(), 'feature_extractor1.pth')
# 训练结束后清空CUDA缓存
torch.cuda.empty_cache()
