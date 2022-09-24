# 모델 불러오기
model = Efficientnet().to(DEVICE)
model.load_state_dict(torch.load("/AXR/efficientnet.pth"))

# test dataset
class TestDataset(data.Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform

        self.path = path

        self.img_h = 512
        self.img_w = 512

    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):
        # index번째 AXR 로드
        img_path = self.path[index]
        img_origin = pydicom.read_file(img_path)
        img_arr = img_origin.pixel_array
        img_re = cv2.resize(img_arr, (self.img_h,self.img_w), interpolation=cv2.INTER_LINEAR)
        img = img_re.astype(np.float32)

        # transform
        img_transformed = self.transform(img)

        return img_transformed

# test dataset loading
test_path = ["/AXR/ileus_1.dcm", "/AXR/normal_1.dcm"]

test_dataset = TestDataset(test_path, transform=ImageTransform())

test_dataloader = torch.utils.data.DataLoader(test_dataset)

# inference
model.eval()
for test in test_dataloader:
    test = test.to(DEVICE)
    output = model(test)
    print(output)
    output_n = output.cpu().detach().numpy()
    print(np.argmax(output_n))
