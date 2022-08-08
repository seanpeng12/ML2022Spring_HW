import torch
import torchvision

def dataloader(batch_size_train,batch_size_test):
  train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data', train=True, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ])),
    batch_size=batch_size_train, shuffle=True)

  test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data', train=False, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ])),
    batch_size=batch_size_test, shuffle=True)
  # print(len(train_loader))
  print(len(test_loader)) # output 10
  print(test_loader)

  examples = enumerate(test_loader) #將資料由 ['Spring', 'Summer'] 改成 [(0, 'Spring'), (1, 'Summer')]，回傳object
  # print(list(examples)) # list(可iterable object)-串列：此資料型態可存放多種不同資料型態資料 like:[1,3,"peter"]。Iterable/Modifiable

  batch_idx, (example_data, example_targets) = next(examples) # ???????????
      
  # print(example_data.shape) #we have 1000 examples of 28x28 pixels in grayscale
  return example_data, example_targets

def plot(example_data, example_targets):
  import matplotlib.pyplot as plt

  for i in range(24):
    plt.subplot(4,6,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])

  plt.show()

def main():
  # hyperParameter setting 
  n_epochs = 3
  batch_size_train = 64
  batch_size_test = 1000
  learning_rate = 0.01
  momentum = 0.5
  log_interval = 10

  random_seed = 1
  torch.backends.cudnn.enabled = False
  torch.manual_seed(random_seed)

  # dataSet & dataloader definition
  # example_data,example_targets = dataloader(batch_size_train,batch_size_test)

  # Plot the dataset
  # plot(example_data,example_targets)

  

if __name__ == '__main__':
    main()