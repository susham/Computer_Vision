import torch
print (torch.__version__)

import torchvision
print (torchvision.__version__)

import matplotlib.pyplot
import numpy
import tqdm

dblTrain = []
dblValidation = []

# creating a data loader for the training samples of the mnist dataset
# specifying the batch size as well as the normalization transform
# it will fetch and process the data in a separate background thread

objectTrain = torch.utils.data.DataLoader(
	batch_size=64,
	shuffle=True,
	num_workers=1,
	pin_memory=True,
	dataset=torchvision.datasets.MNIST(
		root='./mnist/',
		train=True,
		download=True,
		transform=torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(tuple([ 0.1307 ]), tuple([ 0.3081 ]))
		])
	)
)

# creating a data loader for the validation samples of the mnist dataset

objectValidation = torch.utils.data.DataLoader(
	batch_size=64,
	shuffle=True,
	num_workers=1,
	pin_memory=True,
	dataset=torchvision.datasets.MNIST(
		root='./mnist/',
		train=False,
		download=True,
		transform=torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(tuple([ 0.1307 ]), tuple([ 0.3081 ]))
		])
	)
)

# visualizing some samples and their labels from the validation set

objectFigure, objectAxis = matplotlib.pyplot.subplots(2, 4)

for objectRow in objectAxis:
	for objectCol in objectRow:
		tensorInput, tensorTarget = next(iter(objectValidation))

		objectCol.grid(False)
		objectCol.set_title(tensorTarget[0])
		objectCol.imshow(tensorInput[0].permute(1, 2, 0).squeeze(), cmap='gray')
	# end
# end

matplotlib.pyplot.show()

# defining the network depending on one of the discussed types
# afterwards creating an instance of it and making it cuda aware

strType = 'mlp'

if strType == 'mlp':
	class Network(torch.nn.Module):
		def __init__(self):
			super(Network, self).__init__()

			self.fc1 = torch.nn.Linear(784, 15)
			self.fc2 = torch.nn.Linear(15, 10)
		# end

		def forward(self, x):
			x = x.view(-1, 784)
			x = self.fc1(x)
			x = torch.nn.functional.sigmoid(x)
			x = self.fc2(x)

			return torch.nn.functional.log_softmax(x, dim=1)
		# end
	# end

elif strType == 'cnn':
	class Network(torch.nn.Module):
		def __init__(self):
			super(Network, self).__init__()

			self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5)
			self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5)
			self.fc1 = torch.nn.Linear(256, 200)
			self.fc2 = torch.nn.Linear(200, 10)
		# end

		def forward(self, x):
			x = self.conv1(x)
			x = torch.nn.functional.relu(x)
			x = torch.nn.functional.max_pool2d(x, kernel_size=3)
			x = self.conv2(x)
			x = torch.nn.functional.relu(x)
			x = torch.nn.functional.max_pool2d(x, kernel_size=2)
			x = x.view(-1, 256)
			x = self.fc1(x)
			x = torch.nn.functional.relu(x)
			x = self.fc2(x)

			return torch.nn.functional.log_softmax(x, dim=1)
		# end
	# end

elif strType == 'allconv':
	class Network(torch.nn.Module):
		def __init__(self):
			super(Network, self).__init__()

			self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5)
			self.conv1_pool = torch.nn.Conv2d(32, 32, kernel_size=3, stride=3)
			self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5)
			self.conv2_pool = torch.nn.Conv2d(64, 64, kernel_size=2, stride=2)
			self.conv3 = torch.nn.Conv2d(64, 200, kernel_size=2)
			self.conv4 = torch.nn.Conv2d(200, 10, kernel_size=1)
		# end

		def forward(self, x):
			x = self.conv1(x)
			x = torch.nn.functional.relu(x)
			x = self.conv1_pool(x)
			x = self.conv2(x)
			x = torch.nn.functional.relu(x)
			x = self.conv2_pool(x)
			x = self.conv3(x)
			x = torch.nn.functional.relu(x)
			x = self.conv4(x)

			return torch.nn.functional.log_softmax(x, dim=1).squeeze()
		# end
	# end

elif strType == 'dropout':
	class Network(torch.nn.Module):
		def __init__(self):
			super(Network, self).__init__()

			self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5)
			self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5)
			self.fc1 = torch.nn.Linear(256, 200)
			self.fc2 = torch.nn.Linear(200, 10)
		# end

		def forward(self, x):
			x = self.conv1(x)
			x = torch.nn.functional.relu(x)
			x = torch.nn.functional.max_pool2d(x, kernel_size=3)
			x = self.conv2(x)
			x = torch.nn.functional.relu(x)
			x = torch.nn.functional.max_pool2d(x, kernel_size=2)
			x = x.view(-1, 256)
			x = self.fc1(x)
			x = torch.nn.functional.dropout(x, p=0.25, training=self.training)
			x = torch.nn.functional.relu(x)
			x = self.fc2(x)

			return torch.nn.functional.log_softmax(x, dim=1)
		# end
	# end

elif strType == 'batchnorm':
	class Network(torch.nn.Module):
		def __init__(self):
			super(Network, self).__init__()

			self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5)
			self.conv1_bn = torch.nn.BatchNorm2d(32)
			self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5)
			self.conv2_bn = torch.nn.BatchNorm2d(64)
			self.fc1 = torch.nn.Linear(256, 200)
			self.fc2 = torch.nn.Linear(200, 10)
		# end

		def forward(self, x):
			x = self.conv1(x)
			x = self.conv1_bn(x)
			x = torch.nn.functional.relu(x)
			x = torch.nn.functional.max_pool2d(x, kernel_size=3)
			x = self.conv2(x)
			x = self.conv2_bn(x)
			x = torch.nn.functional.relu(x)
			x = torch.nn.functional.max_pool2d(x, kernel_size=2)
			x = x.view(-1, 256)
			x = self.fc1(x)
			x = torch.nn.functional.relu(x)
			x = self.fc2(x)

			return torch.nn.functional.log_softmax(x, dim=1)
		# end
	# end

# end

moduleNetwork = Network().cuda()

# specifying the optimization algorithm, stochastic gradient descent
# it will be responsible for updating the parameters of the network

objectOptimizer = torch.optim.SGD(params=moduleNetwork.parameters(), lr=0.01, momentum=0.5)

def train():
	# setting the network to the training mode, some modules behave differently during training

	moduleNetwork.train()

	# obtain samples and their ground truth from the training dataset, one minibatch at a time

	for tensorInput, tensorTarget in tqdm.tqdm(objectTrain):
		# wrapping the loaded tensors into variables, allowing them to have gradients
		# in the future, pytorch will combine tensors and variables into one type
		# the variables are set to be not volatile such that they retain their history

		variableInput = torch.autograd.Variable(data=tensorInput, volatile=False).cuda()
		variableTarget = torch.autograd.Variable(data=tensorTarget, volatile=False).cuda()

		# setting all previously computed gradients to zero, we will compute new ones

		objectOptimizer.zero_grad()

		# performing a forward pass through the network while retaining a computational graph

		variableEstimate = moduleNetwork(variableInput)

		# computing the loss according to the cross-entropy / negative log likelihood
		# the backprop is done in the subsequent step such that multiple losses can be combined

		variableLoss = torch.nn.functional.nll_loss(input=variableEstimate, target=variableTarget)

		variableLoss.backward()

		# calling the optimizer, allowing it to update the weights according to the gradients

		objectOptimizer.step()
	# end
# end

def evaluate():
	# setting the network to the evaluation mode, some modules behave differently during evaluation

	moduleNetwork.eval()

	# defining two variables that will count the number of correct classifications

	intTrain = 0
	intValidation = 0

	# iterating over the training and the validation dataset to determine the accuracy
	# this is typically done one a subset of the samples in each set, unlike here
	# otherwise the time to evaluate the model would unnecessarily take too much time

	for tensorInput, tensorTarget in objectTrain:
		variableInput = torch.autograd.Variable(data=tensorInput, volatile=True).cuda()
		variableTarget = torch.autograd.Variable(data=tensorTarget, volatile=True).cuda()

		variableEstimate = moduleNetwork(variableInput)

		intTrain += variableEstimate.data.max(dim=1, keepdim=False)[1].eq(variableTarget.data).sum()
	# end

	for tensorInput, tensorTarget in objectValidation:
		variableInput = torch.autograd.Variable(data=tensorInput, volatile=True).cuda()
		variableTarget = torch.autograd.Variable(data=tensorTarget, volatile=True).cuda()

		variableEstimate = moduleNetwork(variableInput)

		intValidation += variableEstimate.data.max(dim=1, keepdim=False)[1].eq(variableTarget.data).sum()
	# end

	# determining the accuracy based on the number of correct predictions and the size of the dataset

	dblTrain.append(100.0 * intTrain / len(objectTrain.dataset))
	dblValidation.append(100.0 * intValidation / len(objectValidation.dataset))

	print('')
	print('train: ' + str(intTrain) + '/' + str(len(objectTrain.dataset)) + ' (' + str(dblTrain[-1]) + '%)')
	print('validation: ' + str(intValidation) + '/' + str(len(objectValidation.dataset)) + ' (' + str(dblValidation[-1]) + '%)')
	print('')
# end

# training the model for 100 epochs, one would typically save / checkpoint the model after each one

for intEpoch in range(100):
	train()
	evaluate()
# end

# plotting the learning curve according to the accuracies determined in the evaluation function

matplotlib.pyplot.figure(figsize=(4.0, 5.0), dpi=150.0)
matplotlib.pyplot.ylim(93.5, 100.5)
matplotlib.pyplot.plot(dblTrain)
matplotlib.pyplot.plot(dblValidation)
matplotlib.pyplot.show()