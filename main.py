from masked_model import *
from testing import *


if __name__ == '__main__':
	#Load the image data in
	train_loader, test_loader = retrieveData()

	net = Net()
	net.to(device)

	#defining a loss function
	criterion = nn.CrossEntropyLoss()

	#defining an optimiser
	optimiser = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	#epochs = input("Enter number of epochs: ")
	train(net, criterion, optimiser, train_loader, 2)
	print('Training Complete')


	conf_matrix = confusion_matrix(net, test_loader)

	p = Precision(conf_matrix[0][0], conf_matrix[1][0])

	r = Recall(conf_matrix[0][0], conf_matrix[0][1])

	f1 = F1(conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0])

	print("\nTrue Positives: " + str(conf_matrix[0][0]) + "\nFalse Positives: " + str(conf_matrix[0][1]) + "\nFalse Negatives: " + str(conf_matrix[1][0]) + "\nTrue Negatives: " + str(conf_matrix[1][1]))
	print("Precision: " + str(p))
	print("Recall: " + str(r))
	print("F1: " + str(f1))